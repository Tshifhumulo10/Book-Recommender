import streamlit as st 
import pandas as pd
import numpy as np
# Entity featurization and similarity computation
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
import random


st.title("Book Recommender Application")
books=pd.read_csv("Book.csv" , encoding="utf-8")
books=books.drop_duplicates()
st.subheader("Content Based recommender")
books_list = list(books['Book'])
selected_option = st.selectbox('Select a book you have read and  liked', options=books_list)
selected_number = st.selectbox('Select a number of books you want to be recommended', range(1, 30))
button_clicked = st.button("Recommend", key="predict_button", kwargs={"style": "background-color: red; color: red"})

if button_clicked:
    with st.spinner('Loading...'):
        def combine_columns(row):
            return f"{row['Author']}, {row['Description']}, {', '.join(row['Genres'])}, {row['Avg_Rating']}, {row['Num_Ratings']}"
        books['book info'] = books.apply(combine_columns, axis=1)
        titles=books['Book']
        indices=pd.Series(books.index ,index=books['Book'])
        #Building a vectorizer
        t_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,4),min_df=0.0, stop_words='english')
        #Fitting the vectorizer
        tf_bookinfo = t_vec.fit_transform(books['book info'])
        cosine_similarity_info = cosine_similarity(tf_bookinfo, tf_bookinfo)
        def content_based_recommender(Book_Name, n =5):
            book_index = indices[Book_Name]
            score = list(enumerate(cosine_similarity_info[book_index]))
            score = sorted(score, key=lambda x: x[1], reverse=True)
            score = score[1:n+1] 
            book_indices = [i[0] for i in score]
            book=titles.iloc[book_indices].reset_index()
            #books=list(enumerate(books, start=1))
            book_1=list(book['Book'])
            index_1=list(book['index'])
            rating=[]
            for i in index_1:
                rating.append(books.loc[i, "Avg_Rating"])
            author=[]
            for i in index_1:
                author.append(books.loc[i, "Author"])
            dictionary={"Book name": book_1, "Author": author, "Rating": rating}  
            df=pd.DataFrame(dictionary)
            return df
        
        Answer=content_based_recommender(selected_option,selected_number )
    st.write(Answer)

st.subheader('Random Selection')
selected_number = st.selectbox('Select a number of books you want to be randomly selected', range(1, 30))
button_clicked = st.button("Random Selection", key="Random _sample_button", kwargs={"style": "background-color: red; color: red"})
if button_clicked:
    with st.spinner('Loading...'):
        def select_random_numbers(selected_number):
            def random_selector(selected_number):
                books_range= list(range(0, 9936))
                selected_numbers = random.sample(books_range, selected_number)
                return(selected_numbers)
            results=random_selector(selected_number)
            book=[]
            for i in results:
                book.append(books.loc[i, "Book"])
        
            ratings=[] 
            for i in results:
                ratings.append(books.loc[i, "Avg_Rating"])
        
            author=[]
            for i in results:
                author.append(books.loc[i, "Author"])
        
            dictionary={"Book name": book, "Author": author, "Rating": ratings}  
            df=pd.DataFrame(dictionary)
            return (df)
        Answer=select_random_numbers(selected_number)
    st.write(Answer)       




