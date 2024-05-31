import streamlit as st
from dotenv import load_dotenv
import os
import requests
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import re


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_movies(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    response = requests.get(url)
    data = response.json()
    return data["results"]


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
movie_data = fetch_movies("movie")
texts = [f"{movie['title']} - {movie['overview']}" for movie in movie_data]
vectorstore = FAISS.from_texts(texts, embeddings)


llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)


movie_recommendation_pattern = r"(you\s*(should|could|might)\s*(watch|like|enjoy)\s*(the\s*movie[s]?)\s*(.+?)[,.\?])"

def handle_userinput(user_question):
    
    response = qa({"question": user_question, "chat_history": st.session_state.chat_history})

 
    st.session_state.chat_history.append({"content": user_question, "role": "human"})
    st.session_state.chat_history.append({"content": response["answer"], "role": "assistant"})

    for message in st.session_state.chat_history:
        if message["role"] == "human":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

    
    movie_recommendations = re.findall(movie_recommendation_pattern, response["answer"], re.IGNORECASE)
    if movie_recommendations:
        for recommendation in movie_recommendations:
            movies = fetch_movies(recommendation[-1])
            if movies:
                for movie in movies:
                    st.write(f"{movie['title']} ({movie['release_date'][:4]})")
                    st.write(f"Rating: {movie['vote_average']}/10")
                    st.write(f"Overview: {movie['overview']}")
                    st.write(f"[Watch Now](https://www.themoviedb.org/movie/{movie['id']})")
                    st.write("---")

def main():
    st.set_page_config(page_title="Movie Recommender", page_icon=":movie_camera:")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Moviewhiz :movie_camera:")
    st.write("Hello, I'm here to help you choose which movie to watch!")

    user_question = st.text_input("Ask me about movies you'd like to watch:")
    search_button = st.button("Search")

    if search_button:
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()