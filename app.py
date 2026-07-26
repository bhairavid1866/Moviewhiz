import streamlit as st
from dotenv import load_dotenv
import os
import requests
import re

from langchain_groq import ChatGroq

from htmlTemplates import css, bot_template, user_template

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Filler phrases stripped from the question before using it as an OMDb search term.
FILLER_PATTERN = re.compile(
    r"\b(tell me about|what is|what's|recommend|suggest|do you know|can you|"
    r"i want to know about|info(?:rmation)? (?:on|about)|the movie[s]?)\b",
    re.IGNORECASE,
)


def extract_search_terms(question):
    """Strip common filler phrases so we're left with a usable OMDb search term."""
    cleaned = FILLER_PATTERN.sub("", question)
    cleaned = cleaned.strip(" ?.!,")
    return cleaned if cleaned else question


def fetch_movies(query, limit=5):
    """Search OMDb, then fetch full details (plot, rating) for each match."""
    search_url = f"https://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={query}&type=movie"
    search_response = requests.get(search_url)
    search_data = search_response.json()

    if search_data.get("Response") != "True":
        return []

    movies = []
    for item in search_data.get("Search", [])[:limit]:
        detail_url = f"https://www.omdbapi.com/?apikey={OMDB_API_KEY}&i={item['imdbID']}&plot=short"
        detail_response = requests.get(detail_url)
        detail_data = detail_response.json()
        if detail_data.get("Response") == "True":
            movies.append(detail_data)
    return movies


# Groq-hosted LLM. Swap model to any Groq-supported model, e.g. "llama-3.3-70b-versatile".
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
)


def handle_userinput(user_question):
    search_terms = extract_search_terms(user_question)
    movies = fetch_movies(search_terms)

    if movies:
        context = "\n".join(
            f"- {m['Title']} ({m.get('Year', 'N/A')}): {m.get('Plot', 'No plot available.')} "
            f"[IMDb rating: {m.get('imdbRating', 'N/A')}]"
            for m in movies
        )
    else:
        context = "No matching movies were found in OMDb for this question."

    history_text = "\n".join(
        f"{'User' if m['role'] == 'human' else 'Assistant'}: {m['content']}"
        for m in st.session_state.chat_history
    )

    prompt = (
        "You are a helpful movie recommendation assistant. Use the movie data below to answer "
        "the user's question. Only reference movies from this data; if it doesn't contain what "
        "the user asked about, say so honestly.\n\n"
        f"Movie data:\n{context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User's new question: {user_question}"
    )

    response_text = llm.invoke(prompt).content

    st.session_state.chat_history.append({"content": user_question, "role": "human"})
    st.session_state.chat_history.append({"content": response_text, "role": "assistant"})

    for message in st.session_state.chat_history:
        if message["role"] == "human":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

    if movies:
        for movie in movies:
            st.write(f"{movie['Title']} ({movie.get('Year', 'N/A')})")
            st.write(f"Rating: {movie.get('imdbRating', 'N/A')}/10")
            st.write(f"Overview: {movie.get('Plot', 'N/A')}")
            st.write(f"[Watch Now](https://www.imdb.com/title/{movie['imdbID']}/)")
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
