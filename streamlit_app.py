import streamlit as st
from st_pages import Page, show_pages

show_pages(
    [
        Page("pages/home.py", "🏠 Start here"),
        Page("pages/one_shot_prompting.py", "1️🔫 One-shot Prompting"),
        Page("pages/few_shot_prompting.py", "🎲🔫 Few-shot Prompting"),
        Page("pages/ner_one_shot_prompting.py", "📑➕1️🔫 NER + One-shot Prompting"),
        Page("pages/ner_few_shot_prompting.py", "📑➕🎲🔫 NER + Few-shot Prompting"),
        Page("pages/rag.py", "️⚒️ RAG")
    ]
)

st.switch_page("pages/home.py")
