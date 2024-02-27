import streamlit as st
import streamlit_authenticator as stauth
import yaml
from st_pages import Page, show_pages
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="Email to Quote - Streamlit App",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide expandable menu
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

with open('authentication.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    # config['preauthorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: grid
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    show_pages(
        [
            Page("streamlit_app.py", "🏠 Home"),
            Page("pages/one_shot_prompting.py", "1️🔫 One-shot Prompting"),
            Page("pages/few_shot_prompting.py", "🎲🔫 Few-shot Prompting"),
            Page("pages/ner_one_shot_prompting.py", "📑➕1️🔫 NER + One-shot Prompting"),
            Page("pages/ner_few_shot_prompting.py", "📑➕🎲🔫 NER + Few-shot Prompting"),
            Page("pages/rag.py", "️⚒️ RAG")
        ]
    )
    authenticator.logout(location="sidebar")
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
