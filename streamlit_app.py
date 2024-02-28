import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from constants import HIDE_SIDEBAR_HTML, DISPLAY_SIDEBAR_HTML, HIDE_STREAMLIT_STYLE

# Set page configuration (only this page is affected by this configuration)
st.set_page_config(
    page_title="Email to Quote - Streamlit App",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit main menu, header and footer
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

# Hide expandable menu
st.markdown(HIDE_SIDEBAR_HTML, unsafe_allow_html=True)

# Load authentication credentials from .yaml file
with open('authentication.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# create an instance of the Authenticate class with the credentials
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Add login form
authenticator.login(fields={
    "Form name": "Login 🔐",
    "Username": "username",
    "Password": "password",
    "Login": "Enter to AI world 🤖"
})

if st.session_state["authentication_status"]:
    # Display expandable menu
    st.markdown(DISPLAY_SIDEBAR_HTML, unsafe_allow_html=True)

    # Add a logout button
    authenticator.logout(location="sidebar")

    # ---- MAIN PAGE ----
    st.title("Email to Quote - Streamlit App 📧➡️💰")
    st.markdown("""
    ---
    ⬅️ Use the sidebar to navigate through the app.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
    aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
    sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """)

    # ---- TABS ----
    zero_shot_prompting_tab, few_shot_prompting_tab, ner_zero_shot_prompting, ner_few_shot_prompting, rag = st.tabs([
        "Zero-shot Prompting 0️⃣🔫",
        "Few-shot Prompting 3️⃣🔫",
        "NER + Zero-shot Prompting ✍️➕0️⃣🔫",
        "NER + Few-shot Prompting ✍️➕3️⃣🔫",
        "RAG 📄"
    ])

    with zero_shot_prompting_tab:
        st.header("Zero-shot Prompting 0️⃣🔫")
        st.markdown("""
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
        aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
        sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
        """)

    with few_shot_prompting_tab:
        st.header("Few-shot Prompting 3️⃣🔫")
        st.markdown("""
                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
                aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
                Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
                sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                """)

    with ner_zero_shot_prompting:
        st.header("NER + Zero-shot Prompting ✍️➕0️⃣🔫")
        st.markdown("""
                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
                aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
                Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
                sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                """)

    with ner_few_shot_prompting:
        st.header("NER + Few-shot Prompting ✍️➕3️⃣🔫")
        st.markdown("""
                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
                aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
                Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
                sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                """)

    with rag:
        st.header("RAG 📄")
        st.markdown("""
                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
                aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
                Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur 
                sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                """)

    # ---- SIDEBAR ----
    st.sidebar.header("Model parameters:")
    st.sidebar.subheader("Here you can set the parameters for the model.")

    with st.sidebar.form("model_parameters_form"):
        st.write("Some parameters are disabled as for now we have only one LLM.")

        # Form fields
        st.text_input(label="Azure endpoint", value="https://open-ai-resource-gen-ai.openai.azure.com/", disabled=True)
        st.text_input(label="OpenAI API version", value="2023-07-01-preview", disabled=True)
        st.text_input(label="OpenAI API key", value="******", disabled=True)
        st.text_input(label="OpenAI API type", value="azure", disabled=True)
        st.text_input(label="Deployment name", value="gpt-35-dev", disabled=True)
        st.text_input(label="Model name", value="gpt-35-turbo", disabled=True)
        st.text_input(label="Model version", value="0613", disabled=True)

        slider_val = st.slider("Temperature", 0.0, 1.0, 0.1)

        # Submit button
        submitted = st.form_submit_button(label="Save", on_click=st.balloons)
        if submitted:
            st.write("Model parameters saved with the following values:")
            st.write("Temperature", slider_val)

elif st.session_state["authentication_status"] is False:
    # Display an error message if the user enters the wrong credentials
    st.error('Username or/and password is incorrect. Try again.')
elif st.session_state["authentication_status"] is None:
    # Display a warning message if the user does not enter the credentials/leaves the fields empty
    st.warning('Please enter your username and password.')
