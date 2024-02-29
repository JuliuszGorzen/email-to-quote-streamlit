import streamlit as st
import streamlit_authenticator as stauth
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from yaml.loader import SafeLoader

from constants import HIDE_SIDEBAR_HTML, DISPLAY_SIDEBAR_HTML, HIDE_STREAMLIT_ELEMENTS


def main():
    # Set page configuration
    create_page_config()

    # Hide Streamlit main menu, header and footer
    st.markdown(HIDE_STREAMLIT_ELEMENTS, unsafe_allow_html=True)

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
        "Form name": "Login :closed_lock_with_key:",
        "Username": "username",
        "Password": "password",
        "Login": "Enter to AI world :mechanical_arm::mechanical_leg:"
    })

    if st.session_state["authentication_status"]:
        # Display expandable menu
        st.markdown(DISPLAY_SIDEBAR_HTML, unsafe_allow_html=True)

        # Add a logout button
        authenticator.logout(
            button_name="Leave AI world :disappointed_relieved::broken_heart:",
            location="sidebar"
        )

        # ---- MAIN PAGE ----
        st.title("Email to Quote :e-mail::arrow_right::moneybag:")

        with st.expander("**:rainbow[How to start?]** :thinking_face:"):
            st.markdown(read_md_file("markdowns/main-page-description.md"))

        # ---- TABS ----
        zero_shot_prompting_tab, few_shot_prompting_tab, ner_zero_shot_prompting, ner_few_shot_prompting, rag = st.tabs(
            [
                "Zero-shot Prompting :zero::gun:",
                "Few-shot Prompting :1234::gun:",
                "NER + Zero-shot Prompting :writing_hand::heavy_plus_sign::zero::gun:",
                "NER + Few-shot Prompting :writing_hand::heavy_plus_sign::1234::gun:",
                "RAG :bookmark_tabs:"
            ])

        with zero_shot_prompting_tab:
            st.header(":orange[Zero-shot Prompting] :zero::gun:")
            st.markdown(read_md_file("markdowns/zero-shot-prompting-description.md"))

            with st.expander("**See example** :eyes:"):
                st.markdown(read_md_file("markdowns/zero-shot-prompting-example.md"))

            with st.form("zero_shot_prompting_form"):
                st.header("Try Zero-shot Prompting")
                user_input_val = st.text_area(
                    label="Enter email here:",
                    placeholder="""Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days"""
                )
                prompt_val = st.text_area(
                    label="Enter your prompt here:",
                    placeholder="""System:
You are a bot that extract data from emails in different languages. Below are the rules that you have to follow:
- You can only write valid JSONs based on the documentation below:

{"from_address": "string", "to_address": "string"}

- Your goal is to transform email input into structured JSON. Comments are not allowed in the JSON. Text outside the 
JSON is strictly forbidden.
- Users provide the email freight orders as input, which you will transform into JSON format given the JSON 
documentation.
- You can enhance the output with general common-knowledge facts about the world relevant to the procurement event.
- If you cannot find a piece of information, you can leave the corresponding attribute as "".

User:
{email}

Assistant:"""
                )
                submitted = st.form_submit_button(label="Sent prompt to model :rocket:")

                if submitted:
                    if user_input_val == "":
                        user_input_val = """Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days"""
                    if prompt_val == "":
                        prompt_val = """System:
You are a bot that extract data from emails in different languages. Below are the rules that you have to follow:
- You can only write valid JSONs based on the documentation below:
```
("from_address": "string", "to_address": "string")
```
- Your goal is to transform email input into structured JSON. Comments are not allowed in the JSON. Text outside the JSON is strictly forbidden.
- Users provide the email freight orders as input, which you will transform into JSON format given the JSON documentation.
- You can enhance the output with general common-knowledge facts about the world relevant to the procurement event.
- If you cannot find a piece of information, you can leave the corresponding attribute as "".

User:
"
FROM: Juliusz Gorzen
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

I would like to book a FTL transport from Safranberg 123, 12345 Ulm to
Wietrzna 34, Wroclaw 52-023, Poland next Monday.

Thanks.

Sincerely,
Juliusz
"

Assistant:
(from_address": "Safranberg 123, 12345 Ulm", "to_address": "Wietrzna 34, Wroclaw 52-023, Poland")

User:
"
FROM: Abc def
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

I would like to book a magic transport from 3486 Tuna Street, 48302, Bloomfield Township to
1011 Franklin Avenue, Daytona Beach 32114, US on 2024-03-03 10 pm.

Thanks you.

BR,
Abc
"

Assistant:
("from_address": "3486 Tuna Street, 48302, Bloomfield Township", "to_address": "1011 Franklin Avenue, Daytona Beach 32114, US")

User: 
"
FROM: John Taylor
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

Pleas book the transport.
From: 2132 Thomas Street, Wheeling, US
To: -
Date: 2024-02-11 09:30
"

Assistant:
("from_address": "2132 Thomas Street, Wheeling, US", "to_address": "")

User:
{email}

Assistant:"""
                    llm = AzureChatOpenAI(
                        azure_endpoint="https://open-ai-resource-gen-ai.openai.azure.com/",
                        openai_api_version="2023-07-01-preview",
                        openai_api_key="",
                        openai_api_type="azure",
                        deployment_name="gpt-35-dev",
                        model_name="gpt-35-turbo",
                        model_version="0613",
                        temperature=0.4
                    )
                    prompt = ChatPromptTemplate.from_template(prompt_val)

                    chain = prompt | llm

                    with st.spinner('Generating response...'):
                        response = chain.invoke({"email": user_input_val})
                    st.toast('Done!', icon='😍')
                    st.markdown("#### Bot response:")
                    st.text(response.content)

        with few_shot_prompting_tab:
            st.header("Few-shot Prompting :1234::gun:")

            with st.expander("See example :eyes:"):
                st.markdown(read_md_file("markdowns/few-shot-prompting-description.md"))

            with st.form("few_shot_prompting_form"):
                st.header("Try Few-shot Prompting")
                user_input = st.text_input(
                    label="Enter your prompt here:",
                    value="Some example email."
                )
                submitted = st.form_submit_button(
                    label="Sent prompt to model :rocket:",
                    disabled=True
                )

                if submitted:
                    st.write("You entered:", user_input)

        with ner_zero_shot_prompting:
            st.header("NER + Zero-shot Prompting :writing_hand::heavy_plus_sign::zero::gun:")

            with st.expander("See example :eyes:"):
                st.markdown(read_md_file("markdowns/ner-zero-shot-prompting-description.md"))

            with st.form("ner_zero_shot_prompting_form"):
                st.header("Try NER + Zero-shot Prompting")
                user_input = st.text_input(
                    label="Enter your prompt here:",
                    value="Some example email."
                )
                submitted = st.form_submit_button(
                    label="Sent prompt to model :rocket:",
                    disabled=True
                )

                if submitted:
                    st.write("You entered:", user_input)

        with ner_few_shot_prompting:
            st.header("NER + Few-shot Prompting :writing_hand::heavy_plus_sign::1234::gun:")

            with st.expander("See example :eyes:"):
                st.markdown(read_md_file("markdowns/ner-few-shot-prompting-description.md"))

            with st.form("ner_few_shot_prompting_form"):
                st.header("Try NER + Few-shot Prompting")
                user_input = st.text_input(
                    label="Enter your prompt here:",
                    value="Some example email."
                )
                submitted = st.form_submit_button(
                    label="Sent prompt to model :rocket:",
                    disabled=True
                )

                if submitted:
                    st.write("You entered:", user_input)

        with rag:
            st.header("RAG :bookmark_tabs:")

            with st.expander("See example :eyes:"):
                st.markdown(read_md_file("markdowns/rag-description.md"))

            with st.form("rag_form"):
                st.header("Try RAG")
                user_input = st.text_input(
                    label="Enter your prompt here:",
                    value="Some example email."
                )
                submitted = st.form_submit_button(
                    label="Sent prompt to model :rocket:",
                    disabled=True
                )

                if submitted:
                    st.write("You entered:", user_input)

        # ---- SIDEBAR ----
        st.sidebar.header("Model parameters:")
        st.sidebar.subheader("Here you can set the parameters for the model.")

        with st.sidebar.form("model_parameters_form"):
            st.write("Some parameters are disabled as for now we have only one LLM.")

            # Form fields
            azure_endpoint_val = st.text_input(
                label="Azure endpoint",
                value="https://open-ai-resource-gen-ai.openai.azure.com/",
                type="password",
                disabled=True
            )
            openai_api_version_val = st.text_input(
                label="OpenAI API version",
                value="2023-07-01-preview",
                disabled=True
            )
            openai_api_key_val = st.text_input(
                label="OpenAI API key",
                placeholder="<api-key>",
                type="password",
                help="You can find your API key in the Azure or contact the Generative AI team."
            )

            if openai_api_key_val == "<api-key>" or openai_api_key_val == "":
                st.warning("Please enter your OpenAI API key.")

            openai_api_type_val = st.text_input(
                label="OpenAI API type",
                value="azure",
                disabled=True
            )
            deployment_name_val = st.text_input(
                label="Deployment name",
                value="gpt-35-dev",
                disabled=True
            )
            model_name_val = st.text_input(
                label="Model name",
                value="gpt-35-turbo",
                disabled=True
            )
            model_version_val = st.text_input(
                label="Model version",
                value="0613",
                disabled=True
            )
            temperature_slider_val = st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="A higher temperature value typically makes the output more diverse and creative but might also "
                     "increase its likelihood of straying from the context. Conversely, a lower temperature value "
                     "makes the AI's responses more focused and deterministic, sticking closely to the most likely "
                     "prediction."
            )

            # Submit button
            submitted = st.form_submit_button(
                label="Save :white_check_mark:",
                on_click=st.balloons,
                use_container_width=True
            )

            if submitted:
                st.write("Model parameters saved with the following values:")
                st.write("Temperature:", temperature_slider_val)
                st.write("OpenAI API key:", openai_api_key_val)

        with st.sidebar.expander("Model description :pencil2:"):
            st.markdown(read_md_file("markdowns/model-description.md"))

    elif st.session_state["authentication_status"] is False:
        # Display an error message if the user enters the wrong credentials
        st.error('Username or/and password is incorrect. Try again.')
    elif st.session_state["authentication_status"] is None:
        # Display a warning message if the user does not enter the credentials/leaves the fields empty
        st.warning('Please enter your username and password.')


def create_page_config():
    st.set_page_config(
        page_title="Email to Quote 📧➡️💰",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def read_md_file(file_path: str) -> str:
    return open(file_path, encoding="utf8").read()


if __name__ == "__main__":
    main()
