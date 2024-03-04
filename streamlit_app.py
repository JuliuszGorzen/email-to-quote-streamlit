import json

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from yaml.loader import SafeLoader

import constants


def main() -> None:
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url="https://genai-dev-keyvault.vault.azure.net/", credential=credential)
    openai_api_key_secret = secret_client.get_secret("openai-api-key")
    llm = create_azure_openai_model(openai_api_key_secret, 0.2)

    create_page_config()
    remove_unused_html()

    authenticator = create_authenticator()
    create_login_page(authenticator)

    if st.session_state["authentication_status"]:
        # Display expandable menu
        st.markdown(constants.DISPLAY_SIDEBAR_HTML, unsafe_allow_html=True)

        create_logout_button(authenticator)
        create_main_page()
        create_sidebar()
        create_tabs(llm)
    elif st.session_state["authentication_status"] is False:
        st.error(constants.LOGIN_PAGE_ERROR_MESSAGE)
    elif st.session_state["authentication_status"] is None:
        st.warning(constants.LOGIN_PAGE_WARNING_MESSAGE)


def create_logout_button(authenticator) -> None:
    authenticator.logout(
        button_name=constants.LOGOUT_BUTTON_TEXT,
        location="sidebar"
    )


def create_login_page(authenticator) -> None:
    authenticator.login(fields={
        "Form name": constants.LOGIN_PAGE_NAME,
        "Username": "username",
        "Password": "password",
        "Login": constants.LOGIN_BUTTON_TEXT
    })


def create_main_page() -> None:
    st.title(constants.MAIN_PAGE_HEADER)
    with st.expander(constants.MAIN_PAGE_EXPANDER):
        st.markdown(read_md_file("markdowns/main-page-description.md"))


def create_sidebar() -> None:
    st.sidebar.header(constants.SIDEBAR_HEADER)
    st.sidebar.error(constants.SIDEBAR_ERROR_MESSAGE)
    st.sidebar.subheader(constants.SIDEBAR_SUBHEADER)

    with st.sidebar.form("model_parameters_form"):
        st.write(constants.SIDEBAR_FORM_DESCRIPTION)
        st.text_input(
            label=constants.SIDEBAR_FORM_AZURE_ENDPOINT,
            value="https://open-ai-resource-gen-ai.openai.azure.com/",
            type="password",
            disabled=True
        )
        st.text_input(
            label=constants.SIDEBAR_FORM_OPENAI_API_VERSION,
            value="2023-07-01-preview",
            disabled=True
        )
        openai_api_key_val = st.text_input(
            label=constants.SIDEBAR_FORM_OPENAI_API_KEY,
            placeholder="<api-key>",
            type="password",
            help=constants.SIDEBAR_FORM_OPENAI_API_KEY_HELP,
            disabled=True
        )

        # For now there is only one llm
        # if openai_api_key_val == "":
        #     st.warning(constants.SIDEBAR_FORM_OPENAI_API_KEY_WARNING)

        st.text_input(
            label=constants.SIDEBAR_FORM_OPENAI_API_TYPE,
            value="azure",
            disabled=True
        )
        st.text_input(
            label=constants.SIDEBAR_FORM_DEPLOYMENT_NAME,
            value="gpt-35-dev",
            disabled=True
        )
        st.text_input(
            label=constants.SIDEBAR_FORM_MODEL_NAME,
            value="gpt-35-turbo",
            disabled=True
        )
        st.text_input(
            label=constants.SIDEBAR_FORM_MODEL_VERSION,
            value="0613",
            disabled=True
        )
        temperature_slider_val = st.slider(
            label=constants.SIDEBAR_FORM_TEMPERATURE,
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help=constants.SIDEBAR_FORM_TEMPERATURE_HELP
        )
        submitted = st.form_submit_button(
            label=constants.SIDEBAR_FORM_SUBMIT_BUTTON,
            on_click=st.balloons,
            use_container_width=True,
            disabled=True
        )

        if submitted:
            st.write(constants.SIDEBAR_FORM_SUMMARY)
            st.write("Temperature:", temperature_slider_val)
            st.write("OpenAI API key:", openai_api_key_val)

    with st.sidebar.expander(constants.SIDEBAR_FORM_MODEL_DESCRIPTION):
        st.markdown(read_md_file("markdowns/model-description.md"))


def create_tabs(llm: AzureChatOpenAI) -> None:
    zero_shot_prompting_tab, few_shot_prompting_tab, ner_zero_shot_prompting, ner_few_shot_prompting, rag = st.tabs(
        [
            constants.TAB_NAME_ZERO_SHOT_PROMPTING,
            constants.TAB_NAME_FEW_SHOT_PROMPTING,
            constants.TAB_NAME_NER_ZERO_SHOT_PROMPTING,
            constants.TAB_NAME_NER_FEW_SHOT_PROMPTING,
            constants.TAB_NAME_RAG
        ])

    create_zero_shot_prompting_tab(zero_shot_prompting_tab, llm)
    create_few_shot_prompting_tab(few_shot_prompting_tab, llm)
    create_ner_zero_shot_prompting_tab(ner_zero_shot_prompting, llm)
    create_ner_few_shot_prompting_tab(ner_few_shot_prompting, llm)

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


def create_ner_few_shot_prompting_tab(ner_few_shot_prompting, llm):
    with ner_few_shot_prompting:
        st.header(constants.NER_FEW_SHOT_PROMPTING_TAB_HEADER)
        st.markdown(read_md_file("markdowns/ner-few-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/ner-few-shot-prompting-example.md"))

        with st.form("ner_few_shot_prompting_form"):
            st.header(constants.NER_FEW_SHOT_PROMPTING_TAB_FORM_HEADER)
            system_message = st.text_area(
                label=constants.TAB_FORM_SYSTEM_MESSAGE,
                placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                height=200
            )
            ner_message = st.text_area(
                label="NER (categories definition)",
                placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_CATEGORIES,
                height=200
            )

            st.subheader("First example")
            col1, col2 = st.columns(2)

            with col1:
                human_message_1 = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1,
                    height=200
                )

            with col2:
                ai_message_1 = st.text_area(
                    label=constants.TAB_FORM_AI_MESSAGE,
                    placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1,
                    height=200
                )

            st.subheader("Actual email")
            human_message_2 = st.text_area(
                label=constants.TAB_FORM_HUMAN_MESSAGE,
                placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2,
                height=200
            )
            submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

            if submitted:
                if is_any_field_empty([system_message, ner_message, human_message_1, ai_message_1, human_message_2]):
                    st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                    st.stop()

                st.balloons()

                with get_openai_callback() as callbacks:
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_message),
                        HumanMessage(human_message_1),
                        AIMessage(ai_message_1),
                        HumanMessage(human_message_2)
                    ])
                    chain = prompt | llm

                    with st.spinner("Generating response..."):
                        response = chain.invoke(input={"categories": ner_message})

                    st.markdown(constants.TAB_FORM_BOT_RESPONSE)

                    try:
                        st.json(json.loads(response.content))
                    except ValueError:
                        st.text(response.content)

                    st.markdown(constants.TAB_FORM_FULL_PROMPT)
                    st.text(prompt.format(categories=ner_message))
                    st.markdown(constants.TAB_FORM_REQUEST_STATS)
                    st.text(callbacks)
                    st.toast("Done!", icon="😍")


def create_ner_zero_shot_prompting_tab(ner_zero_shot_prompting, llm):
    with ner_zero_shot_prompting:
        st.header(constants.NER_ZERO_SHOT_PROMPTING_TAB_HEADER)
        st.markdown(read_md_file("markdowns/ner-zero-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/ner-zero-shot-prompting-example.md"))

        with st.form("ner_zero_shot_prompting_form"):
            st.header(constants.NER_ZERO_SHOT_PROMPTING_TAB_FORM_HEADER)
            system_message = st.text_area(
                label=constants.TAB_FORM_SYSTEM_MESSAGE,
                placeholder=constants.NER_ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                height=200
            )
            ner_message = st.text_area(
                label="NER (categories definition)",
                placeholder=constants.NER_ZERO_SHOT_PROMPTING_TAB_CATEGORIES,
                height=200
            )
            human_message = st.text_area(
                label=constants.TAB_FORM_HUMAN_MESSAGE,
                placeholder=constants.NER_ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE,
                height=200
            )
            submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

            if submitted:
                if is_any_field_empty([system_message, ner_message, human_message]):
                    st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                    st.stop()

                st.balloons()

                with get_openai_callback() as callbacks:
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_message),
                        HumanMessage(human_message)
                    ])
                    chain = prompt | llm

                    with st.spinner("Generating response..."):
                        response = chain.invoke(input={"categories": ner_message})

                    st.markdown(constants.TAB_FORM_BOT_RESPONSE)

                    try:
                        st.json(json.loads(response.content))
                    except ValueError:
                        st.text(response.content)

                    st.markdown(constants.TAB_FORM_FULL_PROMPT)
                    st.text(prompt.format(categories=ner_message))
                    st.markdown(constants.TAB_FORM_REQUEST_STATS)
                    st.text(callbacks)
                    st.toast("Done!", icon="😍")


def create_few_shot_prompting_tab(few_shot_prompting_tab, llm):
    with few_shot_prompting_tab:
        st.header(constants.FEW_SHOT_PROMPTING_TAB_HEADER)
        st.markdown(read_md_file("markdowns/few-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/few-shot-prompting-example.md"))

        with st.form("few_shot_prompting_form"):
            st.header(constants.FEW_SHOT_PROMPTING_TAB_FORM_HEADER)
            system_message = st.text_area(
                label=constants.TAB_FORM_SYSTEM_MESSAGE,
                placeholder=constants.FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                height=200
            )
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("First example")
                human_message_1 = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1,
                    height=200
                )
                ai_message_1 = st.text_area(
                    label=constants.TAB_FORM_AI_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1,
                    height=25
                )

            with col2:
                st.subheader("Second example")
                human_message_2 = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2,
                    height=200
                )
                ai_message_2 = st.text_area(
                    label=constants.TAB_FORM_AI_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_2,
                    height=25
                )

            with col3:
                st.subheader("Third example")
                human_message_3 = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_3,
                    height=200
                )
                ai_message_3 = st.text_area(
                    label=constants.TAB_FORM_AI_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_3,
                    height=25
                )

            st.subheader("Actual email")
            human_message_4 = st.text_area(
                label=constants.TAB_FORM_HUMAN_MESSAGE,
                placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_4,
                height=200
            )
            submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

            if submitted:
                if is_any_field_empty([system_message, human_message_1, ai_message_1, human_message_2, ai_message_2,
                                       human_message_3, ai_message_3, human_message_4]):
                    st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                    st.stop()

                st.balloons()

                with get_openai_callback() as callbacks:
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(system_message),
                        HumanMessage(human_message_1),
                        AIMessage(ai_message_1),
                        HumanMessage(human_message_2),
                        AIMessage(ai_message_2),
                        HumanMessage(human_message_3),
                        AIMessage(ai_message_3),
                        HumanMessage(human_message_4)
                    ])
                    chain = prompt | llm

                    with st.spinner("Generating response..."):
                        response = chain.invoke({})

                    st.markdown(constants.TAB_FORM_BOT_RESPONSE)

                    try:
                        st.json(json.loads(response.content))
                    except ValueError:
                        st.text(response.content)

                    st.markdown(constants.TAB_FORM_FULL_PROMPT)
                    st.text(prompt.format())
                    st.markdown(constants.TAB_FORM_REQUEST_STATS)
                    st.text(callbacks)
                    st.toast("Done!", icon="😍")


def create_zero_shot_prompting_tab(zero_shot_prompting_tab, llm):
    with zero_shot_prompting_tab:
        st.header(constants.ZERO_SHOT_PROMPTING_TAB_HEADER)
        st.markdown(read_md_file("markdowns/zero-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/zero-shot-prompting-example.md"))

        with st.form("zero_shot_prompting_form"):
            st.header(constants.ZERO_SHOT_PROMPTING_TAB_FORM_HEADER)
            system_message = st.text_area(
                label=constants.TAB_FORM_SYSTEM_MESSAGE,
                placeholder=constants.ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                height=200
            )
            human_message = st.text_area(
                label=constants.TAB_FORM_HUMAN_MESSAGE,
                placeholder=constants.ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE,
                height=200
            )
            submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

            if submitted:
                if is_any_field_empty([system_message, human_message]):
                    st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                    st.stop()

                st.balloons()

                with get_openai_callback() as callbacks:
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(system_message),
                        HumanMessage(human_message)
                    ])
                    chain = prompt | llm

                    with st.spinner("Generating response..."):
                        response = chain.invoke({})

                    st.markdown(constants.TAB_FORM_BOT_RESPONSE)

                    try:
                        st.json(json.loads(response.content))
                    except ValueError:
                        st.text(response.content)

                    st.markdown(constants.TAB_FORM_FULL_PROMPT)
                    st.text(prompt.format())
                    st.markdown(constants.TAB_FORM_REQUEST_STATS)
                    st.text(callbacks)
                    st.toast("Done!", icon="😍")


def is_any_field_empty(list_of_fields: list) -> bool:
    return any([field == "" for field in list_of_fields])


def create_azure_openai_model(openai_api_key: str, temperature: float) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint="https://open-ai-resource-gen-ai.openai.azure.com/",
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai_api_key,
        openai_api_type="azure",
        deployment_name="gpt-35-dev",
        model_name="gpt-35-turbo",
        model_version="0613",
        temperature=temperature
    )


def create_page_config() -> None:
    st.set_page_config(
        page_title="Email to Quote 📧➡️💰",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def remove_unused_html() -> None:
    # Hide Streamlit main menu, header and footer
    st.markdown(constants.HIDE_STREAMLIT_ELEMENTS, unsafe_allow_html=True)
    # Hide expandable menu
    st.markdown(constants.HIDE_SIDEBAR_HTML, unsafe_allow_html=True)


def create_authenticator() -> stauth.Authenticate:
    # Load authentication credentials from .yaml file
    with open("authentication.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    # create an instance of the Authenticate class with the credentials
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"]
    )


def read_md_file(file_path: str) -> str:
    return open(file_path, encoding="utf8").read()


if __name__ == "__main__":
    main()
