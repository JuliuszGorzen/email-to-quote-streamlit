import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from annotated_text import annotated_text
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from yaml.loader import SafeLoader

import constants


def main() -> None:
    create_page_config()

    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url="https://genai-dev-keyvault.vault.azure.net/", credential=credential)
    openai_api_key_secret = secret_client.get_secret("aikey").value
    llm = create_azure_openai_model(openai_api_key_secret, 0.2)
    embedding_llm = create_azure_openai_embedding_model(openai_api_key_secret)

    hide_sidebar()

    authenticator = create_authenticator()
    create_login_page(authenticator)

    if st.session_state["authentication_status"]:
        display_sidebar()
        create_logout_button(authenticator)
        create_main_page()
        create_sidebar()
        create_tabs(llm, embedding_llm)
    elif st.session_state["authentication_status"] is False:
        st.error(constants.LOGIN_PAGE_ERROR_MESSAGE)
    elif st.session_state["authentication_status"] is None:
        st.warning(constants.LOGIN_PAGE_WARNING_MESSAGE)


def create_logout_button(authenticator: stauth.Authenticate) -> None:
    authenticator.logout(
        button_name=constants.LOGOUT_BUTTON_TEXT,
        location="sidebar"
    )


def create_login_page(authenticator: stauth.Authenticate) -> None:
    authenticator.login(fields={
        "Form name": constants.LOGIN_PAGE_NAME,
        "Username": "username",
        "Password": "password",
        "Login": constants.LOGIN_BUTTON_TEXT
    })


def create_main_page() -> None:
    st.title(constants.MAIN_PAGE_HEADER)

    with st.expander(constants.HOW_TO_START_EXPANDER):
        st.markdown(read_md_file("markdowns/how-to-start-description.md"))

    with st.expander(constants.IMPORTANT_FILES_EXPANDER):
        st.markdown(read_md_file("markdowns/important-files-description.md"))

        col1, col2, col3 = st.columns(3)

        with col1:
            with open("important-files/example_markdown_rag.md", "r", encoding="utf-8") as file:
                st.download_button(
                    label="Example .md file (RAG) :bookmark_tabs:",
                    data=file,
                    file_name="example_markdown_rag.md",
                    mime="text/markdown"
                )
            st.caption("Download the example .md file to use it in the RAG tab")

        with col2:
            with open("important-files/email-2-quote-dataset.xlsx", "rb") as file:
                st.download_button(
                    label="Dataset .xlsx file :bar_chart:",
                    data=file,
                    file_name="email-2-quote-dataset.xlsx",
                    mime="application/vnd.ms-excel"
                )
            st.caption("Download dataset file with sample emails")

        with col3:
            with open("important-files/email-2-quote-openapi.json", "r", encoding="utf-8") as file:
                st.download_button(
                    label="OpenAPI schema :book:",
                    data=file,
                    file_name="email-2-quote-openapi.json",
                    mime="application/json"
                )
            st.caption("Download OpenAPI schema")

    with st.expander(constants.DATASET_EXPANDER):
        df_emails = pd.read_excel("important-files/email-2-quote-dataset.xlsx", "mails")
        st.dataframe(df_emails, use_container_width=True)

        df_objects = pd.read_excel("important-files/email-2-quote-dataset.xlsx", "objects").astype(str)
        st.dataframe(df_objects, use_container_width=True)


def create_sidebar() -> None:
    st.sidebar.divider()
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
            value="<api-key>",
            type="password",
            help=constants.SIDEBAR_FORM_OPENAI_API_KEY_HELP,
            disabled=True
        )
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
            value=0.2,
            step=0.01,
            help=constants.SIDEBAR_FORM_TEMPERATURE_HELP,
            disabled=True
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


def create_tabs(llm: AzureChatOpenAI, embedding_llm: AzureOpenAIEmbeddings) -> None:
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
    create_rag_tab(rag, llm, embedding_llm)


def create_rag_tab(rag: str, llm: AzureChatOpenAI, embedding_llm: AzureOpenAIEmbeddings) -> None:
    with rag:
        st.header(constants.RAG_TAB_HEADER)

        with st.expander(constants.TAB_DESCRIPTION_EXPANDER_TEXT, expanded=True):
            st.markdown(read_md_file("markdowns/rag-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/rag-example.md"))

        with st.expander(constants.TAB_FORM_EXPANDER_TEXT):
            with st.form("rag_form", border=False):
                st.header(constants.RAG_TAB_FORM_HEADER)
                system_message = st.text_area(
                    label=constants.TAB_FORM_SYSTEM_MESSAGE,
                    value=constants.RAG_TAB_SYSTEM_MESSAGE,
                    placeholder=constants.RAG_TAB_SYSTEM_MESSAGE,
                    height=200
                )

                col1, col2 = st.columns(2)

                with col1:
                    human_message = st.text_area(
                        label=constants.TAB_FORM_HUMAN_MESSAGE,
                        value=constants.RAG_TAB_HUMAN_MESSAGE,
                        placeholder=constants.RAG_TAB_HUMAN_MESSAGE,
                        height=200
                    )

                with col2:
                    external_file = st.file_uploader(
                        label=constants.TAB_FORM_FILE,
                        type=["md"]
                    )

                submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

                if submitted:
                    if is_any_field_empty([system_message, human_message]):
                        st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                        st.stop()

                    if external_file is None:
                        st.warning(constants.TAB_FORM_EMPTY_FILE_WARNING)
                        st.stop()

                    with get_openai_callback() as callbacks:
                        headers_to_split_on = [
                            ("#", "Header 1"),
                            ("##", "Header 2"),
                            ("###", "Header 3"),
                            ("####", "Header 4")
                        ]
                        document = external_file.read().decode("utf-8")

                        with st.spinner("Generating response..."):
                            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                            docs = markdown_splitter.split_text(document)
                            vectorstore = FAISS.from_documents(docs, embedding_llm)
                            prompt = ChatPromptTemplate.from_messages([
                                SystemMessagePromptTemplate.from_template(system_message),
                                HumanMessagePromptTemplate.from_template("{input}")
                            ])
                            document_chain = create_stuff_documents_chain(llm, prompt)
                            retriever = vectorstore.as_retriever()
                            retrieval_chain = create_retrieval_chain(retriever, document_chain)
                            response = retrieval_chain.invoke({"context": docs, "input": human_message})

                        st.markdown(constants.TAB_FORM_BOT_RESPONSE)
                        st.text(response["answer"])
                        st.markdown(constants.TAB_FORM_FULL_PROMPT)
                        st.code(
                            prompt.format(context=[d.page_content for d in docs][0], input=human_message),
                            language="text"
                        )
                        st.markdown(constants.TAB_FORM_REQUEST_STATS)
                        st.text(callbacks)
                        create_success_toast()

        with st.expander(constants.TAB_STATS_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/stats-description.md"))


def create_ner_few_shot_prompting_tab(ner_few_shot_prompting: str, llm: AzureChatOpenAI) -> None:
    with ner_few_shot_prompting:
        st.header(constants.NER_FEW_SHOT_PROMPTING_TAB_HEADER)

        with st.expander(constants.TAB_DESCRIPTION_EXPANDER_TEXT, expanded=True):
            st.markdown(read_md_file("markdowns/ner-few-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/ner-few-shot-prompting-example.md"))
            annotated_text(
                "Hello, Please send me your offer for groupage transport for: 1 pallet: ",
                ("120cm x 80cm x 120cm", "load_type"),
                ("155 Kg", "weight"),
                "Loading: ",
                ("300283 Timisoara, Romania", "origin_location"),
                "Unloading: ",
                ("4715-405 Braga, Portugal", "destination_location"),
                "Can be picked up. Payment after 7 days"
            )

        with st.expander(constants.TAB_FORM_EXPANDER_TEXT):
            with st.form("ner_few_shot_prompting_form", border=False):
                st.header(constants.NER_FEW_SHOT_PROMPTING_TAB_FORM_HEADER)
                system_message = st.text_area(
                    label=constants.TAB_FORM_SYSTEM_MESSAGE,
                    value=constants.NER_FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    height=200
                )
                ner_message = st.text_area(
                    label="NER (categories definition)",
                    value=constants.NER_FEW_SHOT_PROMPTING_TAB_CATEGORIES,
                    placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_CATEGORIES,
                    height=200
                )

                st.subheader("First example")
                col1, col2 = st.columns(2)

                with col1:
                    human_message_1 = st.text_area(
                        label=constants.TAB_FORM_HUMAN_MESSAGE,
                        value=constants.NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1,
                        placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1,
                        height=200
                    )

                with col2:
                    ai_message_1 = st.text_area(
                        label=constants.TAB_FORM_AI_MESSAGE,
                        value=constants.NER_FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1,
                        placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1,
                        height=200
                    )

                st.subheader("Actual email")
                human_message_2 = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    value=constants.NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2,
                    placeholder=constants.NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2,
                    height=200
                )
                submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

                if submitted:
                    if is_any_field_empty(
                            [system_message, ner_message, human_message_1, ai_message_1, human_message_2]):
                        st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                        st.stop()

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
                        create_annotated_text(response.content)
                        st.markdown(constants.TAB_FORM_FULL_PROMPT)
                        st.code(prompt.format(categories=ner_message), language="text")
                        st.markdown(constants.TAB_FORM_REQUEST_STATS)
                        st.text(callbacks)
                        create_success_toast()

        with st.expander(constants.TAB_STATS_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/stats-description.md"))


def create_ner_zero_shot_prompting_tab(ner_zero_shot_prompting: str, llm: AzureChatOpenAI) -> None:
    with ner_zero_shot_prompting:
        st.header(constants.NER_ZERO_SHOT_PROMPTING_TAB_HEADER)

        with st.expander(constants.TAB_DESCRIPTION_EXPANDER_TEXT, expanded=True):
            st.markdown(read_md_file("markdowns/ner-zero-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/ner-zero-shot-prompting-example.md"))

        with st.expander(constants.TAB_FORM_EXPANDER_TEXT):
            with st.form("ner_zero_shot_prompting_form", border=False):
                st.header(constants.NER_ZERO_SHOT_PROMPTING_TAB_FORM_HEADER)
                system_message = st.text_area(
                    label=constants.TAB_FORM_SYSTEM_MESSAGE,
                    value=constants.NER_ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    placeholder=constants.NER_ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    height=200
                )
                ner_message = st.text_area(
                    label="NER (categories definition)",
                    value=constants.NER_ZERO_SHOT_PROMPTING_TAB_CATEGORIES,
                    placeholder=constants.NER_ZERO_SHOT_PROMPTING_TAB_CATEGORIES,
                    height=200
                )
                human_message = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    value=constants.NER_ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE,
                    placeholder=constants.NER_ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE,
                    height=200
                )
                submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

                if submitted:
                    if is_any_field_empty([system_message, ner_message, human_message]):
                        st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                        st.stop()

                    with get_openai_callback() as callbacks:
                        prompt = ChatPromptTemplate.from_messages([
                            SystemMessagePromptTemplate.from_template(system_message),
                            HumanMessage(human_message)
                        ])
                        chain = prompt | llm

                        with st.spinner("Generating response..."):
                            response = chain.invoke(input={"categories": ner_message})

                        st.markdown(constants.TAB_FORM_BOT_RESPONSE)
                        st.text(response.content)
                        st.markdown(constants.TAB_FORM_FULL_PROMPT)
                        st.code(prompt.format(categories=ner_message), language="text")
                        st.markdown(constants.TAB_FORM_REQUEST_STATS)
                        st.text(callbacks)
                        create_success_toast()

        with st.expander(constants.TAB_STATS_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/stats-description.md"))


def create_few_shot_prompting_tab(few_shot_prompting_tab: str, llm: AzureChatOpenAI) -> None:
    with few_shot_prompting_tab:
        st.header(constants.FEW_SHOT_PROMPTING_TAB_HEADER)

        with st.expander(constants.TAB_DESCRIPTION_EXPANDER_TEXT, expanded=True):
            st.markdown(read_md_file("markdowns/few-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/few-shot-prompting-example.md"))

        with st.expander(constants.TAB_FORM_EXPANDER_TEXT):
            with st.form("few_shot_prompting_form", border=False):
                st.header(constants.FEW_SHOT_PROMPTING_TAB_FORM_HEADER)
                system_message = st.text_area(
                    label=constants.TAB_FORM_SYSTEM_MESSAGE,
                    value=constants.FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    height=200
                )
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("First example")
                    human_message_1 = st.text_area(
                        label=constants.TAB_FORM_HUMAN_MESSAGE,
                        value=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1,
                        placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1,
                        height=200
                    )
                    ai_message_1 = st.text_area(
                        label=constants.TAB_FORM_AI_MESSAGE,
                        value=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1,
                        placeholder=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1,
                        height=25
                    )

                with col2:
                    st.subheader("Second example")
                    human_message_2 = st.text_area(
                        label=constants.TAB_FORM_HUMAN_MESSAGE,
                        value=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2,
                        placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2,
                        height=200
                    )
                    ai_message_2 = st.text_area(
                        label=constants.TAB_FORM_AI_MESSAGE,
                        value=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_2,
                        placeholder=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_2,
                        height=25
                    )

                with col3:
                    st.subheader("Third example")
                    human_message_3 = st.text_area(
                        label=constants.TAB_FORM_HUMAN_MESSAGE,
                        value=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_3,
                        placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_3,
                        height=200
                    )
                    ai_message_3 = st.text_area(
                        label=constants.TAB_FORM_AI_MESSAGE,
                        value=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_3,
                        placeholder=constants.FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_3,
                        height=25
                    )

                st.subheader("Actual email")
                human_message_4 = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    value=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_4,
                    placeholder=constants.FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_4,
                    height=200
                )
                submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

                if submitted:
                    if is_any_field_empty([system_message, human_message_1, ai_message_1, human_message_2, ai_message_2,
                                           human_message_3, ai_message_3, human_message_4]):
                        st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                        st.stop()

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
                        st.code(prompt.format(), language="text")
                        st.markdown(constants.TAB_FORM_REQUEST_STATS)
                        st.text(callbacks)
                        create_success_toast()

        with st.expander(constants.TAB_STATS_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/stats-description.md"))


def create_zero_shot_prompting_tab(zero_shot_prompting_tab: str, llm: AzureChatOpenAI) -> None:
    with zero_shot_prompting_tab:
        st.header(constants.ZERO_SHOT_PROMPTING_TAB_HEADER)

        with st.expander(constants.TAB_DESCRIPTION_EXPANDER_TEXT, expanded=True):
            st.markdown(read_md_file("markdowns/zero-shot-prompting-description.md"))

        with st.expander(constants.TAB_EXAMPLE_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/zero-shot-prompting-example.md"))

        with st.expander(constants.TAB_FORM_EXPANDER_TEXT):
            with st.form("zero_shot_prompting_form", border=False):
                st.header(constants.ZERO_SHOT_PROMPTING_TAB_FORM_HEADER)
                system_message = st.text_area(
                    label=constants.TAB_FORM_SYSTEM_MESSAGE,
                    value=constants.ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    placeholder=constants.ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE,
                    height=200
                )
                human_message = st.text_area(
                    label=constants.TAB_FORM_HUMAN_MESSAGE,
                    value=constants.ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE,
                    placeholder=constants.ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE,
                    height=200
                )
                submitted = st.form_submit_button(label=constants.TAB_FORM_SUBMIT_BUTTON)

                if submitted:
                    if is_any_field_empty([system_message, human_message]):
                        st.warning(constants.TAB_FORM_EMPTY_FIELD_WARNING)
                        st.stop()

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
                        st.code(prompt.format(), language="text")
                        st.markdown(constants.TAB_FORM_REQUEST_STATS)
                        st.text(callbacks)
                        create_success_toast()

        with st.expander(constants.TAB_STATS_EXPANDER_TEXT):
            st.markdown(read_md_file("markdowns/stats-description.md"))

            # import some data to play with
            iris = datasets.load_iris()
            X = iris.data
            y = iris.target
            class_names = iris.target_names

            # Split the data into a training set and a test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            # Run classifier, using a model that is too regularized (C too low) to see
            # the impact on the results
            classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            titles_options = [
                ("Confusion matrix, without normalization", None),
                ("Normalized confusion matrix", "true"),
            ]
            for title, normalize in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    classifier,
                    X_test,
                    y_test,
                    display_labels=class_names,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                )
                disp.ax_.set_title(title)

                plt.rcParams["figure.figsize"] = (6, 3)

            st.pyplot(plt, use_container_width=False)


def create_success_toast() -> None:
    st.toast("Success!", icon="🎉")


def create_annotated_text(text: str) -> None:
    entities = re.findall(r"\[(.*?)\]\((.*?)\)", text)
    words = re.split("\[(.*?)\)", text)

    for word in words:
        for entity in entities:
            if entity[0] in word and entity[1] in word:
                words[words.index(word)] = entity

    annotated_text(words)


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


def create_azure_openai_embedding_model(openai_api_key: str) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint="https://open-ai-resource-gen-ai.openai.azure.com/",
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai_api_key,
        openai_api_type="azure"
    )


def create_page_config() -> None:
    st.set_page_config(
        page_title="Email to Quote 📧➡️💰",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": "https://www.transporeon.com/en",
            "Report a bug": "https://www.transporeon.com/en",
            "About": constants.MAIN_MENU_ABOUT
        }
    )


def hide_sidebar() -> None:
    st.markdown(constants.HIDE_SIDEBAR_AND_DEPLOY_HTML, unsafe_allow_html=True)


def display_sidebar() -> None:
    st.markdown(constants.DISPLAY_SIDEBAR_HTML, unsafe_allow_html=True)


def create_authenticator() -> stauth.Authenticate:
    with open("authentication.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

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
