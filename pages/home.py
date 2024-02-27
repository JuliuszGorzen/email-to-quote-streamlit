import streamlit as st
from st_pages import add_page_title

add_page_title()


def session_page_views():
    if 'page_views' not in st.session_state:
        st.session_state.page_views = 0
    if st.session_state.page_views == 0:
        st.balloons()
    st.session_state.page_views += 1


session_page_views()
