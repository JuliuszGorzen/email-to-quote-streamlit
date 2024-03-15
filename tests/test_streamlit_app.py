from streamlit.testing.v1 import AppTest


def test():
    at = AppTest.from_file("streamlit_app.py")
    at.run()
    assert at.text_input[0].value == ""
