HIDE_SIDEBAR_HTML = """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
"""

DISPLAY_SIDEBAR_HTML = """
<style>
    [data-testid="collapsedControl"] {
        display: grid
    }
</style>
"""

HIDE_STREAMLIT_ELEMENTS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>            
"""
