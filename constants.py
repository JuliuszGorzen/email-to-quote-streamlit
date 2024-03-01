# --- HTML ---

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

# --- LOGIN/LOGOUT PAGE ---

LOGIN_PAGE_ERROR_MESSAGE = "Username or/and password is incorrect. Try again."

LOGIN_PAGE_WARNING_MESSAGE = "Please enter your username and password."

LOGIN_PAGE_NAME = "Login :closed_lock_with_key:"

LOGIN_BUTTON_TEXT = "Enter to AI world :mechanical_arm::mechanical_leg:"

LOGOUT_BUTTON_TEXT = "Leave AI world :disappointed_relieved::broken_heart:"

# --- MAIN PAGE ---

MAIN_PAGE_HEADER = "Email to Quote :e-mail::arrow_right::moneybag:"

MAIN_PAGE_EXPANDER = "**:rainbow[How to start?]** :thinking_face:"

# --- SIDEBAR ---

SIDEBAR_HEADER = "Model parameters:"

SIDEBAR_SUBHEADER = "Here you can set the parameters for the model."

SIDEBAR_FORM_DESCRIPTION = "Some parameters are disabled as for now we have only one LLM."

SIDEBAR_FORM_AZURE_ENDPOINT = "Azure endpoint"

SIDEBAR_FORM_OPENAI_API_VERSION = "OpenAI API version"

SIDEBAR_FORM_OPENAI_API_KEY = "OpenAI API key"

SIDEBAR_FORM_OPENAI_API_KEY_HELP = "You can find your API key in the Azure or contact the Generative AI team."

SIDEBAR_FORM_OPENAI_API_KEY_WARNING = "Please enter your OpenAI API key."

SIDEBAR_FORM_OPENAI_API_TYPE = "OpenAI API type"

SIDEBAR_FORM_DEPLOYMENT_NAME = "Deployment name"

SIDEBAR_FORM_MODEL_NAME = "Model name"

SIDEBAR_FORM_MODEL_VERSION = "Model version"

SIDEBAR_FORM_TEMPERATURE = "Temperature"

SIDEBAR_FORM_TEMPERATURE_HELP = """A higher temperature value typically makes the output more diverse and creative but 
might also increase its likelihood of straying from the context. Conversely, a lower temperature value 
makes the AI's responses more focused and deterministic, sticking closely to the most likely prediction."""

SIDEBAR_FORM_SUBMIT_BUTTON = "Save :white_check_mark:"

SIDEBAR_FORM_SUMMARY = "Model parameters saved with the following values:"

SIDEBAR_FORM_MODEL_DESCRIPTION = "Model description :pencil2:"

# --- TABS ---

TAB_NAME_ZERO_SHOT_PROMPTING = "Zero-shot Prompting :zero::gun:"

TAB_NAME_FEW_SHOT_PROMPTING = "Few-shot Prompting :1234::gun:"

TAB_NAME_NER_ZERO_SHOT_PROMPTING = "NER + Zero-shot Prompting :writing_hand::heavy_plus_sign::zero::gun:"

TAB_NAME_NER_FEW_SHOT_PROMPTING = "NER + Few-shot Prompting :writing_hand::heavy_plus_sign::1234::gun:"

TAB_NAME_RAG = "RAG :bookmark_tabs:"

TAB_EXAMPLE_EXPANDER_TEXT = "**See example** :eyes:"

TAB_FORM_SYSTEM_MESSAGE = "Enter system message:"

TAB_FORM_HUMAN_MESSAGE = "Enter human message:"

TAB_FORM_SUBMIT_BUTTON = "Sent prompt to model :rocket:"

TAB_FORM_EMPTY_FIELD_WARNING = "Please enter system or/and human message. Or copy from the example above."

TAB_FORM_BOT_RESPONSE = "#### Bot response :speech_balloon:"

TAB_FORM_FULL_PROMPT = "#### Full prompt :capital_abcd:"

TAB_FORM_REQUEST_STATS = "#### Request stats :chart_with_upwards_trend::money_with_wings:"

# --- ZERO-SHOT PROMPTING TAB ---

ZERO_SHOT_PROMPTING_TAB_HEADER = ":orange[Zero-shot Prompting] :zero::gun:"

ZERO_SHOT_PROMPTING_TAB_FORM_HEADER = "Try Zero-shot Prompting"

ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE = """You are a bot that extract data from emails in different languages. Below are the rules that you have to follow:
- You can only write valid JSONs based on the documentation below:
```
{"from_address": "string", "to_address": "string"}
```
- Your goal is to transform email input into structured JSON. Comments are not allowed in the JSON. Text outside the 
JSON is strictly forbidden.
- Users provide the email freight orders as input, which you will transform into JSON format given the JSON 
documentation.
- You can enhance the output with general common-knowledge facts about the world relevant to the procurement event.
- If you cannot find a piece of information, you can leave the corresponding attribute as ""."""

ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE = """Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days"""

# --- FEW-SHOT PROMPTING TAB ---

# --- NER ZERO-SHOT PROMPTING TAB ---

# -- NER FEW-SHOT PROMPTING TAB ---

# --- RAG TAB ---
