# --- HTML ---

HIDE_SIDEBAR_AND_DEPLOY_HTML = """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    footer {visibility: hidden;}
    [data-testid="stDeployButton"] {
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

# --- LOGIN/LOGOUT PAGE ---

LOGIN_PAGE_ERROR_MESSAGE = "Username or/and password is incorrect. Try again."

LOGIN_PAGE_WARNING_MESSAGE = "Please enter your username and password."

LOGIN_PAGE_NAME = "Login :closed_lock_with_key:"

LOGIN_BUTTON_TEXT = "Enter to AI world :mechanical_arm::mechanical_leg:"

LOGOUT_BUTTON_TEXT = "Leave AI world :disappointed_relieved::broken_heart:"

# --- MAIN PAGE ---

MAIN_PAGE_HEADER = "Email to Quote :e-mail::arrow_right::moneybag:"

HOW_TO_START_EXPANDER = "**How to start?** :thinking_face:"

IMPORTANT_FILES_EXPANDER = "**Important files** :open_file_folder:"

DATASETS_EXPANDER = "**Datasets** :bar_chart:"

MAIN_MENU_ABOUT = "This app allows you to test different prompt techniques for the LLM model."

# --- SIDEBAR ---

SIDEBAR_HEADER = "Model parameters:"

SIDEBAR_ERROR_MESSAGE = "For now we have only one LLM. You can't change the parameters! :no_entry:"

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

TAB_NAME_TEST_PROMPT = "Test your prompt :test_tube::dna::pencil2:"

TAB_DESCRIPTION_EXPANDER_TEXT = "**Description** :pencil2:"

TAB_EXAMPLE_EXPANDER_TEXT = "**See example** :eyes:"

TAB_FORM_EXPANDER_TEXT = "**Try it yourself** :rocket:"

TAB_STATS_EXPANDER_TEXT = "**Stats** :bar_chart:"

TAB_FORM_SYSTEM_MESSAGE = "Enter system message:"

TAB_FORM_HUMAN_MESSAGE = "Enter human message:"

TAB_FORM_AI_MESSAGE = "Enter AI message:"

TAB_FORM_PROMPT_MESSAGE = "Enter your prompt:"

TAB_FORM_FILE = "External file from the which the content will be taken:"

TAB_FORM_SUBMIT_BUTTON = "Sent prompt to model :rocket:"

TAB_FORM_EMPTY_FIELD_WARNING = "Please enter system or/and human message. Or copy from the example above."

TAB_FORM_EMPTY_FILE_WARNING = "Please upload a file. Currently only markdown file is only supported!"

TAB_FORM_BOT_RESPONSE = "#### Bot response :speech_balloon:"

TAB_FORM_FULL_PROMPT = "#### Full prompt :capital_abcd:"

TAB_FORM_REQUEST_STATS = "#### Request stats/costs :chart_with_upwards_trend::money_with_wings:"

# --- ZERO-SHOT PROMPTING TAB ---

ZERO_SHOT_PROMPTING_TAB_HEADER = "**Zero-shot Prompting** :zero::gun:"

ZERO_SHOT_PROMPTING_TAB_FORM_HEADER = "Try Zero-shot Prompting"

ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE = """You are a bot that extract data from emails in different languages.
Below are the rules that you have to follow:
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

FEW_SHOT_PROMPTING_TAB_HEADER = "**Few-shot Prompting** :1234::gun:"

FEW_SHOT_PROMPTING_TAB_FORM_HEADER = "Try Few-shot Prompting"

FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE = """You are a bot that extract data from emails in different languages.
Below are the rules that you have to follow:
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

FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1 = """FROM: Juliusz Gorzen
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

I would like to book a FTL transport from Safranberg 123, 12345 Ulm to
Wietrzna 34, Wroclaw 52-023, Poland next Monday.

Thanks.

Sincerely,
Juliusz"""

FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1 = """{"from_address": "Safranberg 123, 12345 Ulm", "to_address": "Wietrzna 34, Wroclaw 52-023, Poland"}"""

FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2 = """FROM: Abc def
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

I would like to book a magic transport from 3486 Tuna Street, 48302, Bloomfield Township to
1011 Franklin Avenue, Daytona Beach 32114, US on 2024-03-03 10 pm.

Thanks you.

BR,
Abc"""

FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_2 = """{"from_address": "3486 Tuna Street, 48302, Bloomfield Township", "to_address": "1011 Franklin Avenue, Daytona Beach 32114, US"}"""

FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_3 = """FROM: John Taylor
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

Pleas book the transport.
From: 2132 Thomas Street, Wheeling, US
To: -
Date: 2024-02-11 09:30"""

FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_3 = """{"from_address": "2132 Thomas Street, Wheeling, US", "to_address": ""}"""

FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_7 = """Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days"""

# --- NER ZERO-SHOT PROMPTING TAB ---

NER_ZERO_SHOT_PROMPTING_TAB_HEADER = "**NER + Zero-shot Prompting** :writing_hand::heavy_plus_sign::zero::gun:"

NER_ZERO_SHOT_PROMPTING_TAB_FORM_HEADER = "Try NER + Zero-shot Prompting"

NER_ZERO_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE = """For each text, mark NER tags. In [] mark the entity and in () mark the tag.
Tag categories:
{categories}
"""

NER_ZERO_SHOT_PROMPTING_TAB_CATEGORIES = """| Tag Name             | Tag Definition                                                                                                                                                                                                                      |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| origin_location      | (Address) First pick-up location of transport. Should contain city and country at a minimum                                                                                                                                         |
| destination_location | (Address) Last delivery location of transport. Should contain city and country at a minimum                                                                                                                                         |
| date_of_mail         | (Date) Date when mail was sent / received                                                                                                                                                                                           |
| weight               | (Number) Load weight, Attribute “unit” contains one of supported weight units: Ton, Kilogram.If not specified assume 22t.                                                                                                           |
| start_of_transport   | (DateTime) Date & time when pick-up is to happen. If not specified, assume in 5 days at 7:00am UTC.                                                                                                                                 |
| num_activities       | (Number) Sum of the total number of pick-up & delivery locations. If not specified assume 2                                                                                                                                         |
| transport_mode       | (String) Required mode of transportation. If not specified, assume “ROAD”. Other options are “AIR”, “RAIL”, “MULTIMODAL”, “RIVER”                                                                                                   |
| load_type            | (String) Type of transport required. If not specified, assume Full Truck Load (FTL). Other options are Less-than Truck Load (LTL), Bulk. The later options usually apply if Note these can usually be inferred from the description |
| vehicle_type         | (String) Type of truck used for the transport. If not specified, assume “standard”. Other options are “Reefer”, “Tanker”, “Flatbed / Open platform trucks”                                                                          |
| hazardous_goods      | (String) Flag in case the transport contains hazardous goods. If not specified assume “No” """

NER_ZERO_SHOT_PROMPTING_TAB_HUMAN_MESSAGE = """Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days"""

# -- NER FEW-SHOT PROMPTING TAB ---

NER_FEW_SHOT_PROMPTING_TAB_HEADER = "**NER + Few-shot Prompting** :writing_hand::heavy_plus_sign::1234::gun:"

NER_FEW_SHOT_PROMPTING_TAB_FORM_HEADER = "Try NER + Few-shot Prompting"

NER_FEW_SHOT_PROMPTING_TAB_SYSTEM_MESSAGE = """For each text, mark NER tags. In [] mark the entity and in () mark the tag.
Tag categories:
{categories}
"""

NER_FEW_SHOT_PROMPTING_TAB_CATEGORIES = """| Tag Name             | Tag Definition                                                                                                                                                                                                                      |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| origin_location      | (Address) First pick-up location of transport. Should contain city and country at a minimum                                                                                                                                         |
| destination_location | (Address) Last delivery location of transport. Should contain city and country at a minimum                                                                                                                                         |
| date_of_mail         | (Date) Date when mail was sent / received                                                                                                                                                                                           |
| weight               | (Number) Load weight, Attribute “unit” contains one of supported weight units: Ton, Kilogram.If not specified assume 22t.                                                                                                           |
| start_of_transport   | (DateTime) Date & time when pick-up is to happen. If not specified, assume in 5 days at 7:00am UTC.                                                                                                                                 |
| num_activities       | (Number) Sum of the total number of pick-up & delivery locations. If not specified assume 2                                                                                                                                         |
| transport_mode       | (String) Required mode of transportation. If not specified, assume “ROAD”. Other options are “AIR”, “RAIL”, “MULTIMODAL”, “RIVER”                                                                                                   |
| load_type            | (String) Type of transport required. If not specified, assume Full Truck Load (FTL). Other options are Less-than Truck Load (LTL), Bulk. The later options usually apply if Note these can usually be inferred from the description |
| vehicle_type         | (String) Type of truck used for the transport. If not specified, assume “standard”. Other options are “Reefer”, “Tanker”, “Flatbed / Open platform trucks”                                                                          |
| hazardous_goods      | (String) Flag in case the transport contains hazardous goods. If not specified assume “No” """

NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_1 = """Hey, 
Hope you're doing well! We need to move some goods from Berlin to Paris. The details are as follows:
- **Origin:** Berlin, Germany
- **Destination:** Paris, France
- **Weight:** 5 tons
- **Start Date & Time:** March 15, 2024, at 09:00 AM
Let me know if you can squeeze this into your schedule. 
Cheers"""

NER_FEW_SHOT_PROMPTING_TAB_AI_MESSAGE_1 = """Hey, 
Hope you're doing well! We need to move some goods from Berlin to Paris. The details are as follows:
- **Origin:** [Berlin, Germany](origin_location)
- **Destination:** [Paris, France](destination_location)
- **Weight:** [5 tons](weight)
- **Start Date & Time:** [March 15, 2024, at 09:00 AM](start_of_transport)
Let me know if you can squeeze this into your schedule. 
Cheers"""

NER_FEW_SHOT_PROMPTING_TAB_HUMAN_MESSAGE_2 = """Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days"""

# --- RAG TAB ---

RAG_TAB_HEADER = "**RAG** :bookmark_tabs:"

RAG_TAB_FORM_HEADER = "Try RAG"

RAG_TAB_SYSTEM_MESSAGE = """You are a bot that answers the following question based only on the provided context:

<context>
{context}
</context>
"""

RAG_TAB_HUMAN_MESSAGE = "Which entities are related to the location. List all with the descriptions."

# --- TEST PROMPT TAB ---
TEST_PROMPT_TAB_HEADER = "**Test your prompt** :test_tube::dna::pencil2:"

TEST_PROMPT_TAB_FORM_HEADER = "Try your prompt"

TEST_PROMPT_TAB_PROMPT = "System: You are a bot that..."

CONFUSION_MATRIX_FORM_HEADER = "Compare results (Confusion Matrix)"

TAB_FORM_ACTUAL_SHEET_NAME = "Sheet name with actual values"

TAB_FORM_PREDICTED_SHEET_NAME = "Sheet name with predicted values"

TAB_FORM_EXCEL_FILE = "Excel file with actual and predicted values"

TAB_FORM_SUBMIT_BUTTON_CONFUSION_MATRIX = "Compare results :bar_chart:"