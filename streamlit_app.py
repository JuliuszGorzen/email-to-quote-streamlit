from st_pages import Page, show_pages, add_page_title

add_page_title()

show_pages(
    [
        Page("home.py", "🏠 Start here"),
        Page("one_shot_prompting.py", "1️⃣🔫 One-shot Prompting"),
        Page("few_shot_prompting.py", "🎲🔫 Few-shot Prompting"),
        Page("ner_one_shot_prompting.py", "📑➕1️🔫 NER + One-shot Prompting"),
        Page("ner_few_shot_prompting.py", "📑➕🎲🔫 NER + Few-shot Prompting"),
        Page("rag.py", "️⚒️ RAG")
    ]
)
