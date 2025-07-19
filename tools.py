from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_text(data: str, filename: str = "research_report.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Text saved to {filename}_{timestamp}.txt"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_text,
    description="Save text to a file",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000, wiki_client=None)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)