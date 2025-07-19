from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# llm_openai = ChatOpenAI(model="gpt-4o-mini")
# llm_anthropic = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", timeout=10, stop=None)
llm_groq = ChatGroq(model="llama3-8b-8192", temperature=0.5)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a reasearch report.
            Answer the user query and neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        (
            "placeholder",
            "{chat_history}"
        ),
        (
            "human",
            "{query}"
        ),
        (
            "placeholder",
            "{agent_scratchpad}"
        )
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)

tools = [search_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm_groq,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("How can I help you today? : ")

result = agent_executor.invoke({"query": query})

print(result["output"])











