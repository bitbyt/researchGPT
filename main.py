import os
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

import json

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage

from pydantic import BaseModel, Field
from typing import Type
from fastapi import FastAPI

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Search Function
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    print('Searching for... ', query)

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    return response.text

# Website Scraper
def scraper(goal: str, url: str):
    print('Scraping website... ', url)

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Parse the data
    data = json.dumps({"url": url})

    # Send the POST request
    browserless_url = "https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(browserless_url, headers=headers, data=data)

    if response.status_code == 200:
         # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        print("Scraped content:", text)

        # Content might be really long and hit the token limit, we should summarize the text
        if len(text) > 10000:
            output = summary(goal, text)
            return output
        else:
            return text
    else:
        print("HTTP request failed with status code {response.status_code}")


# Summarise Function
def summary(goal, content):
    # invoke chatGPT
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # Use LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    
    documents = text_splitter.create_documents([content])

    # Reusable prompt for each content on the split chain
    map_prompt = """
    Summarize of the following text for {goal}:
    "{text}"
    SUMMARY:
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "goal"])
    
    summary_chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )

    # Run the summary chain
    output = summary_chain.run(input_documents=documents, goal=goal)

    return output

# Agent

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scraper"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scraper"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scraper(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

tools = [
    Tool(
        name="Search",
        func=search,
        description="When you need to answer questions about current events, data. Ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content