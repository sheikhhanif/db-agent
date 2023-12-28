# Core imports
import pandas as pd
import ast
import re

# Langchain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.agents import (
    AgentType, create_sql_agent, AgentExecutor, OpenAIFunctionsAgent,
    Tool, initialize_agent
)
from langchain.agents.agent_toolkits import create_retriever_tool, SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain



def initialize_components():
    """Initialize essential components for the agent."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    embeddings = HuggingFaceEmbeddings()
    db = SQLDatabase.from_uri("sqlite:////Users/hanif/personal-finance-agent/data/personal_finance.db")
    return llm, embeddings, db



def load_and_process_data(csv_path):
    """Load and process data from CSV file."""
    sql_qna = pd.read_csv(csv_path)
    query_docs = [
        Document(page_content=sql_qna.user_query[i], metadata={"sql_query": sql_qna.SQL_query[i]})
        for i in range(len(sql_qna))
    ]
    return query_docs


def run_query_save_results(db, query):
    """Run a SQL query and process the results."""
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return res


def create_vector_db_from_documents(documents, embeddings):
    """Create a FAISS vector database from documents."""
    return FAISS.from_documents(documents, embeddings)

def create_vector_db_from_texts(texts, embeddings):
    """Create a FAISS vector database from texts."""
    return FAISS.from_texts(texts, embeddings)


def create_sql_retriever_tool(retriever):
    """Create a retriever tool for SQL queries."""
    tool_description = """
    This tool will help you understand similar examples to adapt them to the user question.
    Input to this tool should be the user question.
    """
    return create_retriever_tool(
        retriever, name="sql_get_similar_examples", description=tool_description
    )

def create_name_retriever_tool(retriever):
    """Create a retriever tool for name searches."""
    description = "use to learn how a piece of data is actually written, can be from transaction category, merchant name etc"
    return create_retriever_tool(retriever, name="name_search", description=description)



def create_agent(llm, db, custom_tool_list):
    """Create and configure the SQL agent."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    custom_suffix = """
    If a user asks for me to filter based on proper nouns, I should first check the spelling using the name_search tool.
    I should first get the similar examples I know.
    If the examples are enough to construct the query, I can build it.
    Otherwise, I can then look at the tables in the database to see what I can query.
    Then I should query the schema of the most relevant tables
    """
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=custom_tool_list,
        suffix=custom_suffix,
    )
