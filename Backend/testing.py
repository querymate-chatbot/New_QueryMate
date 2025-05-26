# import os
# import datetime
# import base64
# import requests
# import socket
# import openai
# import urllib.parse
# from sqlalchemy import create_engine
# from dotenv import load_dotenv, find_dotenv
# from langchain_community.callbacks import get_openai_callback
# from langchain_openai import OpenAI
# from langchain.schema import Document 
# from langchain.chains import ConversationChain

# from langchain_utils import get_user_examples
# _ = load_dotenv(find_dotenv())
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_openai import ChatOpenAI
# from connection import get_db_connection
# import re
# import plotly.express as px
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
# from prompts import plotly_error_fix_prompt, greet_prompt, tables_involved, explore_summary_prompt, identify_columns, dataframe_summary
# from langchain.chains import LLMChain
# import warnings
# import pandas as pd
# import matplotlib
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import base64
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_loaders import UnstructuredExcelLoader,PyPDFLoader,UnstructuredPowerPointLoader,TextLoader
# import tempfile
# from connection import get_db_connection
# from langchain_community.vectorstores import Chroma
# from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# from langchain_openai import OpenAIEmbeddings
# from openai import OpenAI
# from flask import Blueprint, session

# warnings.filterwarnings("ignore")
# matplotlib.use('Agg')

# lang = Blueprint('lang', __name__)
# username = os.getenv("DB_USERNAME")
# password = os.getenv("DB_PASSWORD")
# host = os.getenv("DB_HOST")
# port = os.getenv("DB_PORT")
# database_name = os.getenv("DB_NAME")

# api_key = os.getenv("openai_api_key")
# client = OpenAI(api_key=api_key)

# encoded_password = urllib.parse.quote(password)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY

# cached_example_selector = None
# cached_examples = None

# all_tables = ['DimProduct', 'DimCustomer', 'DimDate', 'DimBranch', 'FactTransaction', 'FactAccountBalance']




# def get_example_selector():
#     global cached_example_selector
#     try:
#         if cached_example_selector is not None:
#             return cached_example_selector
        
#         examples = get_user_examples()
#         if not examples:
#             return None
        
#         cached_example_selector = SemanticSimilarityExampleSelector.from_examples(
#             examples,
#             OpenAIEmbeddings(),
#             Chroma,
#             k=3,
#             input_keys=["input"]
#         )
#         return cached_example_selector
#     except Exception as db_error:
#         print(f"Database connection error: {db_error}")
#         return None
#     except Exception as e:
#         print(f"General error in get_example_selector: {e}")
#         return None        

# example_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("human", "{input}\nSQLQuery:"),
#                 ("ai", "{query}"),
#             ]
#         )

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#             example_prompt=example_prompt,
#             example_selector=get_example_selector(),
#             input_variables=["input"]
#         )

# selected_examples = few_shot_prompt.format(input="greenland")