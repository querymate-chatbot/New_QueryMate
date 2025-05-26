import json
import os
import datetime
import base64
import requests
import socket
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAI
from langchain.schema import Document 
from langchain.chains import ConversationChain
_ = load_dotenv(find_dotenv())
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from prompts import greet_prompt, tables_involved, explore_summary_prompt, identify_columns, dataframe_summary
from langchain.chains import LLMChain
import warnings
import pandas as pd
import matplotlib
import base64
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import UnstructuredExcelLoader,PyPDFLoader,UnstructuredPowerPointLoader,TextLoader
import tempfile
from openai import OpenAI
from flask import Blueprint
from db_config import get_session, get_engine, all_tables
from sqlalchemy import inspect, text
import concurrent.futures
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
from services import get_example_selector_config

lang = Blueprint('lang', __name__)

api_key = os.getenv("openai_api_key")
client = OpenAI(api_key=api_key)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

FRONTEND_APP_URL = "https://127.0.0.1:443/update_status"
def notify_flag(session_id, flag, content, type):
    try:
        requests.post(FRONTEND_APP_URL, json={
            "session_id_stream": session_id,
            "status": flag,
            "content": {
                "content": content,
                "type": type
            }
        }, verify=False)
    except Exception as e:
        print("Failed to notify flag", e)

def get_example_selector(username):
    return get_example_selector_config(username)

def getTablenames(username):
    session = get_session()
    try:
        # Normalize username
        if isinstance(username, list):
            if not username:
                print("Error: Empty username list provided")
                return []
            username = username[0]
        
        if not isinstance(username, str):
            print(f"Error: Invalid username type: {type(username)}")
            return []
        
        # Prepare query
        query = """
        SELECT access
        FROM roles
        JOIN users ON roles.role = users.role
        WHERE username = :username
        """
        params = {"username": username}

        result = session.connection().execute(text(query), params).fetchone()

        if not result or not result[0]:
            print("No access tables found for user.")
            return []

        table_names = [table.strip() for table in result[0].split(',') if table.strip()]

        print(f"User {username} has access to tables: {table_names}")
        return table_names

    except Exception as e:
        print(f"Query execution failed: {e}")
        return []
    
    finally:
        session.close()

def get_chain():
    try:
        print("Creating chain")
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                 
                        You are an advanced SQL assistant specialized in Microsoft SQL Server. Your primary task is to generate syntactically correct and logically sound SQL queries based on the given input, database schema, and applied filters.
                        Key Rules:
                            1. Use Only the Provided Tables and schema name is "bank":  
                                - You can only generate queries using the following tables:  
                                - {table_info}  
                                - Do not reference any table that is not explicitly mentioned in the given schema.

                            2. Ensure SQL Server Compatibility:
                                - Write queries that follow Microsoft SQL Server syntax.  
                                - Use square brackets `[ ]` around reserved keywords if they are used as column or table names.  
                                - Use appropriate joins (`INNER JOIN`, `LEFT JOIN`, etc.) based on logical relationships.

                            3. Apply Given Filters:
                                - Ensure that all conditions and filters provided in the input are correctly applied in the `WHERE` clause.
                                - Convert values to appropriate data types if necessary to prevent errors.

                            4. Column Selection:
                                - Do not use SELECT *.  
                                - Always Strictly add Top 50 to the final SELECT which is responsible for showing content
                                - Always explicitly specify required column names.  
                                - Ensure consistency in column selection to prevent ambiguious column name error.

                            5. Validate Query for Syntax & Logical Accuracy:
                                - Check for missing joins, incorrect column references, or ambiguous fields.  
                                - Ensure that aggregations, `GROUP BY`, and `HAVING` clauses are applied correctly.
                                - Avoid redundant clauses or inefficient query patterns.  

                            6. Focus on SQL-Relevant Information  
                                - Extract only the part of the user's question that is necessary for SQL query generation.  
                                - Ignore any additional context that pertains to NLP-based interpretation or explanation.

                                Example Case:
                                - User's Question:  
                                    "Identify branches that have been active for over 5 years but have the lowest transaction count and also provide the possible reason for the decline."
                                - Extracted SQL Focus:  
                                    - "Identify branches that have been active for over 5 years but have the lowest transaction count."`  
                                - Ignored Part (For NLP Response, Not SQL):  
                                    - "Provide the possible reason for the decline."
                                                
                        Query Output Expectations:
                            - Return a syntactically correct SQL query according to Microsoft SQL Server standards.
                            - The final query should be free of syntax errors, logical mistakes.
                            - Final Output Must Contain Only the SQL Query with no additional explanation, commentary, or formatting. 

                        Example Questions and SQL Queries:- 
                            {fewshotprompting}                
                """),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{input}"),
            ]
        )
        generate_query = final_prompt | llm
        return generate_query
    except Exception as e:
        print(f"Error in get_chain: {e}")
        return None, None
        
def invoke_chain(question, memory, user, token_stream):
    try:
        permission = getTablenames(user)
        db = SQLDatabase(get_engine(),schema='bank' ,include_tables=permission, sample_rows_in_table_info=2) 

        sqlquery = get_chain()
        combined_filters = "No filters applied."
        session = get_session()
        try:
            user_access_query = """
                SELECT [access_level]
                FROM [Car Store DB].[dbo].[user_group]
                WHERE [username] = :username
            """
            access_level_df = pd.read_sql_query(
                sql=text(user_access_query),
                con=session.connection(),
                params={"username": user}
            )
        finally:
            session.close()


        if not access_level_df.empty and 'access_level' in access_level_df.columns:
            filter_list = []
            
            for i, row in access_level_df.iterrows():
                filter_list.append(f"Filter {i+1}:- {row['access_level']}")
            
            combined_filters = "\n".join(filter_list)

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{query}"),
            ]
        )
        notify_flag(token_stream, "success", "Running similarity search...", "message")
        selector = get_example_selector(user)
        
        if selector is None:
            raise RuntimeError("Example selector is not ready. Cannot proceed!")

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=selector,
            input_variables=["input"]
        )

        selected_examples = few_shot_prompt.format(input= question)
        notify_flag(token_stream, "success", "Crafting the SQL query...", "message")
        with get_openai_callback() as cb:
            sqlquery = sqlquery.invoke({
                "input": "Write a  query for MS SQL Server for '"+question + f"' Also user have only acess to this data so make sure to add this filter in the queries accordingly {combined_filters} ",
                "fewshotprompting": selected_examples,
                "table_info":db.get_context(),
                "messages": memory.chat_memory.messages
            })
            token_management(user, cb.total_tokens, cb.total_cost)
        sqlquery =  sqlquery.content.replace("```sql",'').replace("```",'')
        print("SQL Query", sqlquery)

        not_valid_dql_sqlquery = check_valid_dql(sqlquery)

        if not not_valid_dql_sqlquery:
            return " As an AI agent, I cannot modify data, but I can assist you with querying it. Let me know how I can help ", None, None

        not_allowed = list(set(all_tables) - set(permission))
        
        if not_allowed:
            for table in not_allowed:
                if table in sqlquery:
                    return "Table access is restricted, Please contact admin", None, None
                
        
        if not access_level_df.empty and 'access_level' in access_level_df.columns:
            
            query_prompt = f"""
                You are an SQL Expert responsible for validating and enforcing access control in SQL queries. Your task is to ensure that the provided SQL query correctly applies the necessary access level filters. If the query already enforces the required filters, return it unchanged. If the filters are missing, modify the query accordingly.
                Instructions:

                Understand Access Control  
                    - Users have access to all tables except where specific restrictions apply.  
                    - Restricted access conditions are defined as: `{combined_filters}`.

                Review the SQL Query Thoroughly  
                    - Check if the query already includes the necessary access level filters.  
                    - Identify any missing or incorrect filter conditions related to the access level.

                Modify the Query Only If Necessary  
                    - If the required filters are missing, update the query to enforce the correct access controls.  
                    - If the query already has the correct filters, return it unchanged.  
                    - Avoid duplicate or conflicting conditions—do not apply filters redundantly.

                Return Only the Final Query  
                    - If modifications were made, return the updated SQL query.  
                    - If no changes were necessary, return the original query as is.  
                    - Do not include any explanations, comments, or additional text.  

                Provided Input:
                    - User's Question: `{question}`
                    - SQL Query for Review: `{sqlquery}`
                    - Access Level Filters: `{combined_filters}`
                    - Database Model: `{db}`
            """

            response = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": query_prompt
                }],
                model="gpt-4.1-mini",
            )

            sqlquery = response.choices[0].message.content
            not_valid_dql_sqlquery = check_valid_dql(sqlquery)

            if not not_valid_dql_sqlquery:
                return "Data manipulation is not allowed.", None, None
        
        notify_flag(token_stream, "success", "Executing query and fetching data...", "message")
        session = get_session()
        try:
            df = pd.read_sql_query(sqlquery, session.connection())
        except Exception as e:
            print("Taking steps to solve the error")

            sqlquery = sql_error_handling(sqlquery, e, permission, db.get_context())

            if sqlquery:
                try:
                    session = get_session()
                    try:
                        df = pd.read_sql_query(sqlquery, session.connection())
                    except Exception:
                        memory.chat_memory.add_user_message(question)
                        memory.chat_memory.add_ai_message(e + "\n" + sqlquery)
                        return None, None, None
                    finally:
                        session.close()

                    if df.empty:
                        return "No relevant data found.", sqlquery, df
                    
                    df = identify_currency_columns(df, question, user)
                    
                    response =generate_response_from_dataframe(df, question, user)
                    memory.chat_memory.add_user_message(question)
                    memory.chat_memory.add_ai_message(response + df.head(5).to_csv(index=False))

                    return response, sqlquery, df
                
                except Exception as e:
                    print("Error in invoke chain after fixing sql", e)        
                    return None, None, None
        finally:
            session.close()

        if df.empty:
            return "No relevant data found.", sqlquery, df
        
        notify_flag(token_stream, "success", "Data retrieved successfully...", "message")

        df = identify_currency_columns(df, question, user)

        notify_flag(token_stream, "success", "Analyzing insights from the data...", "message")

        response =generate_response_from_dataframe(df, question, user)

        notify_flag(token_stream, "success", "Formulating your response...", "response")
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response + df.head(5).to_csv(index=False))

        return response, sqlquery, df
    except Exception as e:
        print("Error in invoke chain", e)
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(e + "\n" + sqlquery)
            
        return None, None, None
    
def get_error_chain():
    try:
        print("Creating error chain")
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                    You are a SQL Query Optimization Expert tasked with correcting SQL queries to ensure they are syntactically and logically perfect according to Microsoft SQL Server standards. Given an SQL query, an error message, and the database model, your job is to identify the error, understand the context, and provide a corrected SQL query.

                    Analyze the Provided SQL Query:
                        - Review the given SQL query to understand the error in it.

                    Understand the Error:
                        - Examine the provided error message related to the SQL query.
                        - Identify the type of error (syntax, logic, performance, or security).

                    Correlate with the Database Model:
                        - Consider the database schema to ensure the query uses the correct tables, columns, and relationships.
                        - Validate if the query adheres to the schema constraints.

                    Modify and Correct the SQL Query:
                        - Correct any syntactical errors to align with Microsoft SQL Server best practices.
                        - Adjust the logic of the query to ensure it fulfills the requirements and resolves the error.

                    Return the Corrected Query:
                        - Provide the revised SQL query that is free from the previous errors.
                        - Ensure the query is both syntactically and logically correct per the SQL Server standards.
                 
                    Expected Output:
                        - Return a syntactically correct SQL query according to Microsoft SQL Server standards.
                        - The final query should be free of syntax errors, logical mistakes.
                        - Final Output Must Contain Only the SQL Query with no additional explanation, commentary, or formatting.
        
                """),
                ("human", """
                        SQL Query: `{sql_query}`
                        \nError Message: `{error_message}`
                        \nDatabase Model: `{database_model}`
                """),
            ]
        )
        generate_query = final_prompt | llm
        return generate_query
    except Exception as e:
        print(f"Error in get_chain: {e}")
        return None, None
    
def sql_error_handling(sqlquery, error, permission, db_context):
    try:
        error_sqlquery = get_error_chain()
        corrected_sqlquery = error_sqlquery.invoke({
                "sql_query": sqlquery,
                "error_message": error,
                "database_model": db_context
        })
        corrected_sqlquery =  corrected_sqlquery.content.replace("```sql",'').replace("```",'')
        print("Corrected SQL Query", corrected_sqlquery)
        not_valid_dql_sqlquery = check_valid_dql(corrected_sqlquery)

        if not not_valid_dql_sqlquery:
            return " As an AI agent, I cannot modify data, but I can assist you with querying it. Let me know how I can help ", None, None

        not_allowed = list(set(all_tables) - set(permission))
        
        if not_allowed:
            for table in not_allowed:
                if table in sqlquery:
                    return "Table access is restricted, Please contact admin", None, None
                
        return sqlquery
    
    except Exception as e:
        print("Error in sql_error_handling", e)
        return None

def check_valid_dql(sql_query):
    sql_query = re.sub(r'--.*', '', sql_query)
    result = []
    i = 0
    depth = 0
    while i < len(sql_query):
        if sql_query[i:i+2] == '/*':
            depth += 1
            i += 2
            continue
        if sql_query[i:i+2] == '*/' and depth:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            result.append(sql_query[i])
        i += 1
    comments_removed_sql = ''.join(result).strip()

    dql_pattern = re.compile(r'^(SELECT|WITH|SHOW)\b', re.IGNORECASE)
    forbidden_patterns = [
        re.compile(r'\bINSERT\s+INTO\b', re.IGNORECASE),
        re.compile(r'\bUPDATE\s+[^\s]+\s+SET\b', re.IGNORECASE),
        re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE),
        re.compile(r'\bDROP\s+(TABLE|DATABASE)\b', re.IGNORECASE),
        re.compile(r'\bCREATE\s+(TABLE|DATABASE)\b', re.IGNORECASE),
        re.compile(r'\bTRUNCATE\s+TABLE\b', re.IGNORECASE),
        re.compile(r'\bALTER\s+TABLE\b', re.IGNORECASE),
        re.compile(r'\bGRANT\s+', re.IGNORECASE),
        re.compile(r'\bREVOKE\s+', re.IGNORECASE),
        re.compile(r'\bCOMMIT\b', re.IGNORECASE),
        re.compile(r'\bROLLBACK\b', re.IGNORECASE)
    ]
    
    stripped_query = comments_removed_sql.strip()
    if dql_pattern.match(stripped_query) and not any(pattern.search(stripped_query) for pattern in forbidden_patterns):
        return True
    else:
        return False

def token_management(user, total_tokens, total_cost):
    session = get_session()
    try:
        today_date = datetime.date.today()

        # Check if token management record exists
        query_select = """
            SELECT * FROM token_management
            WHERE username = :username AND token_day = :token_day
        """
        params_select = {"username": user, "token_day": today_date}
        result = session.connection().execute(text(query_select), params_select).fetchone()

        if result:
            # Update existing record
            query_update = """
                UPDATE token_management
                SET token_used = token_used + :token_used,
                    token_cost = token_cost + :token_cost
                WHERE username = :username AND token_day = :token_day
            """
            params_update = {
                "token_used": total_tokens,
                "token_cost": total_cost,
                "username": user,
                "token_day": today_date
            }
            session.connection().execute(text(query_update), params_update)
        else:
            # Fetch user's email
            query_email = """
                SELECT email FROM dbo.users
                WHERE username = :username
            """
            params_email = {"username": user}
            email_result = session.connection().execute(text(query_email), params_email).fetchone()
            email = email_result[0] if email_result else None

            # Insert new record
            query_insert = """
                INSERT INTO token_management (username, email, token_used, token_day, token_cost)
                VALUES (:username, :email, :token_used, :token_day, :token_cost)
            """
            params_insert = {
                "username": user,
                "email": email,
                "token_used": total_tokens,
                "token_day": today_date,
                "token_cost": total_cost
            }
            session.connection().execute(text(query_insert), params_insert)

        session.commit()
    except Exception as e:
        print(f"Error occurred: {e}")
        session.rollback()
    finally:
        session.close()


def apply_top_50(sqlquery):
    try:
        select_statements = list(re.finditer(r"(?i)SELECT\s+", sqlquery))
        if not select_statements:
            return sqlquery
        last_select_pos = select_statements[-1].start()
        if re.search(r"(?i)SELECT\s+TOP\s+\d+", sqlquery[last_select_pos:]):
            return sqlquery
        sqlquery = sqlquery[:last_select_pos] + sqlquery[last_select_pos:].replace("SELECT", "SELECT TOP 50", 1)
        return sqlquery
    except Exception as e:
        print(f"Error in apply_top_50: {e}")
        return sqlquery

def generate_response_from_dataframe(df, question, user):
    try:
        if df.empty:
            return "No data available for the query."
        
        dataset = df.to_dict(orient='records')
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        llm_chain = dataframe_summary | llm
        chain = None
        with get_openai_callback() as cb:        
            chain = llm_chain.invoke({"input": question, "dataframe_data": dataset})
            token_management(user, cb.total_tokens, cb.total_cost)

        return chain.content
    except Exception as e:
        print(f"An error occurred generating response from dataframe: {str(e)}")
        return ""
    
def identify_currency_columns(df, question, user):
    try:
        columns_to_check = df.columns.tolist()
        samples = df.head(5).to_dict(orient='records')
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        involved = LLMChain(llm=llm, prompt=identify_columns)
        
        with get_openai_callback() as cb:
            chain = involved.invoke({
                "input": "Identify the columns that contain currency-related data, percentage data, and count data.",
                "User_question": question,
                "sample_data": samples,
                "columns": columns_to_check
            })
            
            token_management(user, cb.total_tokens, cb.total_cost)

        try:

            response = json.loads(chain['text'].replace("```json", "").replace("```", ""))
            
            currency_columns = response.get('currency_columns', [])
            percentage_columns = response.get('percentage_columns', [])
            count_columns = response.get('count_columns', [])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return df

        # Fill NaN values with 0
        df = df.fillna(0)

        # Format currency columns
        if currency_columns:
            for column in currency_columns:
                if column in df.columns:
                    df[column] = df[column].apply(
                        lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x
                    )

        # Format percentage columns
        if percentage_columns:
            for column in percentage_columns:
                if column in df.columns:
                    df[column] = df[column].apply(
                        lambda x: f"{x}%" if isinstance(x, (int, float)) else x
                    )

        return df

    except Exception as e:
        print(f"Error in identify_currency_columns: {e}")
        return df

def memory_to_openai_messages(memory):
    messages = []
    
    if memory:
        for m in memory.chat_memory.messages:
            if m.type == "human":
                messages.append({"role": "user", "content": m.content})
            elif m.type == "ai":
                messages.append({"role": "assistant", "content": m.content})
    
    return messages

def greet(memory):
    try:
        print("Creating greet chain")
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
        chain = ConversationChain(
            prompt=greet_prompt,
            llm=llm,
            memory=memory,
            verbose=False
        )
        return chain
    except Exception as e:
        print(f"Error in greet: {e}")
        return None

def invoke_chain_greet(question, memory, user):
    try:
        chain = greet(memory)
        if chain is None:
            return "Error: Could not initialize conversation chain."
        response = ""
        with get_openai_callback() as cb:    
            response = chain.run(question)
            token_management(user, cb.total_tokens, cb.total_cost)

        return response
    except Exception as e:
        print("Error in invoke_chain_greet", e)

def log_interaction(user_prompt, ai_response, user_ip, classification_result):

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    
    try:
        user_hostname = socket.gethostbyaddr(user_ip)[0]
    except socket.herror:
        user_hostname = "Unknown Hostname"
    
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"User_IP: {user_ip}, User_Hostname: {user_hostname}, Mode: {classification_result}\n")
        f.write(f"User: {user_prompt}\n")
        f.write(f"AI: {ai_response}\n\n")

def invoke_chain_visualise(token_stream, question, classification, user, memory=None, tables_involved=None):  

    if tables_involved is None:
        permission = getTablenames(user)
    else:
        permission = tables_involved

    db = SQLDatabase(get_engine(), schema='bank', include_tables=permission, sample_rows_in_table_info=2) 

    sqlquery = get_chain()
    combined_filters = "No filters applied."

    user_access_query = f"""
        SELECT [access_level]
        FROM [Car Store DB].[dbo].[user_group]
        WHERE [username] = '{user}'
    """
    
    session = get_session()
    try:
        user_access_query = """
            SELECT [access_level]
            FROM [Car Store DB].[dbo].[user_group]
            WHERE [username] = :username
        """
        access_level_df = pd.read_sql_query(
            sql=text(user_access_query),
            con=session.connection(),
            params={"username": user}
        )
    finally:
        session.close()


    if not access_level_df.empty and 'access_level' in access_level_df.columns:
        filter_list = []
        
        for i, row in access_level_df.iterrows():
            filter_list.append(f"Filter {i+1}:- {row['access_level']}")
        
        combined_filters = "\n".join(filter_list)

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{query}"),
        ]
    )
    notify_flag(token_stream, "success", "Running similarity search...", "message")
    selector = get_example_selector(user)
        
    if selector is None:
        raise RuntimeError("Example selector is not ready. Cannot proceed!")

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=selector,
        input_variables=["input"]
    )

    selected_examples = few_shot_prompt.format(input=question)
    notify_flag(token_stream, "success", "Crafting the SQL query...", "message")
    with get_openai_callback() as cb:
        sqlquery = sqlquery.invoke({
            "input": "Write a query for MS SQL Server for '" + question + f"' Also user have only access to this data so make sure to add this filter in the queries accordingly {combined_filters} ",
            "fewshotprompting": selected_examples,
            "table_info": db.get_context(),
            "messages": memory.chat_memory.messages
        })
        token_management(user, cb.total_tokens, cb.total_cost)
    sqlquery = sqlquery.content.replace("```sql", '').replace("```", '')
    print("SQL Query", sqlquery)

    not_valid_dql_sqlquery = check_valid_dql(sqlquery)

    if not not_valid_dql_sqlquery:
        return "As an AI agent, I cannot modify data, but I can assist you with querying it. Let me know how I can help", None, None, None

    not_allowed = list(set(all_tables) - set(permission))
    print(not_allowed)
    
    if not_allowed:
        for table in not_allowed:
            if table in sqlquery:
                return "Table access is restricted, Please contact admin", None, None, None
            
    if not access_level_df.empty and 'access_level' in access_level_df.columns:
        query_prompt = f"""
            You are an SQL Expert responsible for validating and enforcing access control in SQL queries. Your task is to ensure that the provided SQL query correctly applies the necessary access level filters. If the query already enforces the required filters, return it unchanged. If the filters are missing, modify the query accordingly.

            Instructions:

            Understand Access Control  
                - Users have access to all tables except where specific restrictions apply.  
                - Restricted access conditions are defined as: `{combined_filters}`.

            Review the SQL Query Thoroughly  
                - Check if the query already includes the necessary access level filters.  
                - Identify any missing or incorrect filter conditions related to the access level.

            Modify the Query Only If Necessary  
                - If the required filters are missing, update the query to enforce the correct access controls.  
                - If the query already has the correct filters, return it unchanged.  
                - Avoid duplicate or conflicting conditions—do not apply filters redundantly.

            Return Only the Final Query  
                - If modifications were made, return the updated SQL query.  
                - If no changes were necessary, return the original query as is.  
                - Do not include any explanations, comments, or additional text.  

            Provided Input:
                - User's Question: `{question}`
                - SQL Query for Review: `{sqlquery}`
                - Access Level Filters: `{combined_filters}`
                - Database Model: `{db}`
        """

        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": query_prompt
            }],
            model="gpt-4.1-mini",
        )
        sqlquery = response.choices[0].message.content
        not_valid_dql_sqlquery = check_valid_dql(sqlquery)

        if not not_valid_dql_sqlquery:
            return "Data manipulation is not allowed.", None, None
        
    notify_flag(token_stream, "success", "Executing query and fetching data...", "message")   
    session = get_session()
    try:
        df = pd.read_sql_query(sqlquery, session.connection())
    except Exception:
        return None, None, None, None
    finally:
        session.close()

    if df.empty:
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message("No relevant data found.")
        return "No relevant data found.", sqlquery, ""
    
    if classification != "Explore":
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(df.head(5).to_csv(index=False))
    
    try:
        notify_flag(token_stream, "success", "Analyzing insights from the data...", "message")
        nlp_response = generate_response_from_dataframe(identify_currency_columns(df, question, user), question, user)

        notify_flag(token_stream, "success", "Generating visualization...", "message")
        visualization_meta = generate_visualization_meta(df.head(5).to_dict(orient='list'), question, user)

        visual_html = df.fillna('0').head(50).to_dict(orient='list')

    except Exception as e:
        print("Error in invoke_chain_visualise", e)
        return None, None, None, None
    
    return nlp_response, sqlquery, visual_html, visualization_meta

import re
from fuzzywuzzy import fuzz
import json

def generate_visualization_meta(df, question, user):
    try:
        dataset = df
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

        patterns = {
            'bar': r'\b(bar chart|bar graph|bar plot|histogram|grouped bar|stacked bar)\b',
            'line': r'\b(line chart|line graph|line plot|trend chart|trend graph|trend plot|time series chart|time series graph|time series plot)\b',
            'scatter': r'\b(scatter chart|scatter graph|scatter plot|dot chart|dot graph|dot plot|point chart|point graph|point plot|bubble chart|bubble plot)\b',
            'pie': r'\b(pie chart|pie graph|pie plot|donut chart|donut graph|donut plot|proportion chart|proportion graph|proportion plot)\b',
            'radar': r'\b(radar chart|radar graph|radar plot|spider chart|spider graph|spider plot|web chart|web graph|web plot)\b',
            'area': r'\b(area chart|area graph|area plot|stacked area chart|stacked area graph|stacked area plot)\b',
            'column': r'\b(column chart|column graph|column plot|vertical bar chart|vertical bar graph|vertical bar plot)\b'
        }

        v_type_s = None
        for v_type, pattern in patterns.items():
            if re.search(pattern, question.lower()):
                v_type_s = v_type
                break

        if not v_type_s:
            question_lower = question.lower()
            best_score = 0
            for v_type, pattern in patterns.items():
                common_terms = pattern.replace(r'\b', '').split('|')
                for term in common_terms:
                    score = fuzz.partial_ratio(term, question_lower)
                    if score > best_score and score > 80:
                        best_score = score
                        v_type_s = v_type

        if not v_type_s:
            v_type_s = 'bar'
        print("Visualization Type:", v_type_s)

        visualization_meta = ChatPromptTemplate.from_messages([
            ("system", """
            You are a data visualization assistant. Your task is to analyze the given question and dataframe, and determine the appropriate visualization type and axis mappings.

            Instructions:
            - Identify the visualization type from the list: [bar, line, scatter, pie, radar, column].
            - Determine which column should be mapped to the x-axis and y-axis.
            - If there are multiple y-axis columns, include them under "more_y_columns" these columns are required to answer the user questions like showing all the other fields as examples.
            - Ensure the response is strictly in JSON format with the following structure:
              {{
                  "v_type": "<visualization_type>",
                  "x_axis": "<column_name>",
                  "y_axis": "<column_name>",
                  "more_y_columns": ["<column_name_1>", "<column_name_2>", ...]
              }}
            - Do not include any additional text, explanations, or comments in the response.

            Example:
            Question: "Draw a line chart to show sales of 2021 to 2022."
            Dataframe:
            {{
                "transaction_month": [1, 2, 3, 4, 5],
                "transaction_2021": [132, 123, 123, 1234, 342],
                "transaction_2022": [132, 123, 123, 1234, 342]
            }}
            Expected Output:
            {{
                "v_type": "line",
                "x_axis": "transaction_month",
                "y_axis": "transaction_2021",
                "more_y_columns": ["transaction_2021", "transaction_2022"]
            }}
            """),
            ("human", """
            Analyze the question and dataframe to provide the visualization metadata in the specified JSON format.
                Question: {input}
                Dataframe data: {dataframe_data}
            """)
        ])

        llm_chain = visualization_meta | llm
        chain = None
        with get_openai_callback() as cb:        
            chain = llm_chain.invoke({"input": question, "dataframe_data": dataset})
            token_management(user, cb.total_tokens, cb.total_cost)

        # Parse LLM response
        response_content = chain.content.replace("```json", "").replace("```", "").strip()
        try:
            response_json = json.loads(response_content)
        except json.JSONDecodeError:
            print(f"Error parsing LLM response as JSON: {response_content}")
            return ""

        # Compare LLM v_type with pattern-matched v_type_s
        if response_json.get('v_type') != v_type_s:
            print(f"LLM v_type ({response_json['v_type']}) differs from pattern-matched v_type_s ({v_type_s}). Updating to {v_type_s}.")
            response_json['v_type'] = v_type_s

        # Ensure the response is in valid JSON format
        final_response = json.dumps(response_json)
        print("Visualization Meta:", final_response)
        return final_response

    except Exception as e:
        print(f"An error occurred generating response from dataframe: {str(e)}")
        return ""
    
def process_csv(files, prompt, memory):
    try:
        combined_content = ""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        for i, file in enumerate(files):
            if hasattr(file, 'read'):
                file_content = file.read().decode("utf-8")  
            else:
                raise ValueError("The uploaded file is not in the expected format.")
            
            with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            try:
                loader = TextLoader(tmp_file_path)
                data = loader.load()

                if data:
                    combined_content += f"CSV File {i+1}: {file.filename}\n"
                    for doc in data:
                        combined_content += doc.page_content + "\n\n"
                else:
                    combined_content += f"CSV File {i+1}: {file.filename}\nNo content extracted.\n\n"
            except Exception as e:
                combined_content += f"CSV File {i+1}: {file.filename}\nError processing the file: {str(e)}\n\n"
            os.remove(tmp_file_path)
        if prompt:
            try:
                document = Document(page_content=combined_content)
                chain = load_qa_chain(llm, chain_type="stuff")
                csv_response = chain.run(input_documents=[document], question=f"{prompt}. Also specify the row/column from where you got the information. Please provide the result in a concise and well-structured markdown format. Ensure clarity and organization, with no unnecessary markdown elements, such as headers.")
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(re.sub(r'```markdown', ' ', csv_response).replace("```", ""))
                return re.sub(r'```markdown', ' ', csv_response).replace("```", "")
            except Exception as e:
                return f"An error occurred during CSV processing: {str(e)}"
        else:
            return "No prompt provided to generate a response."

    except Exception as e:
        return f"An error occurred in processing csv: {str(e)}"

def process_pdf(files, prompt, memory):
    try:
        combined_content = ""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        for i, file in enumerate(files):
            if hasattr(file, 'read'):
                file_content = file.read()
            else:
                raise ValueError("The uploaded file is not in the expected format.")
            
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_file_path, extract_images=True)
                data = loader.load()

                if data:
                    combined_content += f"PDF File {i+1}: {file.filename}\n"
                    for doc in data:
                        combined_content += doc.page_content + "\n\n"
                else:
                    combined_content += f"PDF File {i+1}: {file.filename}\nNo content extracted.\n\n"
            except Exception as e:
                combined_content += f"PDF File {i+1}: {file.filename}\nError processing the file.\n\n"
            
            os.remove(tmp_file_path)

        if prompt:
            try:
                document = Document(page_content=combined_content)
                chain = load_qa_chain(llm, chain_type="stuff")
                pdf_response = chain.run(input_documents=[document], question=prompt + " Please provide the result in a concise and well-structured markdown format. For example for comparision use the table format etc. Ensure clarity and organization, with no unnecessary markdown elements, such as headers")
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(re.sub(r'```markdown', ' ', pdf_response).replace("```", ""))
                return re.sub(r'```markdown', ' ', pdf_response).replace("```", "")
            except Exception as e:
                return f"An error occurred during PDF processing: {str(e)}"

    except Exception as e:
        return f"An error occurred in processing pdf: {str(e)}"

def process_ppt(files, prompt, memory):
    try:
        combined_content = ""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        for i, file in enumerate(files):
            if not hasattr(file, 'read'):
                raise ValueError("The uploaded file is not in the expected format.")
            
            file_content = file.read()

            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            try:

                loader = UnstructuredPowerPointLoader(tmp_file_path, extract_images=True)
                data = loader.load()

                if data:
                    combined_content += f"PowerPoint File {i+1}: {file.filename}\n"
                    for doc in data:
                        combined_content += doc.page_content + "\n\n"
                else:
                    combined_content += f"PowerPoint File {i+1}: {file.filename}\nNo content extracted.\n\n"
            except Exception as e:
                combined_content += f"PowerPoint File {i+1}: {file.filename}\nError processing the file: {str(e)}\n\n"
            os.remove(tmp_file_path)
        if prompt:
            try:
                document = Document(page_content=combined_content)
                chain = load_qa_chain(llm, chain_type="stuff")
                ppt_response = chain.run(input_documents=[document], question=prompt + " Please provide the result in a concise and well-structured markdown format. Ensure clarity and organization, with no unnecessary markdown elements, such as headers.")
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(re.sub(r'```markdown', ' ', ppt_response).replace("```", ""))
                return re.sub(r'```markdown', ' ', ppt_response).replace("```", "")
            except Exception as e:
                return f"An error occurred during PowerPoint processing: {str(e)}"
        else:
            return "No prompt provided to generate a response."
    except Exception as e:
        return f"An error occurred in processing ppt: {str(e)}"

def process_excel(files, prompt, memory):
    try:
        combined_content = ""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        for i, file in enumerate(files):
            if hasattr(file, 'read'):
                file_content = file.read()
            else:
                raise ValueError("The uploaded file is not in the expected format.")
            
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            try:
                loader = UnstructuredExcelLoader(tmp_file_path, mode="elements")
                data = loader.load()

                if data:
                    combined_content += f"Excel File {i+1}: {file.filename}\n"
                    for doc in data:
                        combined_content += doc.page_content + "\n\n"
                else:
                    combined_content += f"Excel File {i+1}: {file.filename}\nNo content extracted.\n\n"
            except Exception as e:
                combined_content += f"Excel File {i+1}: {file.filename}\nError processing the file: {str(e)}\n\n"
            os.remove(tmp_file_path)

        if prompt:
            try:
                document = Document(page_content=combined_content)
                chain = load_qa_chain(llm, chain_type="stuff")
                excel_response = chain.run(input_documents=[document], question=prompt + " Please provide the result in a concise and well-structured markdown format. Ensure clarity and organization, with no unnecessary markdown elements, such as headers.")
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(re.sub(r'```markdown', ' ', excel_response).replace("```", ""))
                return re.sub(r'```markdown', ' ', excel_response).replace("```", "")
            except Exception as e:
                return f"An error occurred during Excel processing: {str(e)}"

    except Exception as e:
        return f"An error occurred in processing excel : {str(e)}"

def process_image(files, prompt, memory):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    print("Processing Image",files)
    image_urls = []
    for file in files:
        image_data = file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        image_urls.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg/png;base64,{encoded_image}"
            }
        })
    if "dashboard" in prompt:
        llm_context = """
            - When the user provides an image, treat the image content as directly accessible and analyze it as if it were data presented in any other form.
            - Analyze the data presented in the dashboard carefully.
            - Extract all key insights by summarizing trends, anomalies, or significant data points.
            - Provide actionable recommendations based on the insights, clearly explaining how the data supports each suggestion.
            - Ensure the analysis is thorough and detailed.
            - Please provide the result in a concise and well-structured markdown format. Ensure clarity and organization, with no unnecessary markdown elements, such as headers.
            """
    else:
        llm_context = """
            - When the user provides an image, treat the image content as directly accessible and analyze it as if it were data presented in any other form.
            - Analyze the image based on the provided prompt or context.
            - Respond accurately and appropriately to what is requested in the user prompt.
            - Please provide the result in a concise and well-structured markdown format. Ensure clarity and organization, with no unnecessary markdown elements, such as headers.
        """
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "User's prompt:-"+prompt + llm_context }] + image_urls
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()
    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(re.sub(r'```markdown', ' ', result['choices'][0]['message']['content']).replace("```", ""))
    return re.sub(r'```markdown', ' ', result['choices'][0]['message']['content']).replace("```", "") 

def process_media(token_stream, grouped_files, prompt, memory):
    try:
        notify_flag(token_stream, "success", "Analyzing the uploaded media...", "message")
        pdf_files = [file.filename for file in grouped_files["pdf"]]
        image_files = [file.filename for file in grouped_files["image"]]
        excel_files = [file.filename for file in grouped_files["excel"]]
        ppt_files = [file.filename for file in grouped_files["ppt"]]
        csv_files = [file.filename for file in grouped_files["csv"]]

        file_groups = {
            "pdf": pdf_files,
            "image": image_files,
            "excel": excel_files,
            "ppt": ppt_files,
            "csv": csv_files
        }

        non_empty_groups = {key: files for key, files in file_groups.items() if files}
        
        if len(non_empty_groups) == 1 and len(next(iter(non_empty_groups.values()))) == 1:
            file_type = next(iter(non_empty_groups.keys()))
            print(f"Directly classified as {file_type}")
            if "pdf" == file_type:
                return process_pdf(grouped_files["pdf"], prompt, memory)
            elif "image" == file_type:
                return process_image(grouped_files["image"], prompt, memory)
            elif "csv" == file_type:
                return process_csv(grouped_files["csv"], prompt, memory)
            elif "excel" == file_type:
                return process_excel(grouped_files["excel"], prompt, memory)
            elif "ppt" == file_type:
                return process_ppt(grouped_files["ppt"], prompt, memory)
            else:
                return "I couldn't determine which files should be processed. Could you please elaborate more on your request?"

        llm_prompt = f"""
        I have the following uploaded files:
        - PDF files: {', '.join(pdf_files) if pdf_files else 'None'}
        - Image files: {', '.join(image_files) if image_files else 'None'}
        - Excel files: {', '.join(excel_files) if excel_files else 'None'}
        - PowerPoint files: {', '.join(ppt_files) if ppt_files else 'None'}
        - CSV files: {', '.join(csv_files) if csv_files else 'None'}

        Strictly follow this rule :- If only one type of file is mentioned and others are listed as "None," then classify it directly as that file type.
        Based on the following prompt, decide which group of files should be processed. Respond with a single word: one of "image", "pdf", "excel", "ppt", or "csv". If the prompt is too generalized (e.g., "Describe the content of file") and you are unsure which file type to process, respond with "unknown".
        Prompt: '{prompt}'
        """

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1)
        prompt_template = PromptTemplate(input_variables=["llm_prompt"], template="{llm_prompt}")
        chain = LLMChain(llm=llm, prompt=prompt_template)
        classification_result = chain.run({"llm_prompt": llm_prompt}).strip().lower()
        print("Classified as", classification_result)

        if "pdf" in classification_result:
            return process_pdf(grouped_files["pdf"], prompt, memory)
        elif "image" in classification_result:
            return process_image(grouped_files["image"], prompt, memory)
        elif "csv" in classification_result:
            return process_csv(grouped_files["csv"], prompt, memory)
        elif "excel" in classification_result:
            return process_excel(grouped_files["excel"], prompt, memory)
        elif "ppt" in classification_result:
            return process_ppt(grouped_files["ppt"], prompt, memory)
        else:
            return "I couldn't determine which files should be processed. Could you please elaborate more on your request?"
    except Exception as e:
        print("Error Occurred", e)


def get_json_schema_from_engine(engine, schema='dbo', tables=None):
    inspector = inspect(engine)
    result = {}
    table_names = tables if tables else inspector.get_table_names(schema=schema)

    for table in table_names:
        columns = inspector.get_columns(table, schema=schema)
        col_def = {
            col["name"]: str(col["type"]).split(" COLLATE")[0].strip()
            for col in columns
        }
        result[table] = col_def

    return result


def generate_data_analysis_questions(user_prompt, user):
    if not user_prompt:
        raise ValueError("User prompt cannot be empty!")

    engine = get_engine()
    schema = 'bank'
    tables = getTablenames(user)
    json_schema = get_json_schema_from_engine(engine, schema=schema, tables=tables)
    db_context = json.dumps(json_schema, indent=4)

    system_prompt = """
    You are a Transactional Data Analyst Assistant helping build Power BI report using SQL-based datasets.

    Your task is to generate exactly 4 **data visualization** questions based on:
    - The user’s business prompt
    - The given database schema

    Make sure each question:
    - Sounds like a **real analyst or business manager** would ask it — keep phrasing natural and decision-driven
    - Is focused on **business trends, financial metrics, customer behavior, and strategic comparisons**
    - Avoids questions about metadata (e.g., launch dates, product status, or product descriptions, or active)
    - Refers to a **distinct insight** — such as growth trends, top segments, or performance 
    - Groups by **categorical fields** (e.g., ProductType, Country, Month, CustomerSegment, Year)
    - Clearly mentions a supported **visualization type** in the phrasing (Line chart, Bar chart, Column chart, Area chart, Pie chart, Scatter plot)
    - Can be **implemented directly using SQL and Power BI** without complex modeling or DAX

    Keep the tone professional but natural — imagine a data-savvy business user exploring financial KPIs and behavior.

    Sample response:
    User Question :- Give me Overview of the products
    Response:-
    {
        "questions": [
            "Draw a line chart showing the monthly trend of total TransactionAmount grouped by ProductType.",
            "Draw a bar chart comparing the average TransactionAmount across CustomerSegments to identify high-spending groups.",
            "Draw a pie chart showing the distribution of total TransactionCount by ProductType for the current year.",
            "Draw a column chart showing the average InterestRate by ProductType to evaluate product profitability."
        ]
    }
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_prompt}\n\nDatabase Schema:\n{db_context}"}
        ]
    )

    try:
        content = response.choices[0].message.content
        questions_json = json.loads(content)
        return questions_json["questions"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError("Invalid JSON format received from LLM.") from e



def invoke_explore(token_stream, user_prompt, classification, memory, user):
    try:
        # Generate the shortlisted questions
        explore_prompt_list = generate_data_analysis_questions(user_prompt, user)
        print("Shortlisted Questions:", explore_prompt_list)

        tables_involved = getTablenames(user)
        
        # Initialize result lists
        f_nlp_res = [None] * len(explore_prompt_list)
        f_sql_res = [None] * len(explore_prompt_list)
        f_vis_res = [None] * len(explore_prompt_list)
        f_vmeta_res = [None] * len(explore_prompt_list)
        
        # Function to process a single question
        def process_question(idx, question, classification, user, memory):
            try:
                # Call the visualization chain using shared engine
                f_nlp, f_sql, f_vis, f_vmeta = invoke_chain_visualise(token_stream, question, classification, user, memory, tables_involved)
                
                if f_nlp == "No relevant data found.":
                    f_nlp, f_sql, f_vis, f_vmeta = "", "", "", ""
                
                return idx, (f_nlp, f_sql, f_vis, f_vmeta)
            except Exception as e:
                print(f"Error processing question {question}: {e}")
                return idx, ("", "", "", "")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(process_question, idx, question, classification, user, memory)
                for idx, question in enumerate(explore_prompt_list)
            ]
            
            # Wait for all futures to complete and collect results
            for future in concurrent.futures.as_completed(futures):
                idx, (f_nlp, f_sql, f_vis, f_vmeta) = future.result()
                f_nlp_res[idx] = f_nlp
                f_sql_res[idx] = f_sql
                f_vis_res[idx] = f_vis
                f_vmeta_res[idx] = f_vmeta

        tables_involved = get_involved_tables(user_prompt, f_sql_res, user)
        explore_summary = generate_explore_summary(" ".join(f_nlp_res), user_prompt, user)
        
        return f_vis_res, tables_involved, explore_summary, f_vmeta_res
    
    except Exception as e:
        print(f"Error occurred in Explore: {e}")
        return None, None, None, None

def get_involved_tables(user_query, sql_queries, user):
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        involved = LLMChain(llm=llm, prompt=tables_involved)

        sql_queries_context = "\n".join([str(q) for q in sql_queries if q is not None])

        chain = None
        with get_openai_callback() as cb:
            chain = involved.invoke({"input": user_query, "sql_queries": sql_queries_context})
            token_management(user, cb.total_tokens, cb.total_cost)

        return chain['text']
    except Exception as e:
        print(f"An error occurred in tables involved: {e}")
        return "An error occurred while processing your request."


def generate_explore_summary(summary_text, user_prompt, user):
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        involved = LLMChain(llm=llm, prompt=explore_summary_prompt)
        chain = None
        with get_openai_callback() as cb:
            chain = involved.invoke({"summary_text": summary_text, "input":
                                user_prompt + """,Extract all key insights by summarizing trends, anomalies, or significant data points.
                                                - Provide actionable recommendations based on the insights, clearly explaining how the data supports each suggestion.
                                                - Ensure the analysis is thorough and detailed.
                                                - Please provide the result in a pointwise and well-structured markdown format. Ensure clarity and organization, with no unnecessary markdown elements, such as headers."""})
            print("Token Management 1",user)
            token_management(user, cb.total_tokens, cb.total_cost)
            
        return chain['text'] if chain.get('text') else "No summary text available." 
    except Exception as e:
        print("Error in Explore summary:", e)
        return "An error occurred while generating the explore summary."  
    
def speech_to_text(audio_file_in_memory):
    try:
        api_url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        audio_file_in_memory.seek(0)
        files = {
            'file': ('temp_audio.mp3', audio_file_in_memory, 'audio/mp3')
        }
        data = {
            'model': 'whisper-1',
            'temperature': 0,
            'response_format': 'text'
        }
        response = requests.post(api_url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error in transcription: {response.text}")
            return None
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None