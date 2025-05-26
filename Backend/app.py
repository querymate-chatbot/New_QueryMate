from datetime import datetime, timedelta
import sys
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv, find_dotenv
import os
import redis
import hashlib
import json
import io
import requests
import time
import uuid
from werkzeug.security import generate_password_hash
import secrets
import jwt
from sqlalchemy import text
from flask import render_template, request, jsonify 
from werkzeug.security import check_password_hash
import warnings
from langdetect import detect
from langchain_utils import (
    invoke_chain,
    invoke_chain_greet,
    invoke_chain_visualise,
    log_interaction,
    process_media,
    invoke_explore,
    lang,
    speech_to_text,
    notify_flag
    )
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from functools import wraps
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from openai import OpenAI
from db_config import get_session, get_engine
from services import add_new_example_for_training, initialize_vector_store
from cachetools import TTLCache


app = Flask(__name__)
CORS(app, resources={"*"})
cache = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

Talisman(app, content_security_policy={
    'default-src': "'self'",
    'img-src': "'self' data:",
    'script-src': "'self'",
    'style-src': "'self' 'unsafe-inline'",
})

app.secret_key = secrets.token_hex(256)
SECRET_KEY = "secrets.token_hex(256)"
app.register_blueprint(lang)

warnings.filterwarnings("ignore")

_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"status": "error", "message": "Token is missing"}), 403
        try:
            token = token.replace("Bearer ", "")
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = decoded_token["sub"]  
        except jwt.ExpiredSignatureError:
            return jsonify({"status": "error", "message": "Token has expired"}), 403
        except jwt.InvalidTokenError:
            return jsonify({"status": "error", "message": "Invalid token"}), 403

        return f(*args, **kwargs)
    return decorated

user_memories = TTLCache(maxsize=20, ttl=3600)

def get_user_memory(session_user_id):

    if session_user_id is None:
        return None

    if session_user_id not in user_memories:
        user_memories[session_user_id] = ConversationBufferWindowMemory(k=5, return_messages=True)

    return user_memories[session_user_id]

def process_classification_result(token_stream, user_prompt, classification_result, user, uploaded_files=None, memory=None):

    response_data = {"status": "success", "response": "", "sql": "", "results": "", "table_html": ""}
    try:
        if classification_result == "Database":
            response, sqlquery, df = invoke_chain(user_prompt, memory, user, token_stream)
            detected_language = detect(response)

            if sqlquery:
                response_data.update({
                    "response": response,
                    "sql": sqlquery,
                    "results": df.to_dict(orient="records"),
                    "table_html": df.to_html(classes='styled-table', index=False, escape=False),
                    "detected_language": detected_language,
                    "feedmode": classification_result
                })
            else:
                response_data["response"] = '<i class="fa fa-warning" style="color: #e9d502;"></i>' + " " + response + " " + '<i class="fa fa-warning" style="color: #e9d502;"></i>'
                response_data["detected_language"] = detected_language

        elif classification_result == "Chat":
            response = invoke_chain_greet(user_prompt, memory, user)
            detected_language = detect(response)
            response_data["response"] = response
            response_data["detected_language"] = detected_language

        elif classification_result == "Explore":
            dash_vis, dash_tab, dash_sum, vis_meta = invoke_explore(token_stream, user_prompt, classification_result, memory, user)
            response_data.update({
                "explore_html_1": dash_vis[0],
                "explore_html_2": dash_vis[1],
                "explore_html_3": dash_vis[2],
                "explore_html_4": dash_vis[3],
                "visualization_meta_1": vis_meta[0],
                "visualization_meta_2": vis_meta[1],
                "visualization_meta_3": vis_meta[2],
                "visualization_meta_4": vis_meta[3],
                "involved_tables": dash_tab.replace("Tables Involved: ", " "),
                "explore_summary": dash_sum
            })

        elif classification_result == "Visualization":
            nlp_response, sqlquery, visualization_html, visualization_meta = invoke_chain_visualise(token_stream, user_prompt, classification_result, user, memory)
            detected_language = detect(nlp_response)
            response_data.update({
                "response": nlp_response,
                "sql": sqlquery,
                "visualization_html": visualization_html,
                "visualization_meta": visualization_meta,
                "detected_language": detected_language,
                "feedmode": classification_result
            })

        elif classification_result == "File" or classification_result == "Media":
            response_data_temp = ""

            grouped_files = {
                "pdf": [],
                "image": [],
                "excel": [],
                "ppt": [],
                "csv": []
            }

            for uploaded_file in uploaded_files:
                if uploaded_file.filename.endswith(".pdf"):
                    grouped_files["pdf"].append(uploaded_file)
                elif uploaded_file.filename.endswith((".png", ".jpg", ".jpeg")):
                    grouped_files["image"].append(uploaded_file)
                elif uploaded_file.filename.endswith((".xlsx", ".xls")):
                    grouped_files["excel"].append(uploaded_file)
                elif uploaded_file.filename.endswith((".ppt", ".pptx")):
                    grouped_files["ppt"].append(uploaded_file)
                elif uploaded_file.filename.endswith(".csv"):
                    grouped_files["csv"].append(uploaded_file)
                else:
                    response_data_temp = f"Unsupported file type: {uploaded_file.filename}"

            if grouped_files:
                response_data_temp = process_media(token_stream, grouped_files, user_prompt, memory)

            response_data["response"] = response_data_temp

        else:
            response_data = {"status": "error", "message": "No file uploaded!"}

        user_ip = request.remote_addr
        log_interaction(user_prompt, response_data["response"], user_ip, classification_result)

    except Exception as e:
        response_data = {"status": "error", "message": f"Error: {str(e)}"}

    return response_data


def get_user_token():
    user_id = getattr(request, "user", None)    
    return str(user_id) if user_id else get_remote_address()

limiter = Limiter(
    key_func=get_user_token,
    app=app,
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "status": "success",
        "response": "Please wait some time before sending another request."
    }), 429

@app.route("/process", methods=["POST"])
@token_required
@limiter.limit("20 per minute") 
def process_input():
    try:
        user_id = request.user
        user_prompt = request.form.get("prompt", "")
        session_user_id = request.form.get("session_user_id", "")
        uploaded_file = request.files.getlist("file")
        classification_result = request.form.get("selected_option", "")
        token_stream = request.headers.get("Authorization").replace("Bearer ", "")

        if not user_prompt and not uploaded_file:
            return jsonify({"status": "error", "message": "Please provide a prompt or upload a file!"})

        notify_flag(token_stream, "success", "Analyzing your question...", "message")

        start_time = time.time()
        try:
            memory = get_user_memory(session_user_id)
            result = process_classification_result(token_stream, user_prompt, classification_result, user_id, uploaded_file, memory)

            end_time = time.time()
            print(f"Processing Time: {end_time - start_time:.4f} seconds")

            response = jsonify(result)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'

            notify_flag(token_stream, "success", "Processing Completed", "message")
            return response

        except Exception as e:
            print("Error occurred in the process", e)
            notify_flag(token_stream, "success", f"Error occurred: {str(e)}", "message")
            return jsonify({"status": "error", "message": f"Error processing input: {str(e)}"})

    except Exception as e:
        print("Error occurred in the process", e)
        return jsonify({"status": "error", "message": f"Error processing input: {str(e)}"})
    
@app.route('/process_audio', methods=['POST'])  
@token_required
def process_audio():
    try:
        audio_file = request.files.get('audio')
        
        if audio_file:
            audio_data = audio_file.read()
            audio_file_in_memory = io.BytesIO(audio_data)

            transcript = speech_to_text(audio_file_in_memory)

            if transcript:
                return jsonify({'status': 'success', 'transcript': transcript})
            else:
                return jsonify({'status': 'error', 'message': 'Error in transcription.'})

        else:
            return jsonify({'status': 'error', 'message': 'No audio file received'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})
    
@app.route('/')
def hello():
    return "Server is up and running!"

@app.route("/clear_memory", methods=["POST"])
@token_required
def clear_memory():
    user = request.json.get("session_user_id")
    try:
        if user in user_memories:
            del user_memories[user]
        return jsonify({"status": "success", "message": "Memory Cleared"}), 200
    except Exception as e:
        print("Error occurred while clearing memory", e)
        return jsonify({"status": "error", "message": f"Error clearing memory: {str(e)}"}), 500

@app.route('/submit_feedback', methods=['POST'])
@token_required
def submit_feedback():
    session = get_session()
    try:
        user = request.user
        data = request.get_json()
        prompt = data.get('prompt')
        sqlquery = data.get('sqlquery')
        feedmode = data.get('feedmode')
        timestamp = datetime.now()

        if not prompt or not sqlquery:
            return jsonify({
                "status": "error",
                "message": "Prompt and SQL query are required."
            }), 400

        # Check for duplicate prompt and handle based on weightage
        query_check = """
            SELECT username, weightage FROM bank.examples
            WHERE input = :prompt
        """
        params_check = {"prompt": prompt}
        result = session.connection().execute(text(query_check), params_check).fetchone()

        if result:
            existing_username, existing_weightage = result
            if existing_weightage == 100:
                return jsonify({
                    "status": "success",
                    "message": "Duplicate entry, this input already exists"
                }), 200
            elif existing_weightage == 50 and existing_username == user:
                return jsonify({
                    "status": "success",
                    "message": "Duplicate entry, this input already exists"
                }), 200

        query_role = """
            SELECT role FROM dbo.users
            WHERE username = :username
        """
        params_role = {"username": user}
        role_result = session.connection().execute(text(query_role), params_role).fetchone()
        role = role_result[0] if role_result else None

        # Insert into global DB
        query_insert = """
            INSERT INTO bank.examples (input, query, mode, weightage, username, timestamp, role)
            VALUES (:input, :query, :mode, :weightage, :username, :timestamp, :role)
        """
        params_insert = {
            "input": prompt,
            "query": sqlquery,
            "mode": feedmode,
            "weightage": 50,
            "username": user,
            "timestamp": timestamp,
            "role": role
        }
        session.connection().execute(text(query_insert), params_insert)
        session.commit()

        add_new_example_for_training(prompt, sqlquery, user, feedmode, weightage=50)

        return jsonify({
            "status": "success",
            "message": "Your feedback has been submitted successfully."
        }), 200

    except Exception as e:
        print("Exception:", e)
        session.rollback()
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500
    finally:
        session.close()
    
FEATURE_FLAG = "on"

@app.before_request
def check_coming_soon():
    global FEATURE_FLAG
    if FEATURE_FLAG == 'off':
        if request.path not in ('/', '/toggle_feature_flag','/statuschange'):
            return render_template('coming_soon.html'), 503

@app.route("/logout", methods=["POST"])
@token_required
def logout():
    user = request.user
    try:
        if user in user_memories:
            del user_memories[user]
        return jsonify({"status": "success", "message": "Logged out successfully"}), 200
    except Exception as e:
        print("Error occurred during logout:", e)
        return jsonify({"status": "error", "message": f"Error logging out: {str(e)}"}), 500
     
@app.route("/login", methods=["POST"])
def login():
    session = get_session()
    try:
        username = request.json.get("username")
        password = request.json.get("password")

        # Fetch user data
        query = """
            SELECT [password], [set_flag]
            FROM [Car Store DB].[dbo].[users]
            WHERE [username] = :username
        """
        params = {"username": username}
        result = session.connection().execute(text(query), params).fetchone()

        if result:
            stored_password, set_flag = result
            if check_password_hash(stored_password, password):
                if set_flag == 0:
                    return jsonify({
                        "status": "error",
                        "message": "Your account is not activated. Please contact your administrator."
                    }), 403
                else:
                    token = jwt.encode({
                        'sub': username,
                        'iat': datetime.utcnow(),
                        'exp': datetime.utcnow() + timedelta(hours=10)
                    }, SECRET_KEY, algorithm="HS256")

                    response = make_response(jsonify({
                        "status": "success",
                        "message": "Login Successful",
                        "token": token
                    }), 200)
                    return response
            else:
                return jsonify({
                    "status": "error",
                    "message": "Invalid username or password"
                }), 403
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid username or password"
            }), 403

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "details": str(e)
        }), 500
    finally:
        session.close()    

@app.route("/signup", methods=["POST"])
def handle_signup():
    session = get_session()
    try:
        data = request.get_json()
        new_username = data.get("username")
        new_password = data.get("password")
        new_email = data.get("email")
        new_company = data.get("company-name")

        if not all([new_username, new_password, new_email, new_company]):
            return jsonify({"error": "All fields are required."}), 400

        hashed_password = generate_password_hash(new_password)
        new_userid = str(uuid.uuid4())

        # Check for duplicate email
        query_email = """
            SELECT COUNT(*)
            FROM [Car Store DB].[dbo].[users]
            WHERE [email] = :email
        """
        params_email = {"email": new_email}
        email_count = session.connection().execute(text(query_email), params_email).fetchone()[0]

        if email_count > 0:
            return jsonify({"error": "This email is already registered. Please use a different email."}), 409

        # Insert new user
        query_user = """
            INSERT INTO [Car Store DB].[dbo].[users] 
            ([userid], [username], [password], [email], [company_name], [set_flag])
            VALUES (:userid, :username, :password, :email, :company_name, :set_flag)
        """
        params_user = {
            "userid": new_userid,
            "username": new_username,
            "password": hashed_password,
            "email": new_email,
            "company_name": new_company,
            "set_flag": 0
        }
        session.connection().execute(text(query_user), params_user)

        # Insert token management record
        query_token = """
            INSERT INTO token_management (username, email, token_used, token_day, token_limit, token_cost)
            VALUES (:username, :email, :token_used, :token_day, :token_limit, :token_cost)
        """
        params_token = {
            "username": new_username,
            "email": new_email,
            "token_used": 0,
            "token_day": datetime.now(),
            "token_limit": 500000,
            "token_cost": 0
        }
        session.connection().execute(text(query_token), params_token)

        session.commit()

        return jsonify({"message": "Signup successful! Please login to continue."}), 200

    except Exception as e:
        print(f"Error during signup: {e}")
        session.rollback()
        return jsonify({"error": "Internal server error.", "details": str(e)}), 500
    finally:
        session.close()

@app.route("/teams_login", methods=["POST"])
def teams_login():
    session = get_session()
    try:
        username = request.json.get("username")

        # Fetch user data
        query = """
            SELECT [set_flag]
            FROM [Car Store DB].[dbo].[users]
            WHERE [username] = :username
        """
        params = {"username": username}
        result = session.connection().execute(text(query), params).fetchone()

        if result:
            set_flag = result[0]
            if set_flag == 0:
                return jsonify({
                    "status": "error",
                    "message": "Your account is not activated. Please contact your administrator."
                }), 403
            else:
                token = jwt.encode({
                    'sub': username,
                    'iat': datetime.utcnow(),
                    'exp': datetime.utcnow() + timedelta(hours=10)
                }, SECRET_KEY, algorithm="HS256")

                response = make_response(jsonify({
                    "status": "success",
                    "message": "Login Successful",
                    "token": token
                }), 200)
                return response
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid username"
            }), 403

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "details": str(e)
        }), 500
    finally:
        session.close()
        
@app.route("/health")
def health():
    return "ok", 200

@app.teardown_appcontext
def remove_session(exception=None):
    try:
        dev_session = get_session()
        if dev_session is not None:
            dev_session.close()
        
    except Exception as e:
        print(f"Error closing sessions: {str(e)}")
    
if __name__ == "__main__":
    try:
        initialize_vector_store()
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Database dialect is working", conn.dialect.name)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cert_path = os.path.join(base_dir, 'certificates', 'cert.txt')
            key_path = os.path.join(base_dir, 'certificates', 'key.txt')

            ssl_context = (cert_path, key_path)

        app.run(debug=False, host='0.0.0.0', port=5000, ssl_context=ssl_context)
    except Exception as e:
        print(f"Error initializing the application: {e}")
        sys.exit(1)
