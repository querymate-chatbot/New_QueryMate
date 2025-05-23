from collections import defaultdict
from functools import wraps
import sys
import threading
import time
import uuid
import httpx
from flask import Flask, flash, make_response, redirect, render_template, request, jsonify, send_from_directory, stream_with_context, url_for
from flask_session import Session
from dotenv import load_dotenv, find_dotenv
import os
from functools import lru_cache
import msal
import pandas as pd
import secrets
from flask import Flask, render_template, request, jsonify, session, Response
import warnings
import requests
from db_config import get_engine, get_session
from flask_cors import CORS
import uuid
import json
import hmac
import hashlib
import secrets
from flask_limiter.util import get_remote_address
from sqlalchemy import text
from datetime import datetime, timedelta
import asyncio
import hashlib
import base64

from admin import admin
app = Flask(__name__)
CORS(app) 

app.secret_key = secrets.token_hex(24)
app.register_blueprint(admin)

app.config["SESSION_TYPE"] = "filesystem" 
app.config["SESSION_PERMANENT"] = False
Session(app)
warnings.filterwarnings("ignore")

app.config.update(
    SESSION_COOKIE_SECURE=True, 
    SESSION_COOKIE_HTTPONLY=False,  
    SESSION_COOKIE_SAMESITE='None',
)


_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

FEATURE_FLAG = "on"


#teams credentials
CLIENT_ID = '67a8027e-b04c-4563-9947-178e69585bd9'
CLIENT_SECRET = 'OeM8Q~qa_NevOyovEHVVeob2gGG~vs96Y4Pcjb42'
AUTHORITY = 'https://login.microsoftonline.com/c30dffe5-1038-430d-99ab-e4e39218b243'
REDIRECT_PATH = '/querymate/banking'
SCOPE = ['User.Read']
API_ENDPOINT = 'https://graph.microsoft.com/v1.0/me'


#temp login
base_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(base_dir, 'links.xlsx')
EXCEL_FILE = excel_file_path

# Initialize Excel file
def init_excel():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=['token', 'link', 'expiry_seconds', 'ip_address', 'is_active'])
        df.to_excel(EXCEL_FILE, index=False)

# Load links from Excel
def load_links():
    init_excel()
    return pd.read_excel(EXCEL_FILE)

# Save links to Excel
def save_links(df):
    df.to_excel(EXCEL_FILE, index=False)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            resp = make_response(redirect(url_for('login')))
            resp.set_cookie('token', '', max_age=0, secure=True, httponly=False, samesite='None')
            flash('Please log in to access this page', 'error')
            return resp

        login_time = session.get('login_time')
        is_auto_login = session.get('auto_login', False)
        expiry_seconds = int(session.get('expiry_seconds', 10))

        if is_auto_login and login_time:
            try:
                login_time_dt = datetime.fromisoformat(login_time)
                if datetime.utcnow() > login_time_dt + timedelta(seconds=expiry_seconds):
                    session.clear()
                    resp = make_response(redirect(url_for('login')))
                    resp.set_cookie('token', '', max_age=0, secure=True, httponly=False, samesite='None')
                    flash('Session has expired', 'error')
                    return resp
            except ValueError:
                session.clear()
                resp = make_response(redirect(url_for('login')))
                resp.set_cookie('token', '', max_age=0, secure=True, httponly=False, samesite='None')
                flash('Invalid session data', 'error')
                return resp

        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        elif session['username'].lower() != 'admin':
            flash('You are not authorized to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/generate_link', methods=['POST'])
@admin_required
def generate_link():
    data = request.get_json()
    expiry_seconds = int(data.get('expiry_seconds', 10)) if data else 10
    if expiry_seconds <= 0:
        expiry_seconds = 10

    token = str(uuid.uuid4())
    link = url_for('access_link', token=token, _external=True)

    df = load_links()
    new_entry = {
        'token': token,
        'link': link,
        'expiry_seconds': expiry_seconds,
        'ip_address': '',
        'is_active': True
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    save_links(df)

    return jsonify({"link": link, "token": token, "expiry_seconds": expiry_seconds})

@app.route('/delete_link/<token>', methods=['POST'])
@admin_required
def delete_link(token):
    df = load_links()
    if token in df['token'].values:
        df = df[df['token'] != token]
        save_links(df)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Link not found"}), 404

@app.route('/toggle_link_status/<token>', methods=['POST'])
@admin_required
def toggle_link_status(token):
    df = load_links()
    if token in df['token'].values:
        current_status = df.loc[df['token'] == token, 'is_active'].iloc[0]
        df.loc[df['token'] == token, 'is_active'] = not current_status
        save_links(df)
        return jsonify({"success": True, "is_active": not current_status})
    return jsonify({"success": False, "error": "Link not found"}), 404

@app.route('/list_links', methods=['GET'])
@admin_required
def list_links():
    df = load_links()
    df['ip_address'] = df['ip_address'].fillna('').astype(str)
    links = []
    for _, row in df.iterrows():
        links.append({
            'token': row['token'],
            'link': row['link'],
            'expiry_seconds': int(row['expiry_seconds']),
            'ip_address': row['ip_address'],
            'is_active': bool(row['is_active'])
        })
    return jsonify(links)

@app.route('/demo/<token>', methods=['GET'])
async def access_link(token):
    if token == 'QuerymateDemo.mp4':
        demo_folder = os.path.join(app.root_path, 'demo')
        requested_file = os.path.join(demo_folder, token)

        if os.path.isfile(requested_file):
            return send_from_directory(demo_folder, token)

    df = load_links()
    print("links", df)
    token_data = df[df['token'] == token]

    print("Token data", token_data)

    if token_data.empty or not token_data.iloc[0]['is_active']:
        session.clear()
        flash('Invalid or inactive link', 'error')
        return redirect(url_for('login'))

    token_data = token_data.iloc[0]
    expiry_seconds = token_data['expiry_seconds']

    # Store IP address
    ip_address = request.remote_addr
    df.loc[df['token'] == token, 'ip_address'] = ip_address
    save_links(df)

    username = 'Nice'
    password = 'Demo@123'

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(
                "https://localhost:5000/login",
                json={'username': username, 'password': password},
                timeout=5.0
            )

            if response.status_code == 200:
                token = response.json().get('token')
                if not token:
                    session.clear()
                    flash('No token received from backend', 'error')
                    return redirect(url_for('login'))

                session_id = str(uuid.uuid4())
                session['session_user_id'] = session_id
                session['username'] = username
                session['auto_login'] = True
                session['login_time'] = datetime.utcnow().isoformat()
                session['expiry_seconds'] = expiry_seconds

                resp = make_response(redirect(url_for("admin" if username.lower() == "admin" else "banking")))
                resp.set_cookie('token', token, max_age=expiry_seconds, secure=True, httponly=False, samesite='None')
                return resp
            else:
                session.clear()
                flash(response.json().get('message', 'Authentication failed'), 'error')
                return redirect(url_for('login'))
        except httpx.RequestError as e:
            session.clear()
            flash(f'Unable to reach the backend server: {str(e)}', 'error')
            return redirect(url_for('login'))

FEATURE_FLAG = 'on'

ip_counters = defaultdict(lambda: {'count': 0, 'reset_time': datetime.utcnow() + timedelta(minutes=5)})

MAX_REQUESTS_PER_IP = 200
TIME_WINDOW = timedelta(minutes=5)
ALLOWED_IP = '127.0.0.1' 

@app.before_request
def handle_limits_and_flags():
    global FEATURE_FLAG

    if FEATURE_FLAG == 'off':
        if ip != ALLOWED_IP and 'static' not in request.path and 'admin' not in request.path:
            return render_template("coming_soon.html"), 503

    ip = request.remote_addr
    now = datetime.utcnow()

    ip_data = ip_counters[ip]
    if now > ip_data['reset_time']:
        ip_counters[ip] = {'count': 1, 'reset_time': now + TIME_WINDOW}
    else:
        ip_data['count'] += 1

    if ip_data['count'] > MAX_REQUESTS_PER_IP:
        return render_template("too_many_request.html"), 503

@app.route("/admin/feature_flag_status", methods=["GET"])
@admin_required
def feature_flag_status():
    global FEATURE_FLAG
    return jsonify({"feature_flag": FEATURE_FLAG})

@app.route("/admin/toggle_feature_flag", methods=["POST"])
@admin_required
def toggle_feature_flag():
    global FEATURE_FLAG
    FEATURE_FLAG = "on" if FEATURE_FLAG == "off" else "off"  
    return jsonify({"status": "success", "feature_flag": FEATURE_FLAG})

@app.route('/')
def index():
    return render_template('index.html')  
    
@app.route('/querymate')
def querymate():
    return render_template('querymate2.html')
    

def getTablenames(username):
    session = get_session()
    try:
        query = text("""
            SELECT access
            FROM roles JOIN users ON roles.role = users.role
            WHERE username = :username
        """)
        result = session.execute(query, {"username": username}).fetchall()
        if not result:
            return []
        table_names = result[0][0].split(', ')
        return table_names
    except Exception:
        return []
        
@lru_cache(maxsize=128)
def fetch_examples():
    db_session = get_session()
    try:
        username = session.get('username')
        if not username:
            return []

        query = text("SELECT input, query, mode FROM bank.examples WHERE weightage = 100 OR username = :username")
        result = db_session.execute(query, {"username": username})
        df = pd.DataFrame(result.fetchall(), columns=[col.lower() for col in result.keys()])
        return df[["input", "query", "mode"]].to_dict(orient="records")
    except Exception:
        return []

def clear_memory_in_background(username, authorization_token):
    try:
        data = {
            'session_user_id': username
        }
        headers = {
            'Authorization': authorization_token
        }
        response = requests.post("https://localhost:5000/clear_memory", json=data, headers=headers, verify=False)
        if response.status_code == 200:
            print(f"Memory cleared successfully for {username}")
        else:
            print(f"Failed to clear memory for {username}:", response.json())
    except Exception as e:
        print(f"Error occurred while trying to clear memory for {username}:", e)

        
@app.route("/querymate/clear_history", methods=["POST"])
@login_required
def clear_history():
    print("Clearing history", request.headers.get("Authorization"))
    try:
        authorization_token = request.headers.get("Authorization")
        try:
            threading.Thread(target=clear_memory_in_background, args=(session.get('session_user_id'), authorization_token)).start()
            return jsonify({"status": "success", "message": "History cleared successfully"}), 200
        except Exception as e:
            print(f"Error occurred while trying to clear history: {e}")
            return jsonify({"status": "error", "message": "Failed to clear history"}), 500
    except Exception as e:
        print(f"Error occurred while trying to clear history: {e}")


@app.route("/querymate/banking")
@login_required
def banking():
    threading.Thread(target=clear_memory_in_background, args=(session.get('session_user_id'), request.cookies.get('token'))).start()
    return render_template("carDB.html")

@app.route("/querymate/fetch_examples")
@login_required
def fetch_examples_route(): 
    try:
        examples = fetch_examples()
        response = {
            "status": "success",
            "data": examples,
            "message": "Examples fetched successfully"
        }
        return make_response(jsonify(response), 200)
    except Exception as e:
        response = {
            "status": "error",
            "data": None,
            "message": f"Failed to fetch examples: {str(e)}"
        }
        return make_response(jsonify(response), 500)

@app.route("/querymate/fetch_sidebarcontent")
@login_required
def fetch_sidebar_content(): 
    try:
        table_columns = SidebarContent()
        response = {
            "status": "success",
            "data": table_columns,
            "message": "Sidebar content fetched successfully"
        }
        return make_response(jsonify(response), 200)
    except Exception as e:
        response = {
            "status": "error",
            "data": None,
            "message": f"Failed to fetch sidebar content: {str(e)}"
        }
        return make_response(jsonify(response), 500)

@lru_cache(maxsize=128)
def SidebarContent():
    table_names = getTablenames(session.get('username'))
    if not table_names:
        return {}

    db_session = get_session()
    try:
        table_columns = {}
        for table in table_names:
            query = text("""
                SELECT 
                    c.column_name,
                    CASE
                        WHEN tc.constraint_type = 'PRIMARY KEY' THEN 'PK'
                        WHEN tc.constraint_type = 'FOREIGN KEY' THEN 'FK'
                        ELSE 'normal'
                    END AS column_type
                FROM information_schema.columns c
                LEFT JOIN information_schema.key_column_usage kcu 
                    ON c.column_name = kcu.column_name
                    AND c.table_name = kcu.table_name
                LEFT JOIN information_schema.table_constraints tc
                    ON kcu.constraint_name = tc.constraint_name
                    AND tc.table_name = c.table_name
                WHERE c.table_schema = 'bank' 
                    AND c.table_name = :table_name
            """)
            result = db_session.execute(query, {"table_name": table}).fetchall()
            column_details = {row[0]: row[1] for row in result}
            table_columns[table] = column_details

        return table_columns
    except Exception:
        return {}

@app.route('/all_tables_column', methods=['GET'])
def all_tables_column():
    table_names = getTablenames('admin')
    if not table_names:
        return jsonify([])

    db_session = get_session()
    try:
        tables_list = []

        for table in table_names:
            query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'bank'
                  AND table_name = :table_name
                ORDER BY ordinal_position
            """)
            result = db_session.execute(query, {"table_name": table}).fetchall()
            columns = [row[0] for row in result]

            tables_list.append({
                "name": table,
                "columns": columns
            })

        return jsonify(tables_list)

    except Exception as e:
        print(f"Error fetching table columns: {e}")
        return jsonify([])
    

DATA_MODEL_DIR = os.path.join(base_dir, 'saved_data_models')
DATA_MODEL_FILE = os.path.join(base_dir, 'saved_data_models', 'data_model.json')

if not os.path.exists(DATA_MODEL_DIR):
    os.makedirs(DATA_MODEL_DIR)

@app.route('/save_data_model', methods=['POST'])
def save_data_model():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        filepath = os.path.join(DATA_MODEL_DIR, DATA_MODEL_FILE)

        # Save JSON to file (overwrite mode)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return jsonify({"status": "success", "message": f"Data model saved successfully"}), 200

    except Exception as e:
        print(f"Error saving data model: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/querymate/getAutoSchema', methods=['GET'])
def get_auto_schema():
    try:
        filepath = os.path.join(DATA_MODEL_DIR, DATA_MODEL_FILE)

        if not os.path.exists(filepath):
            return jsonify({"status": "error", "message": "No data model found"}), 404

        # Read the saved JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return jsonify({"status": "success", "data_model": data}), 200

    except Exception as e:
        print(f"Error reading data model: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/querymate/admin', methods=['GET', 'POST'])
@admin_required
def admin():
    return render_template('admin.html')

@app.route('/login_request', methods=['POST'])
async def login_request():
    username = request.form.get('username')
    password = request.form.get('password')
 
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(
                "https://localhost:5000/login",
                json={'username': username, 'password': password},
                timeout=5.0
            )
 
            if response.status_code == 200:
                token = response.json().get('token')
                if not token:
                    flash('No token received from backend', 'error')
                    return redirect(url_for('login'))
                
                # Generate and store UUID for the session
                session_id = str(uuid.uuid4())

                session['session_user_id'] = session_id
                session['username'] = username
                session['auto_login'] = False
                session['login_time'] = datetime.utcnow().isoformat()
 
                if username.lower() == "admin":
                    resp = make_response(redirect(url_for("admin")))
                    resp.set_cookie('token', token, max_age=60*60*24*7, secure=True, httponly=False, samesite='None')
                    return resp
                else:
                    resp = make_response(redirect(url_for("banking")))
                    resp.set_cookie('token', token, max_age=60*60*24*7, secure=True, httponly=False, samesite='None')
                    return resp
            else:
                flash(response.json().get('message', 'Authentication failed'), 'error')
                return redirect(url_for('login'))
 
        except httpx.RequestError as e:
            flash(f'Unable to reach the backend server: {str(e)}', 'error')
            return redirect(url_for('login'))
        
@app.route("/signup", methods=["GET", "POST"])
async def signup():
    if request.method == "POST":
        new_username = request.form.get("username")
        new_password = request.form.get("password")
        new_email = request.form.get("email")
        new_company_name = request.form.get("company-name")

        print("New user data:", new_username, new_password, new_email, new_company_name)

        try:
            async with httpx.AsyncClient(verify=False, timeout=100) as client:
                response = await client.post(
                    'https://localhost:5000/signup',
                    json={
                        'username': new_username,
                        'password': new_password,
                        'email': new_email,
                        'company-name': new_company_name
                    }
                )
                res_json = response.json()
                if response.status_code == 200:
                    flash(res_json.get("message", "Signup successful!"), "success")
                else:
                    flash(res_json.get("error", "Signup failed."), "error")
        except Exception as e:
            print("Error occurred during frontend signup:", e)
            flash("Failed to connect to signup service.", "error")

        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/querymate/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")

@app.route("/querymate/expire_session", methods=["POST"])
def expire_session():
    flash("Session Expired. Please Login Again.", "error")
    return '', 204 
       
@app.route('/login_request_teams', methods=['POST'])
async def login_request_teams():
    username = request.form.get('username').split("@")[0]
 
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post("https://localhost:5000/teams_login", json={'username': username})
 
            if response.status_code == 200:
                    
                    session_id = str(uuid.uuid4())

                    session['session_user_id'] = session_id
                    session['username'] = username
                    session['auto_login'] = False
                    session['login_time'] = datetime.utcnow().isoformat()

                    token = response.json().get('token')
 
                    if username == "admin":
                        resp = make_response(redirect(url_for("admin")))
                        resp.set_cookie('token', token, max_age=60*60*24*7, secure=True, httponly=False, samesite='None')
                        return resp
 
                    else:
                        resp = make_response(redirect(url_for("banking")))
                        resp.set_cookie('token', token, max_age=60*60*24*7, secure=True, httponly=False, samesite='None')
                        return resp
            else:
                flash(response.json().get('message'), 'error')
                return redirect(url_for('login'))
 
        except httpx.RequestError as e:
            return jsonify({'status': 'error', 'message': f'Unable to reach the backend server: {str(e)}'}), 500

def _build_msal_app():
    return msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET
    )

@app.route('/login_request_microsoft')
def login_request_microsoft():
    msal_app = _build_msal_app()
    auth_url = msal_app.get_authorization_request_url(SCOPE, redirect_uri=url_for('authorized', _external=True))
    return redirect(auth_url)

@app.route("/querymate/authorized")
async def authorized():
    msal_app = _build_msal_app()
    code = request.args.get('code')
    
    if not code:
        return jsonify({'status': 'error', 'message': 'Authorization code missing'}), 400
    
    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=SCOPE,
        redirect_uri=url_for('authorized', _external=True)
    )
    
    if 'access_token' not in result:
        error = result.get('error', 'Unknown error')
        error_description = result.get('error_description', 'No description provided')
        return jsonify({
            'status': 'error',
            'message': 'Failed to acquire access token',
            'error': error,
            'error_description': error_description
        }), 401
    
    access_token = result['access_token']
    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        user_info = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers).json()
    except requests.RequestException as e:
        return jsonify({'status': 'error', 'message': f'Failed to retrieve user info: {str(e)}'}), 500
    
    username_from_email = user_info.get('userPrincipalName', 'Not Available')
    
    if '#EXT#' in username_from_email:
        username_from_email = username_from_email.split('#EXT#')[0]
    
    username = username_from_email.split('_')[0]

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post("https://localhost:5000/teams_login", json={'username': username})
    
            if response.status_code == 200:
                token = response.json().get('token')
                session['username'] = username
    
                resp = make_response(redirect(url_for("banking")))
                resp.set_cookie('token', token, max_age=60*60*24*7, secure=True, httponly=False, samesite='None')
                return resp
            else:
                flash(response.json().get('message'), 'error')
                return redirect(url_for('login'))
    
        except httpx.RequestError as e:
            return jsonify({'status': 'error', 'message': f'Unable to reach the backend server: {str(e)}'}), 500
    
    return jsonify({'status': 'error', 'message': 'Unexpected error occurred'}), 500


@app.route('/querymate/logout', methods=['GET', 'POST']) 
@login_required
def logout():
    try:
        headers = {
            "Authorization": request.headers.get("Authorization")
        }
        
        with httpx.Client(verify=False) as client:
            response = client.post("https://localhost:5000/logout", headers=headers)
            response_json = response.json()
            
            if response_json.get("status") == "success":
                session.clear()
                return jsonify({
                    'status': 'success',
                    'message': 'Successfully logged out'
                }), 200
            else:
                return jsonify({
                    'status': 'error',
                    'message': response_json.get('message', 'Logout failed')
                }), response.status_code

    except httpx.RequestError as e:
        return jsonify({
            'status': 'error',
            'message': f'Unable to reach the backend server: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error during logout: {str(e)}'
        }), 500
from threading import Lock
 
processing_status = {}
clients = {}
clients_lock = Lock()
 
@app.route("/update_status", methods=["POST"])
def update_status():
    data = request.json
    session_id_stream = data.get("session_id_stream")  
    status = data.get("status")
    content = data.get("content")
   
    content_str = content if isinstance(content, str) else json.dumps(content)
 
    if status == "success":
        processing_status[session_id_stream] = content_str
 
    with clients_lock:
        listeners = clients.get(session_id_stream, [])
        for listener in listeners:
            listener.append(content_str)
 
    return "OK"
 
 
@app.route("/process_status")
def process_status():
    session_id_stream = request.args.get("session_id_stream")
    if not session_id_stream:
        return "No session id", 400
 
    def event_stream():
        messages = []
        timeout = time.time() + 60
 
        with clients_lock:
            if session_id_stream not in clients:
                clients[session_id_stream] = []
            clients[session_id_stream].append(messages)
 
        try:
            last_msg_content = None
            while time.time() < timeout:
                while messages:
                    msg = messages.pop(0)
                    try:
                        msg_data = json.loads(msg)
                    except json.JSONDecodeError:
                        msg_data = {"content": msg}
 
                    yield f"data: {json.dumps({'content': msg_data})}\n\n"
                    last_msg_content = msg_data.get("content", "")
 
                    if isinstance(last_msg_content, str) and (
                        last_msg_content == "Processing Completed" or last_msg_content.startswith("Error")
                    ):
                        return
 
        finally:
            with clients_lock:
                try:
                    clients[session_id_stream].remove(messages)
                    if not clients[session_id_stream]:
                        del clients[session_id_stream]
 
                except ValueError:
                    print(f"[process_status] Listener already removed for {session_id_stream}")
 
    return Response(stream_with_context(event_stream()), content_type='text/event-stream')
 
@app.route('/process_frontend', methods=['POST'])
def process_frontend():
    return asyncio.run(handle_process())
 
async def handle_process():
    try:
        form_data = request.form
        files = request.files
        headers = {
            "Authorization": request.headers.get("Authorization")
        }
 
        files_to_send = []
        form_data_with_session = dict(form_data)
 
        session_id = session.get('session_user_id')
        if session_id:
            form_data_with_session['session_user_id'] = session_id
        else:
            return jsonify({'status': 'error', 'message': 'Session ID not found'}), 401
 
        if files:
            for key in files:
                file_values = files.getlist(key)
                for file in file_values:
                    files_to_send.append((key, (file.filename, file.stream, file.mimetype)))
 
        try:
            async with httpx.AsyncClient(verify=False, timeout=200) as client:
                response = await client.post(
                    'https://localhost:5000/process',  
                    headers=headers,
                    data=form_data_with_session,
                    files=files_to_send
                )
                response.encoding = 'utf-8'
                return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    except httpx.RequestError as e:
        return jsonify({'status': 'error', 'message': f'Unable to reach backend: {str(e)}'}), 500
 
# import json

# last_payload_hash = None
# last_response_data = None

# def generate_payload_hash(form_data, files):
#     """
#     Create a hash based on form data and file metadata (not full content).
#     """
#     form_dict = dict(form_data)
#     file_summary = {key: [file.filename for file in files.getlist(key)] for key in files}
    
#     payload = {
#         "form": form_dict,
#         "files": file_summary
#     }
#     payload_str = json.dumps(payload, sort_keys=True)
#     return hashlib.md5(payload_str.encode('utf-8')).hexdigest()

# @app.route('/process_frontend', methods=['POST'])
# async def handle_process():
#     global last_payload_hash, last_response_data

#     try:
#         form_data = request.form  # Contains 'prompt', 'selected_option'
#         files = request.files    # Contains 'file' (if uploaded)
#         token = request.headers.get("Authorization")
#         session_id = session.get('session_user_id')  # Retrieve session_id from login session
#         headers = {"Authorization": token}

#         # Generate hash for current payload
#         current_hash = generate_payload_hash(form_data, files)

#         if current_hash == last_payload_hash:
#             print("Duplicate request. Returning cached response.")
#             return jsonify(last_response_data), 200

#         # Prepare files for backend request
#         files_to_send = []
#         for key in files:
#             file_values = files.getlist(key)
#             for file in file_values:
#                 files_to_send.append((key, (file.filename, file.stream, file.mimetype)))

#         try:
#             print("Sending new request to backend")
#             async with httpx.AsyncClient(verify=False, timeout=200) as client:
#                 # Include session_id in the form data
#                 form_data_with_session = dict(form_data)
#                 if session_id:
#                     form_data_with_session['session_user_id'] = session_id
#                 else:
#                     print("Warning: No session_id found in session")
#                     return jsonify({'status': 'error', 'message': 'Session ID not found'}), 401

#                 response = await client.post(
#                     'https://localhost:5000/process',
#                     headers=headers,
#                     data=form_data_with_session,
#                     files=files_to_send
#                 )
#                 response.encoding = 'utf-8'
#                 response_json = response.json()

#                 # Cache the result
#                 last_payload_hash = current_hash
#                 last_response_data = response_json

#                 return jsonify(response_json), response.status_code
#         except Exception as e:
#             print("Error occurred at process frontend:", e)
#             return jsonify({'status': 'error', 'message': 'Internal processing error'}), 500

#     except httpx.RequestError as e:
#         return jsonify({'status': 'error', 'message': f'Unable to reach the backend server: {str(e)}'}), 500
    
@app.route('/process_frontend_audio', methods=['POST'])
async def process_audio():
    try: 
        form_data = request.form
        files = request.files
        headers = {
            "Authorization": request.headers.get("Authorization")
        }
        if files:
            files_to_send = {}
            for key in files:
                files_to_send[key] = (files[key].filename, files[key].stream, files[key].mimetype)

        try:
            async with httpx.AsyncClient(verify=False, timeout=100) as client:
                response = await client.post(
                    'https://localhost:5000/process_audio',  
                    headers=headers, 
                    data=form_data, 
                    files=files_to_send 
                )
                if response.status_code == 200:
                    return jsonify(response.json()), 200
                else:
                    return jsonify(response.json()), response.status_code
        except Exception as e:
            print("Error occured at process frontedn", e)
    except httpx.RequestError as e:
        return jsonify({'status': 'error', 'message': f'Unable to reach the backend server: {str(e)}'}), 500

@app.route('/frontend_submit_feedback', methods=['POST'])
async def process_feedback():
    try: 
        data = request.get_json()
        prompt = data.get('prompt')
        sqlquery = data.get('sqlquery')
        feedmode = data.get('feedmode')
        headers = {
            "Authorization": request.headers.get("Authorization")
        }

        try:
            async with httpx.AsyncClient(verify=False, timeout=100) as client:
                response = await client.post(
                    'https://localhost:5000/submit_feedback',  
                    headers=headers, 
                    json={
                        'prompt': prompt,
                        'sqlquery': sqlquery,
                        'feedmode': feedmode
                    }
                )
                if response.status_code == 200:
                    return jsonify(response.json()), 200
                else:
                    return jsonify(response.json()), response.status_code
        except Exception as e:
            print("Error occurred at process frontend:", e)
            return jsonify({'status': 'error', 'message': 'Failed to process feedback'}), 500
    except httpx.RequestError as e:
        return jsonify({'status': 'error', 'message': f'Unable to reach the backend server: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Unexpected error: {str(e)}'}), 500
    
@app.route('/get_username', methods=['GET'])
@login_required
def get_username_route():
    try:
        username = session.get('username')
        if not username:
            return jsonify({'error': 'Username not found'}), 404

        key_material = hashlib.sha256(f"{username}-super-salt-value".encode()).digest()
        key_b64 = base64.b64encode(key_material).decode()

        print("key", key_b64)

        return jsonify({
            'username': username,
            'key_b64': key_b64
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/agents')
def agents():
    return render_template('agents.html')  

@app.route('/demo/<path:filename>')
def demo_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'demo'), filename)  

    
# @app.route('/get_username', methods=['GET'])
# @login_required
# def get_username_route():
#     try:
#         username = session.get('username')
#         if not username:
#             return jsonify({'error': 'Username not found'}), 404

#         return jsonify({'username': username}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    

@app.route('/new_template', methods=['POST'])
@login_required
def new_template():
    data = request.get_json()
    name = data.get('name')
    messages = data.get('messages', [])
    username = session.get('username')

    if not name or not messages or not username:
        return jsonify({'error': 'Missing name, messages, or username'}), 400

    template_id = str(uuid.uuid4())

    db_session = get_session()
    try:
        # Check for duplicate name for the same username
        query = text("SELECT id FROM templates WHERE name = :name AND username = :username")
        if db_session.execute(query, {"name": name, "username": username}).fetchone():
            return jsonify({'error': 'Template name must be unique for this user'}), 400

        # Insert into templates
        db_session.execute(
            text("INSERT INTO templates (id, name, username) VALUES (:id, :name, :username)"),
            {"id": template_id, "name": name, "username": username}
        )

        # Insert messages
        for msg in messages:
            message_id = str(uuid.uuid4())
            db_session.execute(
                text("INSERT INTO template_messages (id, template_id, content) VALUES (:id, :template_id, :content)"),
                {"id": message_id, "template_id": template_id, "content": msg}
            )

        db_session.commit()
        return jsonify({'message': 'Template saved successfully'}), 200

    except Exception:
        db_session.rollback()
        return jsonify({'error': 'Something went wrong'}), 500

@app.route('/get_templates', methods=['GET'])
@login_required
def get_templates():
    username = session.get('username')
    if not username:
        return jsonify([])

    db_session = get_session()
    try:
        query = text("""
            SELECT t.id, t.name, m.content
            FROM templates t
            LEFT JOIN template_messages m ON t.id = m.template_id
            WHERE t.username = :username
            ORDER BY t.created_at DESC
        """)
        rows = db_session.execute(query, {"username": username}).fetchall()

        templates_dict = {}
        for row in rows:
            if not row or len(row) < 3:
                continue
            template_id, name, content = row
            if template_id not in templates_dict:
                templates_dict[template_id] = {'name': name, 'messages': []}
            if content:  # Handle NULL content from LEFT JOIN
                templates_dict[template_id]['messages'].append(content)

        templates = list(templates_dict.values())
        return jsonify(templates)

    except Exception:
        return jsonify({'error': 'Internal error'}), 500

@app.route('/delete_all_templates', methods=['POST'])
@login_required
def delete_all_templates():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    db_session = get_session()
    try:
        # Delete all template_messages for the user's templates
        db_session.execute(
            text("""
                DELETE FROM template_messages
                WHERE template_id IN (
                    SELECT id FROM templates WHERE username = :username
                )
            """),
            {"username": username}
        )

        # Delete all templates for the user
        db_session.execute(
            text("DELETE FROM templates WHERE username = :username"),
            {"username": username}
        )

        db_session.commit()
        return jsonify({'message': 'All templates deleted successfully'}), 200

    except Exception:
        db_session.rollback()
        return jsonify({'error': 'Something went wrong'}), 500

@app.route('/delete_template', methods=['POST'])
@login_required
def delete_template_by_name():
    data = request.get_json()
    template_name = data.get('template_name')
    username = data.get('username')

    if not template_name:
        return jsonify({'error': 'Template name is required'}), 400

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    db_session = get_session()
    try:
        # Check if the template exists and belongs to this user
        query = text("SELECT id FROM templates WHERE name = :name AND username = :username")
        result = db_session.execute(query, {"name": template_name, "username": username}).fetchone()

        if result is None:
            return jsonify({'error': 'Template not found or unauthorized'}), 403

        template_id = result[0]

        # Delete associated messages
        db_session.execute(
            text("DELETE FROM template_messages WHERE template_id = :template_id"),
            {"template_id": template_id}
        )

        # Delete the template
        db_session.execute(
            text("DELETE FROM templates WHERE id = :template_id"),
            {"template_id": template_id}
        )

        db_session.commit()
        return jsonify({'message': 'Template deleted successfully'}), 200

    except Exception:
        db_session.rollback()
        return jsonify({'error': 'Something went wrong'}), 500

@app.route('/delete_template_messages', methods=['POST'])
@login_required
def delete_template_messages():
    data = request.get_json()
    template_name = data.get('template_name')
    username = data.get('username')
    message_indices = data.get('message_indices', [])

    if not template_name:
        return jsonify({'error': 'Template name is required'}), 400

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    if not message_indices or not isinstance(message_indices, list):
        return jsonify({'error': 'Message indices must be a non-empty list'}), 400

    db_session = get_session()
    try:
        # Check if the template exists and belongs to this user
        query = text("SELECT id FROM templates WHERE name = :name AND username = :username")
        result = db_session.execute(query, {"name": template_name, "username": username}).fetchone()

        if result is None:
            return jsonify({'error': 'Template not found or unauthorized'}), 403

        template_id = result[0]

        # Fetch all messages for the template to map indices to message IDs
        query = text("SELECT id FROM template_messages WHERE template_id = :template_id ORDER BY id")
        messages = db_session.execute(query, {"template_id": template_id}).fetchall()
        message_ids = [msg[0] for msg in messages]

        # Validate message indices
        valid_indices = [idx for idx in message_indices if 0 <= idx < len(message_ids)]
        if not valid_indices:
            return jsonify({'error': 'No valid message indices provided'}), 400

        # Delete selected messages
        message_ids_to_delete = [message_ids[idx] for idx in valid_indices]
        query = text("DELETE FROM template_messages WHERE id IN :ids")
        db_session.execute(query, {"ids": tuple(message_ids_to_delete)})

        db_session.commit()
        return jsonify({'message': 'Messages deleted successfully'}), 200

    except Exception:
        db_session.rollback()
        return jsonify({'error': 'Something went wrong'}), 500
    
@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Database dialect is working", conn.dialect.name)

    except Exception as e:
        print(f"Error initializing the application: {e}")
        sys.exit(1)

    init_excel()

    # Use absolute paths to the certs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(base_dir, 'certificates', 'cert.txt')
    key_path = os.path.join(base_dir, 'certificates', 'key.txt')

    ssl_context = (cert_path, key_path)

    app.run(debug=False, host='0.0.0.0', port=443, ssl_context=ssl_context)


