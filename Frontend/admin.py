from functools import wraps
from flask import Blueprint, jsonify, request, redirect, url_for, session
from db_config import get_session
from sqlalchemy import text
import pandas as pd
from collections import defaultdict
from datetime import datetime, date

admin = Blueprint('admin', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or session['username'].lower() != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@admin.route('/querymate/admin/users', methods=['GET'])
@admin_required
def get_users():
    db_session = get_session()
    try:
        query = text("""
            SELECT users.userid, users.email, users.company_name, users.username, users.password, users.set_flag, users.role, roles.access
            FROM users
            JOIN roles ON roles.role = users.role
        """)
        users = db_session.execute(query).fetchall()

        users_list = [{
            'userid': user[0],
            'email': user[1],
            'company_name': user[2],
            'username': user[3],
            'password': user[4],
            'set_flag': user[5],
            'role': user[6],
            'access': user[7]
        } for user in users]

        return jsonify(users_list)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/update_user', methods=['POST'])
@admin_required
def update_user():
    data = request.get_json()
    userid = data['userid']
    role = data['role']
    set_flag = data['set_flag']

    db_session = get_session()
    try:
        query = text("""
            UPDATE users
            SET role = :role, set_flag = :set_flag
            WHERE userid = :userid
        """)
        db_session.execute(query, {"role": role, "set_flag": set_flag, "userid": userid})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/delete_user', methods=['POST'])
@admin_required
def delete_user():
    data = request.get_json()
    userid = data['userid']

    db_session = get_session()
    try:
        query = text("DELETE FROM users WHERE userid = :userid")
        db_session.execute(query, {"userid": userid})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/roles', methods=['GET'])
@admin_required
def get_roles():
    db_session = get_session()
    try:
        query = text("SELECT role, access FROM roles")
        roles = db_session.execute(query).fetchall()

        roles_list = [{
            'role': role[0],
            'access': role[1]
        } for role in roles]

        return jsonify(roles_list)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/update_role', methods=['POST'])
@admin_required
def update_role():
    data = request.get_json()
    role = data['role']
    access = data['access']

    db_session = get_session()
    try:
        query = text("UPDATE roles SET access = :access WHERE role = :role")
        db_session.execute(query, {"access": access, "role": role})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/add_role', methods=['POST'])
@admin_required
def add_role():
    data = request.get_json()
    role = data['role']
    access = data['access']

    db_session = get_session()
    try:
        query_check = text("SELECT 1 FROM roles WHERE role = :role")
        if db_session.execute(query_check, {"role": role}).fetchone():
            return jsonify({"success": False, "error": "Role already exists"})

        query_insert = text("INSERT INTO roles (role, access) VALUES (:role, :access)")
        db_session.execute(query_insert, {"role": role, "access": access})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/delete_role', methods=['POST'])
@admin_required
def delete_role():
    data = request.get_json()
    role = data['role']

    db_session = get_session()
    try:
        query_check = text("SELECT 1 FROM roles WHERE role = :role")
        if not db_session.execute(query_check, {"role": role}).fetchone():
            return jsonify({"success": False, "error": "Role does not exist"})

        query_delete = text("DELETE FROM roles WHERE role = :role")
        db_session.execute(query_delete, {"role": role})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/query_repository', methods=['GET'])
@admin_required
def get_query():
    db_session = get_session()
    try:
        query = text("SELECT input, query, weightage, username, mode FROM bank.examples")
        queries = db_session.execute(query).fetchall()

        query_list = [{
            'input': que[0],
            'query': que[1],
            'weightage': que[2],
            'username': que[3],
            'mode': que[4]
        } for que in queries]

        return jsonify(query_list)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/update_repository', methods=['POST'])
@admin_required
def update_repository():
    data = request.get_json()
    input_val = data['input']
    query = data['query']
    weightage = data['weightage']
    mode = data['mode']

    db_session = get_session()
    try:
        sql_query = text("""
            UPDATE bank.examples
            SET input = :input, query = :query, weightage = :weightage, mode = :mode
            WHERE input = :input
        """)
        db_session.execute(sql_query, {"input": input_val, "query": query, "weightage": weightage, "mode": mode})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/verify_query', methods=['POST'])
@admin_required
def verify_query():
    data = request.get_json()
    query = data['query']

    print("Sql received", query)

    db_session = get_session()
    try:
        result = db_session.execute(text(query)).fetchall()
        df = pd.DataFrame(result)
        json_result = df.to_json(orient='split')
        return jsonify({"success": True, "data": json_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/delete_query', methods=['POST'])
@admin_required
def delete_query():
    data = request.get_json()
    input_val = data['query']

    db_session = get_session()
    try:
        sql_query = text("DELETE FROM bank.examples WHERE input = :input")
        db_session.execute(sql_query, {"input": input_val})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/repository_statistics', methods=['GET'])
@admin_required
def repository_statistics():
    db_session = get_session()
    query = text("""
        SELECT TOP (1000) [input], [query], [weightage], [mode], [timestamp], [username]
        FROM [Car Store DB].[bank].[examples]
    """)
    rows = db_session.execute(query).fetchall()

    mode_count = defaultdict(int)
    weightage_count = {50: 0, 100: 0}
    total_rows = len(rows)

    for row in rows:
        mode = row[3]
        weightage = row[2]
        mode_count[mode] += 1
        if weightage in weightage_count:
            weightage_count[weightage] += 1

    statistics = {
        'total_rows': total_rows,
        'unique_modes': dict(mode_count),
        'weightage_distribution': weightage_count,
    }

    return jsonify(statistics)

@admin.route('/querymate/admin/policy', methods=['GET'])
@admin_required
def policy():
    db_session = get_session()
    query = text("""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_CATALOG = 'Car Store DB'
          AND TABLE_SCHEMA = 'bank'
          AND TABLE_NAME NOT IN ('examples')
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    rows = db_session.execute(query).fetchall()

    table_columns = {}
    for table_name, column_name in rows:
        if table_name not in table_columns:
            table_columns[table_name] = []
        table_columns[table_name].append(column_name)

    return jsonify(table_columns)

@admin.route('/querymate/admin/get_datalist', methods=['POST'])
@admin_required
def get_datalist():
    data = request.get_json()
    table = data.get('table')
    column = data.get('column')

    if not table or not column:
        return jsonify({'error': 'Table or column not specified'}), 400

    db_session = get_session()
    query = text(f"SELECT DISTINCT {column} FROM bank.{table}")
    rows = db_session.execute(query).fetchall()
    data = [row[0] for row in rows]
    return jsonify(data)

@admin.route('/querymate/admin/create_policy', methods=['POST'])
@admin_required
def create_policy():
    data = request.get_json()
    policy_name = data.get('policy_name')
    access_level = data.get('access_level')

    if not policy_name or not access_level:
        return jsonify({'error': 'policy_name or access_level not specified'}), 400

    db_session = get_session()
    try:
        query = text("""
            INSERT INTO [Car Store DB].[dbo].[access_policy] ([policy_name], [access_level])
            VALUES (:policy_name, :access_level)
        """)
        db_session.execute(query, {"policy_name": policy_name, "access_level": access_level})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/get_policy', methods=['GET'])
@admin_required
def get_policy():
    db_session = get_session()
    try:
        query = text("SELECT policy_name, access_level FROM access_policy")
        policies = db_session.execute(query).fetchall()

        policies_list = [{
            'policy_name': policy[0],
            'access_level': policy[1]
        } for policy in policies]

        return jsonify(policies_list)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/assign_policy', methods=['POST'])
@admin_required
def assign_policy():
    data = request.get_json()
    users = data['users']

    db_session = get_session()
    try:
        for user in users:
            policy_name = user['policy_name']
            username = user['username']

            query = text("SELECT access_level FROM access_policy WHERE policy_name = :policy_name")
            result = db_session.execute(query, {"policy_name": policy_name}).fetchone()

            if result:
                access_level = result[0]
                query_insert = text("""
                    IF NOT EXISTS (SELECT 1 FROM user_group WHERE username = :username AND policy = :policy)
                    BEGIN
                        INSERT INTO user_group (username, policy, access_level)
                        VALUES (:username, :policy, :access_level)
                    END
                """)
                db_session.execute(query_insert, {
                    "username": username,
                    "policy": policy_name,
                    "access_level": access_level
                })
                db_session.commit()

        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/get_users_in_policy', methods=['POST'])
@admin_required
def get_users_in_policy():
    data = request.get_json()
    policy_name = data['policy_name']

    db_session = get_session()
    try:
        query = text("SELECT username FROM user_group WHERE policy = :policy")
        users = db_session.execute(query, {"policy": policy_name}).fetchall()

        user_list = [{'username': user[0]} for user in users]
        return jsonify(user_list)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/delete_users_in_policy', methods=['POST'])
@admin_required
def delete_users_in_policy():
    data = request.get_json()
    username = data['username']
    policy_name = data['policy_name']

    db_session = get_session()
    try:
        query = text("DELETE FROM user_group WHERE username = :username AND policy = :policy")
        db_session.execute(query, {"username": username, "policy": policy_name})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/delete_policy', methods=['POST'])
@admin_required
def delete_policy():
    data = request.get_json()
    policy_name = data['policy_name']

    db_session = get_session()
    try:
        query = text("DELETE FROM access_policy WHERE policy_name = :policy_name")
        db_session.execute(query, {"policy_name": policy_name})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/token_management', methods=['GET'])
@admin_required
def token_management():
    db_session = get_session()
    try:
        query = text("""
            SELECT token_day, username, email, token_used, token_limit, token_cost
            FROM token_management
        """)
        token_users = db_session.execute(query).fetchall()

        aggregated_data = {}
        day_usage = {}

        for token_user in token_users:
            token_day, username, email, token_used, token_limit, token_cost = token_user

            if isinstance(token_day, date):
                token_day_str = token_day.strftime('%Y-%m-%d')
            else:
                token_day_str = str(token_day)

            if username not in aggregated_data:
                aggregated_data[username] = {
                    'email': email,
                    'token_used': 0,
                    'token_limit': token_limit,
                }

            aggregated_data[username]['token_used'] += token_used

            if token_day_str not in day_usage:
                day_usage[token_day_str] = {
                    'token_used': 0,
                    'costperday': 0
                }
            day_usage[token_day_str]['token_used'] += token_used
            day_usage[token_day_str]['costperday'] += token_cost

        result = []
        for username, data in aggregated_data.items():
            result.append({
                'username': username,
                'email': data['email'],
                'token_used': data['token_used'],
                'token_limit': data['token_limit']
            })

        return jsonify({
            'aggregated_data': result,
            'day_usage': day_usage
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@admin.route('/querymate/admin/update_token', methods=['POST'])
@admin_required
def update_token():
    data = request.get_json()
    username = data['username']
    token_limit = data['token_limit']

    db_session = get_session()
    try:
        query = text("UPDATE token_management SET token_limit = :token_limit WHERE username = :username")
        db_session.execute(query, {"token_limit": token_limit, "username": username})
        db_session.commit()
        return jsonify({"success": True})
    except Exception as e:
        print("Error in update_token:", e)
        db_session.rollback()
        return jsonify({"success": False, "error": str(e)})