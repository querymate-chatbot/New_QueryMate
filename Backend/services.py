import os
import threading
import pandas as pd
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import BaseExampleSelector
from langchain_openai import OpenAIEmbeddings
from db_config import get_session
from sqlalchemy import text
from cachetools import TTLCache

# State management
_embeddings = None
_selector_cache = TTLCache(maxsize=20, ttl=3600)
_base_faiss_index_path = os.path.join(os.getcwd(), "faiss_indexes")
_common_faiss_path = os.path.join(_base_faiss_index_path, "common")

def initialize_vector_store():
    global _embeddings
    try:
        _embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")
    ensure_faiss_folder()

def ensure_faiss_folder():
    if not os.path.exists(_base_faiss_index_path):
        os.makedirs(_base_faiss_index_path)
    else:
        print(f"FAISS folder already exists at {_base_faiss_index_path}")

def get_user_faiss_path(username):
    path = os.path.join(_base_faiss_index_path, username)
    return path

def fetch_common_examples():
    session = get_session()
    try:
        query = text("SELECT input, query, mode FROM bank.examples WHERE weightage = 100")
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=[col.lower() for col in result.keys()])
        examples = [{"input": row["input"], "query": row["query"], "mode": row["mode"]} for _, row in df.iterrows()]
        return examples
    except Exception as e:
        print(f"Error fetching common examples: {e}")
        return []

def fetch_user_examples(username):
    session = get_session()
    try:
        query = text("SELECT input, query, mode FROM bank.examples WHERE weightage = 50 AND username = :username")
        result = session.execute(query, {"username": username})
        df = pd.DataFrame(result.fetchall(), columns=[col.lower() for col in result.keys()])
        examples = [{"input": row["input"], "query": row["query"], "mode": row["mode"]} for _, row in df.iterrows()]
        return examples
    except Exception as e:
        print(f"Error fetching user examples for {username}: {e}")
        return []

@lru_cache(maxsize=1)
def load_common_faiss():
    ensure_faiss_folder()

    index_file_path = os.path.join(_common_faiss_path, "index.faiss")

    if not os.path.exists(index_file_path):
        examples = fetch_common_examples()
        if not examples:
            return None

        texts = [ex["input"] for ex in examples]
        metadatas = examples

        vector_store = FAISS.from_texts(
            texts,
            _embeddings,
            metadatas=metadatas
        )
        vector_store.save_local(_common_faiss_path)
    else:
        print("Common FAISS index file already exists.")

    common_store = FAISS.load_local(_common_faiss_path, _embeddings, allow_dangerous_deserialization=True)
    return common_store

def build_user_faiss(username):
    ensure_faiss_folder()
    user_faiss_path = get_user_faiss_path(username)
    if os.path.exists(os.path.join(user_faiss_path, "index.faiss")):
        return

    user_examples = fetch_user_examples(username)
    if not user_examples:
        return

    user_texts = [ex["input"] for ex in user_examples]
    user_metadatas = user_examples

    user_store = FAISS.from_texts(
        user_texts,
        _embeddings,
        metadatas=user_metadatas
    )
    user_store.save_local(user_faiss_path)

def load_user_faiss(username):
    build_user_faiss(username)
    user_faiss_path = get_user_faiss_path(username)
    if not os.path.exists(os.path.join(user_faiss_path, "index.faiss")):
        return None

    user_store = FAISS.load_local(user_faiss_path, _embeddings, allow_dangerous_deserialization=True)
    return user_store

class CombinedExampleSelector(BaseExampleSelector):
    def __init__(self, common_vectorstore, user_vectorstore, k=3):
        self.common_store = common_vectorstore
        self.user_store = user_vectorstore
        self.k = k

    def select_examples(self, input_variables):
        query = input_variables["input"]
        common_results = self.common_store.similarity_search_with_score(query, k=self.k)
        user_results = []
        if self.user_store:
            user_results = self.user_store.similarity_search_with_score(query, k=self.k)
        combined = common_results + user_results
        combined.sort(key=lambda x: x[1])
        top_k = combined[:self.k]
        examples = [doc.metadata for doc, score in top_k]
        return examples

    def add_example(self, example):
        pass

def get_example_selector_config(username):
    if username in _selector_cache:
        return _selector_cache[username]

    common_store = load_common_faiss()
    if common_store is None:
        return None

    user_store = load_user_faiss(username)

    selector = CombinedExampleSelector(
        common_vectorstore=common_store,
        user_vectorstore=user_store,
        k=3
    )
    _selector_cache[username] = selector
    return selector

def add_new_example_for_training(prompt, sqlquery, username, mode="default", weightage=50):
    try:
        _add_new_example_to_selector(prompt, sqlquery, username, mode, weightage)
    except Exception as e:
        print(f"Error adding new example: {e}")

def _add_new_example_to_selector(new_input, query, username, mode="default", weightage=50):
    new_embedding = _embeddings.embed_query(new_input)

    metadata = {
        "input": new_input,
        "query": query,
        "mode": mode,
        "weightage": weightage,
        "username": username
    }

    user_store = load_user_faiss(username)
    if user_store is None:
        user_store = FAISS.from_texts(
            [new_input],
            _embeddings,
            metadatas=[metadata]
        )
    else:
        user_store.add_texts(
            texts=[new_input],
            embeddings=[new_embedding],
            metadatas=[metadata]
        )

    user_faiss_path = get_user_faiss_path(username)
    user_store.save_local(user_faiss_path)

    if username in _selector_cache:
        del _selector_cache[username]