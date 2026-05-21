from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Tuple, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
#from langchain_community.retrievers import BM25Retriever
#from langchain_classic.retrievers import EnsembleRetriever
from transformers import AutoTokenizer
import time
import threading
from functools import wraps
from contextlib import asynccontextmanager
import re
from dotenv import load_dotenv, find_dotenv
import requests
import logging
from readerwriterlock import rwlock
import copy

class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "/healthz" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

#==========================Semantic Engine=================================

num_retrieval=24
_ = load_dotenv(find_dotenv()) # read local .env file
api_key  = os.environ['ANVILGPT_API']


RELOAD_SIGNAL = os.environ.get('DATA_FILE_PATH', './data.jsonl').replace('data.jsonl', '.reload') or os.environ.get('DATA_FILE2_PATH', './course_unit_map.jsonl')

def reload_data_if_needed(app):
    """Check for .reload signal and hot-reload data."""
    if not os.path.exists(RELOAD_SIGNAL):
        return

    print("Reload signal detected. Hot-reloading data...")
    try:
        # Close old chromadb connection before CronJob-cleared dir is reopened.
        # Without this, SQLite detects the deleted file (error 1032: DBMOVED)
        # and puts the connection into readonly mode, blocking the new client.
        with app.state.reload_rwlock.gen_wlock():
            try:
                app.state.vectordb._client.close()
            except Exception as e:
                print(f"Warning: could not close old vectordb client: {e}")
    
            vectordb, docs = load_vectorDB_docs()
            unit_course_map = load_module2course_data()
            bm25_obj, stemmer = load_keyword_docs(docs)
            docs_map = {str(doc.metadata["id"]): doc for doc in docs}
    
            # Atomic swap of app.state
            app.state.vectordb = vectordb
            app.state.docs = docs
            app.state.bm25_obj = bm25_obj
            app.state.stemmer = stemmer
            app.state.docs_map = docs_map
            app.state.unit_course_map = unit_course_map

        os.remove(RELOAD_SIGNAL)
        print("Hot-reload complete.")
    except Exception as e:
        print(f"Hot-reload failed: {e}")

def background_watcher(app, interval=10):
    """Periodically check for reload signal."""
    while True:
        time.sleep(interval)
        reload_data_if_needed(app)

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Initialize the lock and store it in app.state
    app.state.reload_rwlock = rwlock.RWLockFair()

    # Remove stale reload signal on startup
    if os.path.exists(RELOAD_SIGNAL):
        os.remove(RELOAD_SIGNAL)
        print(f"Removed stale reload signal: {RELOAD_SIGNAL}")

    print("Loading Data...")
    vectordb, docs = load_vectorDB_docs()
    unit_course_map = load_module2course_data()
    bm25_obj, stemmer = load_keyword_docs(docs)
    docs_map = {str(doc.metadata["id"]): doc for doc in docs} # to find the original doc with doc id

    #print("Loading Models...")
    #base_model, advance_model = load_rerankers()

    # Store them in app.state
    app.state.vectordb = vectordb
    app.state.docs = docs
    app.state.bm25_obj = bm25_obj
    app.state.stemmer = stemmer
    app.state.docs_map = docs_map
    app.state.unit_course_map = unit_course_map
    #app.state.base_model = base_model
    #app.state.advance_model = advance_model

    watcher = threading.Thread(target=background_watcher, args=(app, 10), daemon=True)
    watcher.start()
    print("Background data watcher started (interval=10s)")

    yield

    # --- CLEANUP (Shutdown) ---
    print("Shutting down... releasing resources.")


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # More precise than time.time()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f}s")
        return result
    return wrapper

def rewrite_query_with_llm(question):
    
    prompt= """
    ### Role
    You are an Academic Curriculum Designer. Your task is to transform casual or informal user queries into professional, high-impact course descriptions suitable for a university or professional training catalog.
    
    ### Objectives
    1. **Academic Tone:** Use formal, pedagogical language (e.g., "examine," "master," "synthesize," "foundational principles").
    2. **Vocabulary Mirroring:** Identify core keywords or technical concepts in the user's query and weave them into the formal description to ensure the course remains relevant to their intent.
    3. **Structure:** The output must be exactly one to three sentences long.
    4. **Directness:** Provide only the rewritten description. Do not include introductory text like "Here is your description."
    
    ### Example
    * **User Query:** "I want to learn how to make cool websites with React and make them look good on phones."
    * **Rewritten Description:** "This course provides a comprehensive deep dive into building responsive web applications using the React framework. Students will master front-end architecture and mobile-first design principles to create seamless, high-performance user interfaces."
    """
    url = "https://anvilgpt.rcac.purdue.edu/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-oss:120b",
        "messages": [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": question   
        }
        ],
        "temperature": 0,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=body)
    # print(response)
    # if response.status_code == 200:
    #     print(response.text)
    # else:
    #     raise Exception(f"Error: {response.status_code}, {response.text}")
    return json.loads(response.text)["choices"][0]["message"]["content"]

def score_with_llm(documents, question):
    
    # Use an f-string to inject the variables into the prompt:
    prompt= f"""
    Role: You are an academic curriculum analyst. Your task is to evaluate a list of course descriptions and score their 'broad relevance' to a specific user-provided Course Query.

    Definitions:
    
    Broad Relevance: The course content overlaps significantly with the themes, learning outcomes, or subject matter of the query, even if the course titles do not match perfectly.
    
    Relevance Score: Assign a score from 0.0 to 1.0 where:
    
    0.0 - 0.49: Irrelevant; the content is entirely unrelated.
    
    0.5 - 0.69: Tangentially related; minor overlap in topics or industry application.

    0.7 - 1.0: Highly relevant; the course covers core concepts central to the query.
    
    Instructions:
    
    Analyze the provided Course Query carefully to understand its core intent.
    
    Evaluate each Course Description against the query.
    
    Output your results strictly as a list of scores, where the index of each score matches the index of the corresponding course description.
    
    Course Query:
    {question}
    
    Course Descriptions:
    {documents}

    """
    url = "https://anvilgpt.rcac.purdue.edu/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-oss:120b",
        "messages": [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": " "   
        }
        ],
        "temperature": 0,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=body)
    # print(response)
    # if response.status_code == 200:
    #     print(response.text)
    # else:
    #     raise Exception(f"Error: {response.status_code}, {response.text}")
    return json.loads(response.text)["choices"][0]["message"]["content"]

@timer_decorator
def get_courses(question, vectordb, num_retrieval=num_retrieval):
    results = vectordb.similarity_search_with_relevance_scores(question, k=num_retrieval)
    #docs = vectordb.similarity_search(question,k=num_retrieval)
    #results = vectordb.similarity_search_with_score(question,k=num_retrieval)
    # for doc in docs:
    #     print(doc.metadata)
    docs = [result[0] for result in results]
    scores = [result[1] for result in results]

    out_scores = scores
    out_docs = docs

    for i in range(len(scores)):
        if scores[i] <= 0.45:
            out_scores = scores[0:i]
            out_docs = docs[0:i]
            break

    #print(scores)
    print(f"semantic matches: {scores}")

    #return docs, scores
    return out_docs, out_scores

app = FastAPI(title="Cyberfaces Smartsearch API", lifespan=lifespan)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["id"] = record.get("id")
#     metadata["title"] = record.get("title")
#     metadata["created_at"] = record.get("created_at")
#     metadata["updated_at"] = record.get("updated_at")
#     metadata["is_course"] = record.get("is_course")

#     return metadata
def load_module2course_data():

    file_path = os.environ.get('DATA_FILE2_PATH', './course_unit_map.jsonl')

    unit_course_map = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
                # Skip empty lines if any
                if not line.strip():
                    continue
                    
                original_data = json.loads(line.strip())
                
                # 2. Extract the IDs
                unit_id = original_data.get("unit_id")
                course_id = original_data.get("course_id")
                
                # 3. Add to dictionary (ensuring unit_id exists)
                if unit_id is not None:
                    unit_course_map[unit_id] = course_id


    return unit_course_map

def load_vectorDB_docs():
    # 1. Load each line of the jsonl file.
    
    file_path = os.environ.get('DATA_FILE_PATH', './data.jsonl')
    
    loader = JSONLoader(
        file_path=file_path,
        #content_key="description", 
        jq_schema='.description',      
        json_lines=True,
        #metadata_func=metadata_func # Uncomment if using a custom metadata extractor
    )
    
    docs = loader.load()
    
    #print(docs)
    # 2. Add other attributes into the meta data.
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            original_data = json.loads(line.strip())
            if i+1 == docs[i].metadata["seq_num"]:
                docs[i].metadata["id"] = original_data.get("id")
                docs[i].metadata["title"] = original_data.get("title")
                docs[i].metadata["created_at"] = original_data.get("created_at")
                docs[i].metadata["updated_at"] = original_data.get("updated_at")
                docs[i].metadata["is_course"] = original_data.get("is_course")
                if not docs[i].page_content:
                    docs[i].page_content = original_data.get("title")
                else:
                    docs[i].page_content = original_data.get("title") + "|" + docs[i].page_content
    
    #print(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/" + "all-mpnet-base-v2")
    
    # 3. Load documents into Chroma and embed them automatically
    def is_dir_empty(path):
        with os.scandir(path) as it:
            return not any(it)
    
    persist_directory = os.environ.get('CHROMADB_PATH', './chromaDB')
    os.makedirs(persist_directory, exist_ok=True)

    # Chroma handles the embedding automatically upon adding documents.
    if is_dir_empty(persist_directory):
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # use consine distance instead of L2
            #collection_metadata={"hnsw:space": "l2"}  # use consine distance instead of L2
        )
        print(vectordb._collection.count())
    else:
        vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings,
        )
        print("Current Space:", vectordb._collection.count())
    
    if vectordb._collection.metadata:
        print("Current Space2:", vectordb._collection.metadata.get("hnsw:space"))
    
    print(f"Data from {file_path} successfully embedded and stored in {persist_directory}")

    #semantic_retriever = vectordb.as_retriever(search_kwargs={"k": num_retrieval})
    return vectordb, docs

@app.post("/search_semantic", response_model=List)
def search_semantic_from_all_courses(question: str, request: Request):
    # Acquire the READ lock
    # Multiple threads can enter this 'gen_rlock' block at the same time.
    with app.state.reload_rwlock.gen_rlock():
        docs, scores = get_courses(question, request.app.state.vectordb, num_retrieval=num_retrieval) 
    #docs = [result[0] for result in results]
    #docs =  semantic_retriever.invoke(question)  
    return docs

#==========================Lexical Engine=================================
# this implement BM25 keywords search
# Download necessary data

def clean_token(w):
    # Allow alphanumeric characters, "/", and "-"
    return "".join(char for char in w if char.isalnum() or char == "-")

def deep_clean_query(query, stemmer):

    stop_words = set(stopwords.words('english'))

    # Pre-process hyphens and slashes
    query = query.replace('-', '').replace('/', ' ')

    tokens = word_tokenize(query.lower())

    # Use a Regex that allows letters, numbers, and tech symbols (#, +)
    # This rejects "pure" punctuation like "." or "!"
    tech_pattern = re.compile(r'^[a-z0-9+#]+$')

    # Remove stopwords AND stem the remaining words
    cleaned = [stemmer.stem(w) for w in tokens if (w.isalpha() or tech_pattern.match(w)) and w not in stop_words]
    
    return cleaned

def load_keyword_docs(docs):

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    stemmer = PorterStemmer()

    # 1. Clean the corpus in the same way as query
    bm25_docs = [deep_clean_query(doc.page_content, stemmer) for doc in docs]
    
    # 2. Initialize the BM25 object
    bm25 = BM25Okapi(bm25_docs)


    return bm25, stemmer

# 3. Clean query to get non-stopping keywords and search
@timer_decorator
def get_courses_from_keywords(query, request, num_retrieval=num_retrieval):

    # the docs returned here from the chorma vectorbase with the same format as semantic search
    
    bm25_query = deep_clean_query(query, request.app.state.stemmer)
    doc_scores = request.app.state.bm25_obj.get_scores(bm25_query)

    # Get top N results
    top_n_indices = np.argsort(doc_scores)[::-1][:num_retrieval]
    #max_score = np.max(doc_scores)
    #min_score = np.min(doc_scores)

   #top_scores = [(doc_scores[i]-min_score)/(max_score-min_score) for i in top_n_indices]
    top_scores = [doc_scores[i]/len(bm25_query) for i in top_n_indices] # apply Query-Length Normalization
    top_docs = [request.app.state.docs[i] for i in top_n_indices]  # from chorma vectorbase instead of bm25_docs
    #top_n = bm25.get_top_n(bm25_query, corpus, n=num_retrieval)

    out_scores = top_scores
    out_docs = top_docs

    for i in range(len(top_scores)):
        if top_scores[i] <= 1.25:
            out_scores = top_scores[0:i]
            out_docs = top_docs[0:i]
            break


    # all_idfs = request.app.state.bm25_obj.idf.values()
    # avg_idf = sum(all_idfs) / len(all_idfs)
    # median_idf = np.median(list(all_idfs))
    
    # print(f"Average IDF: {avg_idf:.2f}")
    # print(f"Median IDF: {median_idf:.2f}")

    print(f"keyword matches: {top_scores}")

    #return top_docs, top_scores
    return out_docs, out_scores


@app.post("/search_lexical", response_model=List)
async def search_lexical_from_all_courses(question: str, request: Request):
    # Acquire the READ lock
    # Multiple threads can enter this 'gen_rlock' block at the same time.
    with app.state.reload_rwlock.gen_rlock():
        docs, scores = get_courses_from_keywords(question, request, num_retrieval=num_retrieval)    
    return docs

#==========================Lexical Engine2=================================
# this uses langchain BM25 keywords search
# not use due to lexical search need processing with docs and query

# bm25_retriever = BM25Retriever.from_documents(docs)
# bm25_retriever.k = k

# @app.post("/search_keywords", response_model=List)
# def search_keywords_from_all_courses(question: str):
#     docs = bm25_retriever.invoke(question)   
#     return docs

#==========================RRF============================================

# not use due to lexical search need processing with docs and query
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, semantic_retriever], 
#     weights=[0.5, 0.5]  # RRF ignores weights, but LangChain requires them for the object
# )

# @app.post("/search_RRF", response_model=List)
# def search_from_all_courses(question: str):
#     docs =  ensemble_retriever.invoke(question)   
#     return docs

def manual_rrf(semantic_docs, keyword_docs, docs_map, k=60):
    rrf_scores = {}
    
    # Process both lists
    for rank, docs in enumerate([semantic_docs, keyword_docs]):
        for i, doc in enumerate(docs):
            doc_id = str(doc.metadata["id"])
            # Rank starts at 1
            score = 1.0 / (k + (i + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
            
    # Sort by score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    output_docs = [docs_map[doc_id] for doc_id, _ in sorted_results]
    scores = [score for _, score in sorted_results]

    print("RRF:")
    print(scores)

    return output_docs, scores

@timer_decorator
def combine_manual_rrf(question, request):    
    semantic_docs, semantic_scores = get_courses(question, request.app.state.vectordb, num_retrieval=num_retrieval)
    keyword_docs, keyword_scores = get_courses_from_keywords(question, request, num_retrieval=num_retrieval)
    return manual_rrf(semantic_docs, keyword_docs, request.app.state.docs_map)

@app.post("/search_RRF", response_model=List)
def search_from_all_courses(question: str, request: Request):
    # Acquire the READ lock
    # Multiple threads can enter this 'gen_rlock' block at the same time.
    with app.state.reload_rwlock.gen_rlock():
        docs, scores = combine_manual_rrf(question, request)
        return docs

@timer_decorator
def add_unit_parent(courses, scores, request): 
    current_course_ids = set([course.metadata["id"] for course in courses])
    out_courses = copy.deepcopy(courses)
    out_scores = copy.deepcopy(scores)

    for i in range(len(scores)):
        unit_id = courses[i].metadata["id"]
        parent_id = request.app.state.unit_course_map.get(unit_id)
        if parent_id not in current_course_ids and parent_id is not None:
            out_courses.append(request.app.state.docs_map[str(parent_id)])
            out_scores.append(scores[i])

    # sort
    sorted_pairs = sorted(zip(out_scores, out_courses), key=lambda x: x[0], reverse=True)
    sorted_list_score, sorted_list_course = zip(*sorted_pairs)
    sorted_list_score = list(sorted_list_score)
    sorted_list_course = list(sorted_list_course)    
    return sorted_list_course, sorted_list_score


@app.post("/search_RRF_unit2course", response_model=List)
# this endpoint incorporate the logic to automatically include parental course if module is a match
def search_from_all_courses_unit2course(question: str, request: Request):
    # Acquire the READ lock
    # Multiple threads can enter this 'gen_rlock' block at the same time.
    with app.state.reload_rwlock.gen_rlock():
        courses, scores = combine_manual_rrf(question, request)
        sorted_list_course, sorted_list_score = add_unit_parent(courses, scores, request)
        return sorted_list_course

#========================== Reranking =====================================


@timer_decorator
def combine_llm_scores(question, request):    
    semantic_docs, _ = get_courses(question, request.app.state.vectordb, num_retrieval=num_retrieval)
    keyword_docs, _ = get_courses_from_keywords(question, request, num_retrieval=num_retrieval)

    # below only take the unique docs (remove duplicates btw semantic search and keyword search)
    seen = set()
    combined_docs = semantic_docs 

    for doc in semantic_docs:
        seen.add(doc.metadata["id"])

    for doc in keyword_docs:
        if doc.metadata["id"] not in seen:
            combined_docs.append(doc)

    doc_list = [doc.page_content for doc in combined_docs]
    print(doc_list)
    return score_with_llm(doc_list, question)
  
@app.post("/search_llm_scores", response_model=List)
def rerank_llm_scores(question: str, request: Request):
    #question = rewrite_query_with_llm(question)
    #print(question)
    # Acquire the READ lock
    # Multiple threads can enter this 'gen_rlock' block at the same time.
    with app.state.reload_rwlock.gen_rlock():
        return json.loads(combine_llm_scores(question, request)) 

