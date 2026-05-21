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
import pandas as pd
import copy

class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "/healthz" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

#==========================Semantic Engine=================================

num_retrieval=24
k_list = [5, 10, 20]
keyword_threshold = 1.25 #1.00

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

    # Remove stale reload signal on startup
    if os.path.exists(RELOAD_SIGNAL):
        os.remove(RELOAD_SIGNAL)
        print(f"Removed stale reload signal: {RELOAD_SIGNAL}")

    print("Loading Data...")
    vectordb, docs = load_vectorDB_docs()
    unit_course_map = load_module2course_data()
    bm25_obj, stemmer = load_keyword_docs(docs)
    docs_map = {str(doc.metadata["id"]): doc for doc in docs} # to find the original doc with doc id

    print("Loading Models...")
    base_model, advance_model = load_rerankers()

    # Store them in app.state
    app.state.vectordb = vectordb
    app.state.docs = docs
    app.state.bm25_obj = bm25_obj
    app.state.stemmer = stemmer
    app.state.docs_map = docs_map
    app.state.base_model = base_model
    app.state.advance_model = advance_model
    app.state.unit_course_map = unit_course_map

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

def score_with_llm(length, documents, question):
    
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
    
    Output a list of scores strictly with the list length: {length}.
    
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

    #print(scores)
    print(f"semantic matches: {scores}")

    return docs, scores
    

@timer_decorator
def get_courses_with_threshold(question, vectordb, num_retrieval=num_retrieval):
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

    print(f"keyword matches: {top_scores}")

    return top_docs, top_scores


@timer_decorator
def get_courses_with_threshold_from_keywords(query, request, num_retrieval=num_retrieval):

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
        if top_scores[i] <= keyword_threshold:
            out_scores = top_scores[0:i]
            out_docs = out_docs[0:i]
            break


    # all_idfs = request.app.state.bm25_obj.idf.values()
    # avg_idf = sum(all_idfs) / len(all_idfs)
    # median_idf = np.median(list(all_idfs))
    
    # print(f"Average IDF: {avg_idf:.2f}")
    # print(f"Median IDF: {median_idf:.2f}")

    print(f"keyword matches: {top_scores}")

    return out_docs, out_scores


@app.post("/search_lexical", response_model=List)
async def search_lexical_from_all_courses(question: str, request: Request):
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
    semantic_docs, semantic_scores = get_courses_with_threshold(question, request.app.state.vectordb, num_retrieval=num_retrieval)
    keyword_docs, keyword_scores = get_courses_with_threshold_from_keywords(question, request, num_retrieval=num_retrieval)
    return manual_rrf(semantic_docs, keyword_docs, request.app.state.docs_map)

@app.post("/search_RRF", response_model=List)
def search_from_all_courses(question: str, request: Request):
    docs, scores = combine_manual_rrf(question, request)
    return docs

@timer_decorator
def combine_all_manual_rrf(question, request):    
    semantic_docs, semantic_scores = get_courses(question, request.app.state.vectordb, num_retrieval=num_retrieval)
    keyword_docs, keyword_scores = get_courses_from_keywords(question, request, num_retrieval=num_retrieval)
    return manual_rrf(semantic_docs, keyword_docs, request.app.state.docs_map)

@app.post("/search_RRF_all_candidates", response_model=List)
def search_from_all_courses_all_candidates(question: str, request: Request):
    docs, scores = combine_all_manual_rrf(question, request)
    return docs

@timer_decorator
def add_unit_parent(courses, scores, request): 
    # current_course = set([course.metadata["id"] for course in courses])
    # parent_course = set()
    # for c in current_course:
    #     parent_course.add(request.app.state.unit_course_map.get(c))
    # full_course = current_course | parent_course
    # full_course.discard(None)
    # full_course = list(full_course)

    # full_course_with_content = [request.app.state.docs_map[str(doc_id)] for doc_id in full_course]

    # return full_course_with_content
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
    courses, scores = combine_manual_rrf(question, request)
    sorted_list_course, sorted_list_score = add_unit_parent(courses, scores, request)
    return sorted_list_course
#========================== Reranking =====================================

def load_rerankers():
    #Base model
    base_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print(f"MiniLM Device: {base_model.model.device}")

    # Advanced model
    #model_name = 'Qwen/Qwen3-Reranker-0.6B'
    # model_name = 'zeroentropy/zerank-2'
    
    # # 1. Manually load and fix the tokenizer first
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    
    # # 2. Pass the fixed tokenizer into the CrossEncoder
    # rerank_model_advance = CrossEncoder(
    #     model_name, 
    #     max_length=512, 
    #     tokenizer_args={'pad_token': tokenizer.pad_token},
    #     trust_remote_code=True
    # )
    # # 3. Explicitly set it on the underlying model just to be safe
    # rerank_model_advance.model.config.pad_token_id = tokenizer.pad_token_id

    # rerank_model_advance = CrossEncoder(
    #     'jinaai/jina-reranker-v2-base-multilingual',
    #     trust_remote_code=True,
    #     automodel_args={"torch_dtype": "float32"})

    from transformers import AutoModel
    
    rerank_model_advance = AutoModel.from_pretrained(
        'jinaai/jina-reranker-v3',
        dtype="auto",
        trust_remote_code=True
    )
    rerank_model_advance.eval()

    #print(f"Advanced rerank Device: {rerank_model_advance.model.device}")
    #print(f"Advanced rerank structure: {rerank_model_advance.model}")

    return base_model, rerank_model_advance

@timer_decorator
def combine_reranking(question, request):    
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
    
    #question = rewrite_query_with_llm(question)
    #print(question)

    pairs = [[question, doc.page_content] for doc in combined_docs]

    scores = request.app.state.base_model.predict(pairs)

    ranked_results = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True) 

    output_docs= [result[1] for result in ranked_results] #if result[0]> -0.3

    output_scores = [result[0] for result in ranked_results] # if result[0]> -0.3
    print("Base Reranking:")
    print(output_scores)

    return output_docs 

@app.post("/search_reranking_base", response_model=List)
def rerank_from_all_courses(question: str, request: Request):
    #question = rewrite_query_with_llm(question)
    #print(question)
    return combine_reranking(question, request)  


@timer_decorator
def combine_advance_reranking(question, request):    
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

    question = rewrite_query_with_llm(question)
    print(question)
    instruction = "Given a course description, identify all documents that provide relevant information, even if they use different terminology or they are sub-topics. "
    #instruction = "Given a search query, find documents that are broadly relevant to the topic, including synonyms and related concepts."
    # 2. Prepend the instruction to the query with a newline
    query_with_instruction = f"instruction: {instruction}\nquery: {question}"

    # pairs = [[query_with_instruction, doc.page_content] for doc in combined_docs]
    # #pairs = [[question, doc.page_content] for doc in combined_docs]

    # scores = app.state.advance_model.predict(pairs)

    # ranked_results = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True)

    # output_docs= [result[1] for result in ranked_results]

    # output_scores = [result[0] for result in ranked_results]

    docs= [doc.page_content for doc in combined_docs]
    try:
        ranked_results = app.state.advance_model.rerank(
            query=query_with_instruction, 
            documents=docs
        )

        #output_docs= [result['document'] for result in ranked_results]
    
        output_scores = [result['relevance_score'] for result in ranked_results]
    
        ranked_results = sorted(zip(output_scores, combined_docs), key=lambda x: x[0], reverse=True)
    
        output_docs= [result[1] for result in ranked_results]
    
        print("Advanced Reranking:")
        print(output_scores)
    
        return output_docs, output_scores 

    except ZeroDivisionError:
        print(f"Reranker failed for query: {query_with_instruction}")
        # Fallback: return original results without reranking
        return [], []

@timer_decorator
def combine_advance_reranking_with_threshold(question, request, threshold): 
    docs, scores = combine_advance_reranking(question, request)

    ranked_results = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    output_docs= [result[1] for result in ranked_results if result[0]> threshold] #

    output_scores = [result[0] for result in ranked_results if result[0]> threshold] 

    return output_docs, output_scores

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

    #print(len(doc_list))
    
    max_retries = 3
    attempts = 0
    
    while attempts < max_retries:
        try:
            # Get raw response from LLM
            raw_response = score_with_llm(len(doc_list), doc_list, question)
            doc_scores = json.loads(raw_response)
    
            # Check if the length matches
            if len(doc_scores) == len(doc_list):
                break
            else:
                print(f"Attempt {attempts + 1}: Length mismatch. Retrying...")
                
        except (json.JSONDecodeError, TypeError):
            print(f"Attempt {attempts + 1}: Invalid JSON format. Retrying...")
        
        attempts += 1
    
    if attempts == max_retries:
        print("Failed to get valid scores after maximum retries.")
    
    correct_doc_scores = doc_scores[:len(doc_list)]
    top_n_indices = np.argsort(correct_doc_scores)[::-1]
    # print(top_n_indices)
    print(len(doc_scores))
    print(len(combined_docs))

    # print(doc_list[41])
    # print(combined_docs[41])

    top_scores = [correct_doc_scores[i] for i in top_n_indices]
    top_docs = [combined_docs[i] for i in top_n_indices]  

    out_scores = top_scores
    out_docs = top_docs

    # for i in range(len(top_scores)):
    #     if top_scores[i] <= 0.5:
    #         out_scores = top_scores[0:i]
    #         out_docs = top_docs[0:i]
    #         break

    return top_docs, top_scores
    #return out_docs, out_scores

@timer_decorator
def combine_llm_scores_with_threshold(question, request, threshold): 
    docs, scores = combine_llm_scores(question, request)
    out_scores = scores
    out_docs = docs

    for i in range(len(top_scores)):
        if top_scores[i] <= threshold:
            out_scores = top_scores[0:i]
            out_docs = top_docs[0:i]
            break
    return out_docs, out_scores

@app.post("/search_reranking_advance", response_model=List)
def rerank_advance_from_all_courses(question: str, request: Request):
    #question = rewrite_query_with_llm(question)
    #print(question)
    docs, scores = combine_advance_reranking_with_threshold(question, request, -0.3)
    return  docs
  
@app.post("/search_llm_scores", response_model=List)
def rerank_llm_scores(question: str, request: Request):
    #question = rewrite_query_with_llm(question)
    #print(question)
    # Acquire the READ lock
    # Multiple threads can enter this 'gen_rlock' block at the same time.
    docs, scores = combine_llm_scores_with_threshold(question, request, 0.5)
    print(scores)
    return docs

def load_eval_mapping(filepath):
    query_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Map query_id -> {text, expected_ids}
            query_map[data['query_id']] = {
                "text": data['user_query'],
                "expected": set(data['expected_ids'])
            }
    return query_map

def load_eval_mapping_unit2course(filepath, request):
    query_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Map query_id -> {text, expected_ids}

            parent_course = set()
            for c in data['expected_ids']:
                parent_course.add(request.app.state.unit_course_map.get(c)) 
            parent_course.discard(None)
    
            query_map[data['query_id']] = {
                "text": data['user_query'],
                "expected": set(data['expected_ids']) | parent_course
            }
    return query_map

def calculate_recall_at_k(actual, predicted, k_list=[5, 10, 20]):
    """
    Calculates Recall@k for multiple values of k.
    
    Args:
        actual (list/set): The ground truth items (unique IDs).
        predicted (list): The ranked list of predicted items (ordered by relevance).
        k_list (list): The cut-off points (e.g., [5, 10, 20]).
        
    Returns:
        dict: Recall values for each k.
    """
    actual_set = set(actual)
    if not actual_set:
        return {k: 0.0 for k in k_list}
    
    results = []
    for k in k_list:
        # Slice the top k predictions
        top_k = list(predicted)[:k]
        
        # Count how many predicted items are in the actual set
        hits = len(set(top_k) & actual_set)
        
        # Recall = (Relevant items retrieved) / (Total relevant items)
        recall = hits / len(actual_set)
        results.append(round(recall, 4)) 
        
    return results

def calculate_mcc(tp, tn, fp, fn):
    # Calculate the numerator
    numerator = (tp * tn) - (fp * fn)
    
    # Calculate the denominator components
    d1 = tp + fp
    d2 = tp + fn
    d3 = tn + fp
    d4 = tn + fn
    
    # Check for zero denominator to avoid division errors
    if 0 in (d1, d2, d3, d4):
        return 0.0
        
    denominator = np.sqrt(d1 * d2 * d3 * d4)
    return numerator / denominator

def calculate_fixed_pool(q_id, retrieved_items, expected_ids, threshold=0.5):
    expected_set = set(expected_ids)
    
    # Identify split point
    split_idx = next((i for i, x in enumerate(retrieved_items) if x['score'] < threshold), len(retrieved_items))
    
    # Split the IDs into two sets
    above_threshold_ids = {item['id'] for item in retrieved_items[:split_idx]}
    below_threshold_ids = {item['id'] for item in retrieved_items[split_idx:]}
    
    # Use set math for counts
    tp = len(above_threshold_ids & expected_set)
    fp = len(above_threshold_ids - expected_set)
    fn = len(below_threshold_ids & expected_set)
    tn = len(below_threshold_ids - expected_set)

    # Count IDs that were expected but never even appeared in the retrieval pool
    retrieved_ids = {item['id'] for item in retrieved_items}
    missing_from_pool = len(expected_set - retrieved_ids)
    print(f"Missing from the pool: {missing_from_pool}")
    fn += missing_from_pool

    # Calcualte recall@k
    #recall = calculate_recall_at_k(expected_ids, retrieved_ids, k_list)
    # Calculate MCC
    mcc = calculate_mcc(tp, tn, fp, fn)

    specificity= tn / (tn + fp) if (tn + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # result = {"query_id": q_id, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
    #         "recall": tp / (tp + fn) if (tp + fn) > 0 else 0, f"recall@{k_list[0]}": recall[0], f"recall@{k_list[1]}": recall[1], f"recall@{k_list[2]}": recall[2]}
    result = {"query_id": q_id, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "specificity": specificity,
            "recall": recall, "mcc": mcc, "balanced_accu": (specificity + recall)/2}

    return result

def my_search_function(q_id, info, method, request):
    retrieved_items = []
    if method == "LLMs":
        docs, scores = combine_llm_scores(info['text'], request)

        for i in range(len(scores)):
            entry = {'id': docs[i].metadata["id"], 'score': scores[i]}
            retrieved_items.append(entry)

        return calculate_fixed_pool(q_id, retrieved_items, info['expected'], 0.5)
    elif method == "CrossEncoder":
        docs, scores = combine_advance_reranking(info['text'], request)

        for i in range(len(scores)):
            entry = {'id': docs[i].metadata["id"], 'score': scores[i]}
            retrieved_items.append(entry)

        return calculate_fixed_pool(q_id, retrieved_items, info['expected'], -0.1)        
    elif method == "RRF":
        semantic_docs, _ = get_courses(info['text'], request.app.state.vectordb, num_retrieval=num_retrieval)
        keyword_docs, _ = get_courses_from_keywords(info['text'], request, num_retrieval=num_retrieval)
        positive_semantic_docs, _ = get_courses_with_threshold(info['text'], request.app.state.vectordb, num_retrieval=num_retrieval)
        positive_keyword_docs, _ = get_courses_with_threshold_from_keywords(info['text'], request, num_retrieval=num_retrieval)
        
        above_threshold_ids = set([doc.metadata["id"] for doc in positive_semantic_docs] + [doc.metadata["id"] for doc in positive_keyword_docs])
        retrieved_ids = set([doc.metadata["id"] for doc in semantic_docs] + [doc.metadata["id"] for doc in keyword_docs])
        below_threshold_ids = retrieved_ids - above_threshold_ids

        tp = len(above_threshold_ids & info['expected'])
        fp = len(above_threshold_ids - info['expected'])
        fn = len(below_threshold_ids & info['expected'])
        tn = len(below_threshold_ids - info['expected'])

        # Count IDs that were expected but never even appeared in the retrieval pool
        missing_from_pool = len(info['expected'] - retrieved_ids)
        print(f"Missing from the pool: {missing_from_pool}")
        fn += missing_from_pool

        # Calculate recall@k
        #recall = calculate_recall_at_k(info['expected'], retrieved_ids, k_list)
        # Calculate MCC
        mcc = calculate_mcc(tp, tn, fp, fn)

        specificity= tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # result = {"query_id": q_id, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        #         "recall": tp / (tp + fn) if (tp + fn) > 0 else 0, f"recall@{k_list[0]}": recall[0], f"recall@{k_list[1]}": recall[1], f"recall@{k_list[2]}": recall[2]}
        result = {"query_id": q_id, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "specificity": specificity,
                "recall": recall, "mcc": mcc, "balanced_accu": (specificity + recall)/2}
        return result
    elif method == "LLMs_unit2course":
        docs, scores = combine_llm_scores(info['text'], request)
        docs, scores = add_unit_parent(docs, scores, request)

        for i in range(len(scores)):
            entry = {'id': docs[i].metadata["id"], 'score': scores[i]}
            retrieved_items.append(entry)

        return calculate_fixed_pool(q_id, retrieved_items, info['expected'], 0.5)
    elif method == "CrossEncoder_unit2course":
        docs, scores = combine_advance_reranking(info['text'], request)
        docs, scores = add_unit_parent(docs, scores, request)

        for i in range(len(scores)):
            entry = {'id': docs[i].metadata["id"], 'score': scores[i]}
            retrieved_items.append(entry)

        return calculate_fixed_pool(q_id, retrieved_items, info['expected'], -0.1)        
    elif method == "RRF_unit2course":
        semantic_docs, _ = get_courses(info['text'], request.app.state.vectordb, num_retrieval=num_retrieval)
        keyword_docs, _ = get_courses_from_keywords(info['text'], request, num_retrieval=num_retrieval)
        positive_semantic_docs, _ = get_courses_with_threshold(info['text'], request.app.state.vectordb, num_retrieval=num_retrieval)
        positive_keyword_docs, _ = get_courses_with_threshold_from_keywords(info['text'], request, num_retrieval=num_retrieval)
        
        above_threshold_ids = set([doc.metadata["id"] for doc in positive_semantic_docs] + [doc.metadata["id"] for doc in positive_keyword_docs])
        parent_course = set()
        for c in above_threshold_ids:
            parent_course.add(request.app.state.unit_course_map.get(c)) 
        parent_course.discard(None)

        above_threshold_ids = above_threshold_ids | parent_course

        retrieved_ids = set([doc.metadata["id"] for doc in semantic_docs] + [doc.metadata["id"] for doc in keyword_docs])
        retrieved_ids = retrieved_ids | parent_course

        below_threshold_ids = retrieved_ids - above_threshold_ids

        tp = len(above_threshold_ids & info['expected'])
        fp = len(above_threshold_ids - info['expected'])
        fn = len(below_threshold_ids & info['expected'])
        tn = len(below_threshold_ids - info['expected'])

        # Count IDs that were expected but never even appeared in the retrieval pool
        missing_from_pool = len(info['expected'] - retrieved_ids)
        print(f"Missing from the pool: {missing_from_pool}")
        fn += missing_from_pool

        # Calculate recall@k
        #recall = calculate_recall_at_k(info['expected'], retrieved_ids, k_list)
        # Calculate MCC
        mcc = calculate_mcc(tp, tn, fp, fn)

        specificity= tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # result = {"query_id": q_id, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        #         "recall": tp / (tp + fn) if (tp + fn) > 0 else 0, f"recall@{k_list[0]}": recall[0], f"recall@{k_list[1]}": recall[1], f"recall@{k_list[2]}": recall[2]}
        result = {"query_id": q_id, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "specificity": specificity,
                "recall": recall, "mcc": mcc, "balanced_accu": (specificity + recall)/2}
        return result

@app.post("/evaluate", response_model=List)
def evaluate_fixed_pool(method: str, request: Request):
    if method.endswith("unit2course"):
        mapping = load_eval_mapping_unit2course('./gemini_generate_dataset_updateByHuman.jsonl', request)
    else:
        mapping = load_eval_mapping('./gemini_generate_dataset_updateByHuman.jsonl')
    results = []
    
    for q_id, info in mapping.items():
        result = my_search_function(q_id, info, method, request) 
        results.append(result)  

    df_results = pd.DataFrame(results)

    balanced_accu = df_results['balanced_accu'].mean()
    balanced_accu_me = df_results['balanced_accu'].median()
    mcc = df_results['mcc'].mean()
    mcc_me = df_results['mcc'].median()
    # precision = df_results['precision'].mean()
    # precision_me = df_results['precision'].median()
    # recall = df_results['recall'].mean()
    # recall_me = df_results['recall'].median()
    # recall_k1 = df_results[f"recall@{k_list[0]}"].mean()
    # recall_k2 = df_results[f"recall@{k_list[1]}"].mean()
    # recall_k3 = df_results[f"recall@{k_list[2]}"].mean()
    # recall_k1_me = df_results[f"recall@{k_list[0]}"].median()
    # recall_k2_me = df_results[f"recall@{k_list[1]}"].median()
    # recall_k3_me = df_results[f"recall@{k_list[2]}"].median()
    # print(f"Average Precision: {precision}")
    # print(f"Average Recall: {recall}")
    # print(f"Average F1: {2*precision*recall/(precision+recall) if (precision+recall)> 0 else 0}")
    # print(f"Average recall@{k_list[0]}: {recall_k1}")
    # print(f"Average recall@{k_list[1]}: {recall_k2}")
    # print(f"Average recall@{k_list[2]}: {recall_k3}")
    # print(f"Median F1: {2*precision_me*recall_me/(precision_me+recall_me) if (precision_me+recall_me)> 0 else 0}")
    # print(f"Median recall@{k_list[0]}: {recall_k1_me}")
    # print(f"Median recall@{k_list[1]}: {recall_k2_me}")
    # print(f"Median recall@{k_list[2]}: {recall_k3_me}")
    print(f"Average MCC: {mcc}")
    print(f"Average Balanced Accuracy: {balanced_accu}")
    print(f"Median MCC: {mcc_me}")
    print(f"Median Balanced Accuracy: {balanced_accu_me}")

    return results 