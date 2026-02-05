from fastapi import FastAPI, HTTPException, Request
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
from functools import wraps
from contextlib import asynccontextmanager

#==========================Semantic Engine=================================

num_retrieval=5

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading Data...")
    vectordb, docs = load_vectorDB_docs()
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

    url = "https://genai.rcac.purdue.edu/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model_name,
        "messages": [
        {
            "role": "system",
            "content": "You rewrite user input as a concise search query for courses"
        },
        {
            "role": "user",
            "content": question   
        }
        ],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=body)
    print(response)
    if response.status_code == 200:
        print(response.text)
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    return json.loads(response.text)["choices"][0]["message"]["content"]

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

app = FastAPI(title="Cyberfaces Smartsearch API", lifespan=lifespan)

# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["id"] = record.get("id")
#     metadata["title"] = record.get("title")
#     metadata["created_at"] = record.get("created_at")
#     metadata["updated_at"] = record.get("updated_at")
#     metadata["is_course"] = record.get("is_course")

#     return metadata

def load_vectorDB_docs():
    # 1. Load each line of the jsonl file.
    
    file_path = './data.jsonl'
    
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
    
    print(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/" + "all-mpnet-base-v2")
    
    # 3. Load documents into Chroma and embed them automatically
    def is_dir_empty(path):
        with os.scandir(path) as it:
            return not any(it)
    
    persist_directory = './chromaDB'
    
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

def deep_clean_query(query, stemmer):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(query.lower())
    
    # Remove stopwords AND stem the remaining words
    cleaned = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    
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
def get_courses_from_keywords(query, request, num_retrieval=num_retrieval):

    # the docs returned here from the chorma vectorbase with the same format as semantic search
    
    bm25_query = deep_clean_query(query, request.app.state.stemmer)
    doc_scores = request.app.state.bm25_obj.get_scores(bm25_query)

    # Get top N results
    top_n_indices = np.argsort(doc_scores)[::-1][:num_retrieval]
    max_score = np.max(doc_scores)
    min_score = np.min(doc_scores)

   #top_scores = [(doc_scores[i]-min_score)/(max_score-min_score) for i in top_n_indices]
    top_scores = [doc_scores[i] for i in top_n_indices]
    top_docs = [request.app.state.docs[i] for i in top_n_indices]  # from chorma vectorbase instead of bm25_docs
    #top_n = bm25.get_top_n(bm25_query, corpus, n=num_retrieval)
    print(f"keyword matches: {top_scores}")

    return top_docs, top_scores


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

    return output_docs

@timer_decorator
def combine_manual_rrf(question, request):    
    semantic_docs, semantic_scores = get_courses(question, request.app.state.vectordb, num_retrieval=num_retrieval)
    keyword_docs, keyword_scores = get_courses_from_keywords(question, request, num_retrieval=num_retrieval)
    return manual_rrf(semantic_docs, keyword_docs, request.app.state.docs_map)

@app.post("/search_RRF", response_model=List)
def search_from_all_courses(question: str, request: Request):
    return combine_manual_rrf(question, request)

#========================== Reranking =====================================

def load_rerankers():
    #Base model
    base_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print(f"MiniLM Device: {base_model.model.device}")

    # Advanced model
    model_name = 'Qwen/Qwen3-Reranker-0.6B'
    
    # 1. Manually load and fix the tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Pass the fixed tokenizer into the CrossEncoder
    rerank_model_advance = CrossEncoder(
        model_name, 
        max_length=512, 
        tokenizer_args={'pad_token': tokenizer.pad_token}
    )
    
    # 3. Explicitly set it on the underlying model just to be safe
    rerank_model_advance.model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Qwen Device: {rerank_model_advance.model.device}")

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

    pairs = [[question, doc.page_content] for doc in combined_docs]

    scores = request.app.state.base_model.predict(pairs)

    ranked_results = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True) 

    output_docs= [result[1] for result in ranked_results if result[0]> -0.3]

    output_scores = [result[0] for result in ranked_results if result[0]> -0.3]
    print("Base Reranking:")
    print(output_scores)

    return output_docs 

@app.post("/search_reranking_base", response_model=List)
def rerank_from_all_courses(question: str, request: Request):
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

    pairs = [[question, doc.page_content] for doc in combined_docs]

    scores = app.state.advance_model.predict(pairs)

    ranked_results = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True) 

    output_docs= [result[1] for result in ranked_results]

    output_scores = [result[0] for result in ranked_results]
    print("Advanced Reranking:")
    print(output_scores)

    return output_docs 

@app.post("/search_reranking_advance", response_model=List)
def rerank_advance_from_all_courses(question: str, request: Request):
    return combine_advance_reranking(question, request) 
  


