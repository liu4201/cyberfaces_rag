from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import os
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
# from .models import Metadata
# from .db import get_collection

#==========================Semantic Engine=================================

k=3

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
    #print(response)
    if response.status_code == 200:
        print(response.text)
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    return json.loads(response.text)["choices"][0]["message"]["content"]

def get_courses(question, k=k):
    results = vectordb.similarity_search_with_relevance_scores(question,k=k)
    #docs = vectordb.similarity_search(question,k=k)
    #results = vectordb.similarity_search_with_score(question,k=k)
    # for doc in docs:
    #     print(doc.metadata)
    docs = [result[0] for result in results]
    scores = [result[1] for result in results]

    print(scores)

    return docs

app = FastAPI(title="Cyberfaces Smartsearch API")

# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["id"] = record.get("id")
#     metadata["title"] = record.get("title")
#     metadata["created_at"] = record.get("created_at")
#     metadata["updated_at"] = record.get("updated_at")
#     metadata["is_course"] = record.get("is_course")

#     return metadata


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

#print(docs)

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
        #collection_metadata={"hnsw:space": "cosine"}  # use consine distance instead of L2
        collection_metadata={"hnsw:space": "l2"}  # use consine distance instead of L2
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


@app.post("/search", response_model=List)
def search_semantic_from_all_courses(question: str):
    docs =  get_courses(question, k=k)   
    return docs

#==========================Lexical Engine=================================

# Download necessary data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stemmer = PorterStemmer()

def deep_clean_query(query):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(query.lower())
    
    # Remove stopwords AND stem the remaining words
    cleaned = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    
    return cleaned

# 1. Clean the corpus in the same way as query
bm25_docs = [deep_clean_query(doc.page_content) for doc in docs]

# 2. Initialize the BM25 object
bm25 = BM25Okapi(bm25_docs)

# 3. Clean query to get non-stopping keywords and search

def get_courses_from_keywords(query, k=k):
    
    bm25_query = deep_clean_query(query)
    doc_scores = bm25.get_scores(bm25_query)

    # Get top N results
    top_n_indices = np.argsort(doc_scores)[::-1][:k]
    max_score = np.max(doc_scores)
    min_score = np.min(doc_scores)

    top_scores = [(doc_scores[i]-min_score)/(max_score-min_score) for i in top_n_indices]
    top_docs = [docs[i] for i in top_n_indices]
    #top_n = bm25.get_top_n(bm25_query, corpus, n=k)
    print(f"Top matches: {top_scores}")

    return top_docs


@app.post("/search_lexical", response_model=List)
def search_keywords_from_all_courses(question: str):    
    return get_courses_from_keywords(question, k=k)
