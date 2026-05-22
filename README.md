# Smart Search Engine for CyberFaCES platform

Smart Search for Cyberfaces is an advanced search system designed to evaluate a diverse range of information retrieval architectures—from traditional indexing to state-of-the-art AI strategies—and deploy the most optimal, high-performance solution for Cyberfaces.

It employs a two-stage retrieval architecture. The initial fast-retrieval stage combines semantic and keyword search, while the secondary stage refines these results using multiple re-ranker models, including conditional RRF, cross-encoders and LLMs, to score and rank candidate courses.

# Directory Layout

├── analysis/               # Evaluation of different re-rankers 
├── prompts/                # Query Expansion Prompts & Cross Encoder Instructions
├── app/                    # API script
├── data.jsonl              # course description from CyberFaCES database
├── ...
└── course_unit_map.jsonl   # the mapping for unit_to_course logic from CyberFaCES database

# Getting Started 

After cloning the repository and navigating into the project directory, run the command below to start the Docker container on your machine:

Once you have cloned the repository, navigate into the directory and run the command below to start the Docker container on your machine:
 
```docker compose up --build```

When the Docker container is up, you will see:

```
cyberface-rag  | INFO:     Application startup complete.
cyberface-rag  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
Now, open your browser and go to `http://localhost:8000/docs` to get started.

# API Usage

## API endpoints

- `POST /search_semantic`
    - Description: semantic search for user query
    - Parameter: user query
- `POST /search_lexical`
    - Description: keyword search for user query
    - Parameter: user query
- `POST /search_RRF`
    - Description: hybrid search combining semantic and keyword search with conditional Reciprocal Rank Fusion (RRF) Reranker
        - "conditional" means Global variable `keyword_threshold` can be changed in `app/main.py`
    - Parameter: user query
- `POST /search_RRF_all_candidates`
    - Description: hybrid search combining semantic and keyword search with Reciprocal Rank Fusion (RRF) Reranker
    - Parameter: user query
- `POST /search_RRF_unit2course`
    - Description: hybrid search combining semantic and keyword search with conditional Reciprocal Rank Fusion (RRF) and unit_to_course logic
        - "conditional" means Global variable `keyword_threshold` can be changed in `app/main.py
        - "unit_to_course" means for any instructional unit matched, the logic will include its broader parent course.
    - Parameter: user query
- `POST /search_reranking_base`
    - Description: hybrid search combining semantic and keyword search with base Cross-encider Reranker: `ms-marco-MiniLM-L-6-v2`
    - Parameter: user query
    - Note: Query Expansion is on, so AnvilGPT API is required.
- `POST /search_reranking_advance`
    - Description: hybrid search combining semantic and keyword search with advanced Cross-encider Reranker: `jina-reranker-v3`
    - Parameter: user query
    - Note: Query Expansion is on, so AnvilGPT API is required.
- `POST /search_llm_scores`
    - Description: hybrid search combining semantic and keyword search with LLM-based scoring (gpt-oss:120b)
    - Parameter: user query
    - Note: AnvilGPT API is required.
- `POST /evaluate`
    - Description: evaluation with a test dataset comprising 35 queries and their corresponding ground-truth courses: `gemini_generate_dataset_updateByHuman.jsonl`
    - Parameter: 

## Run endpoints

- Find the Endpoint: Browse the list of controllers and click on the specific endpoint you want to use (e.g., POST /evaluate, POST /search_RRF).

- Enable Editing: Click the "Try it out" button located on the right side of the endpoint panel.

- Enter Parameters: Fill in any required fields, such as query parameters "RRF", or user query "Anvil S3 storage".

- Execute: Click the large blue "Execute" button.

- View Response: Scroll down to the Responses section to view the HTTP status code, response headers, and the JSON payload returned by the server.