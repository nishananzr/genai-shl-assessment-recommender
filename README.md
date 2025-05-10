# SHL Assessment Recommendation System

This project implements an intelligent recommendation system to help hiring managers find relevant SHL assessments based on natural language queries or job descriptions. It utilizes sentence embeddings for semantic understanding to provide more relevant results than traditional keyword-based searches.

## 1. Project Overview

Hiring managers often face challenges in efficiently identifying the most suitable assessments for specific job roles from the SHL product catalog. The existing methods relying on keyword searches and filters can be time-consuming. This project aims to simplify this process by building an intelligent system that understands natural language input and recommends relevant SHL assessments.

## 2. Accessibility & URLs

*   **UI Demo (Streamlit Web Application):**
    *   **URL:** `https://genai-shl-assessment-recommender-6yd7lr62t6fhq4cmjwege8.streamlit.app/`
    *   *Functionality:* This web application provides a user-friendly interface to input a query or job description and view the top recommended SHL assessments along with their details.

*   **API Endpoint:**
    *   **Base URL:** `https://36f3-103-148-20-51.ngrok-free.app`
    *   **Health Check Endpoint (GET):** `https://36f3-103-148-20-51.ngrok-free.app/health`
    *   **Recommendation Endpoint (POST):** `https://36f3-103-148-20-51.ngrok-free.app/recommend`
        *   *Request Body Example:* `{"query": "I need a test for java developers"}`
    *   *Functionality:* The API provides programmatic access to the recommendation engine.
    *   **API Endpoint Note:** This API endpoint is hosted via an ngrok tunnel to my local development server. For the API to be accessible, my local computer, the Flask server (`python -m flask run --host=0.0.0.0 --port=5001`), and the ngrok client (`ngrok http 5001`) must all be running. I will strive to keep this active during typical review periods. If the API endpoint is unreachable, please refer to the "How to Run Locally" instructions, or use the fully hosted Streamlit Web App Demo which utilizes the same core recommendation logic.

*   **Code Repository (GitHub):**
    *   **URL:** `https://github.com/nishananzr/genai-shl-assessment-recommender`
    *   *Accessibility:* The complete source code, including data (`assessments_data.json`), core logic, API, UI, and evaluation scripts, is accessible via this public GitHub repository.

## 3. Approach & Implementation

*   **Data Collection & Preparation:**
    *   Assessment data was manually curated from the SHL product catalog website, focusing on assessments relevant to the provided test dataset.
    *   Key details extracted for each assessment include its name, SHL URL, a comprehensive description (often enhanced with relevant job roles for improved semantic matching), remote/adaptive support status, duration (in minutes), and test type(s).
    *   This curated dataset is stored in `data/assessments_data.json`.

*   **Core Recommendation Logic (Gen AI Aspect):**
    *   The system leverages semantic search using sentence embeddings to understand the meaning behind user queries and assessment descriptions.
    *   The `sentence-transformers` library with the pre-trained `paraphrase-MiniLM-L3-v2` model (chosen for a balance of performance and a smaller memory footprint suitable for demonstration) is used to convert both the assessment descriptions and the user's input query into high-dimensional vector embeddings.
    *   Cosine similarity is then calculated between the query embedding and all pre-computed assessment embeddings.
    *   The top N (up to 10) assessments with the highest cosine similarity scores are returned as recommendations. This logic is encapsulated in `src/recommender.py`.

*   **API Development:**
    *   A RESTful API was developed using Flask to serve the recommendations.
    *   It includes a `/health` GET endpoint for status checks.
    *   The core `/recommend` POST endpoint accepts a JSON payload with a "query" field and returns a JSON list of recommended assessments, formatted according to the project specifications. This is implemented in `src/api.py`.

*   **User Interface (UI):**
    *   An interactive and user-friendly web application was built using Streamlit.
    *   It provides a text input area for the user's query/job description and a button to trigger the recommendation process.
    *   Results are displayed , including assessment names, descriptions, durations, support details, test types, and direct links to the SHL assessment pages. This is implemented in `src/app_ui.py`.

## 4. Evaluation

*   **Evaluation Strategy:**
    *   The system's performance was measured against the "SHL Assessment Recommendations" test dataset provided in the problem statement PDF. This dataset contains sample queries and their corresponding ground truth relevant assessments.
    *   For each test query, the system generated its top recommendations (specifically, top 10 were fetched to evaluate metrics at K=3). These predicted recommendations (identified by their URLs) were then compared against the ground truth relevant assessment URLs.

*   **Metrics Used:**
    *   As specified, the system was evaluated using Mean Recall@3 (MR@3) and Mean Average Precision@3 (MAP@3).
    *   **Recall@K:** Measures the proportion of actual relevant items that are retrieved in the top K recommendations for a query.
    *   **Average Precision@K (AP@K):** Considers both relevance and the ranking order of retrieved items within the top K results for a query.
    *   **MR@3 and MAP@3:** The Recall@3 and AP@3 values were calculated for each query in the test set, and then averaged across all queries to obtain the final MR@3 and MAP@3 scores.

*   **Implementation of Metrics:**
    *   The evaluation logic is implemented in `src/evaluation.py`. This script:
        1.  Defines the `GROUND_TRUTH_DATA` by parsing the queries and relevant assessment URLs from the project PDF's test set.
        2.  Initializes the recommender system by loading data and the sentence embedding model (from `recommender.py`).
        3.  For each test query, it programmatically calls the `recommender.recommend_assessments` function.
        4.  Calculates `Recall@3` and `AP@3` by comparing the URLs of predicted assessments against the ground truth URLs.
        5.  Averages these scores across all test queries.

*   **Achieved Evaluation Score:**
    *   **Mean Recall@3 (MR@3): 0.1184**
    *   **Mean Average Precision@3 (MAP@3): 0.2698**

## 5. Optimization Efforts & Challenges

*   **Description Enhancement:** Assessment descriptions in `assessments_data.json` were manually augmented with "Relevant job roles" where contextually appropriate, aiming to improve the semantic richness for the embedding model.
*   **Model Selection for Deployment:** Initially, `all-MiniLM-L6-v2` was considered, but due to memory constraints on free-tier hosting platforms (like Render, which resulted in "Out of Memory" errors), the smaller `paraphrase-MiniLM-L3-v2` model was chosen for deployed versions to increase the chances of successful hosting, while acknowledging a potential trade-off in raw embedding quality. The evaluation scores reflect the performance with this model.
*   **API Hosting Challenge:** Persistently hosting the Flask API with its ML model dependencies on a completely free, always-on tier proved challenging due to memory limits. The ngrok solution is used as a fallback for API endpoint accessibility during review.
*   **Time Constraints:** The project was developed within a condensed timeframe, focusing on core functionality and meeting primary requirements.

*Potential Future Optimizations:*
*   Experimentation with larger or domain-specific fine-tuned sentence embedding models.
*   Implementation of a more robust data pipeline (e.g., full catalog scraping).
*   Deployment of the API to a more scalable cloud solution (e.g., Google Cloud Run with a vector database for embeddings) if resource constraints were lifted.
*   Adding more sophisticated filtering based on duration or other metadata post-semantic search.

## 6. Tools and Libraries

*   **Python 3.10+**
*   **Sentence-Transformers:** For generating text embeddings (using `paraphrase-MiniLM-L3-v2` model).
*   **PyTorch:** As a backend for sentence-transformers.
*   **Flask:** For building the REST API.
*   **Streamlit:** For creating the interactive web UI.
*   **NumPy:** For numerical operations in the evaluation script.
*   **Gunicorn:** WSGI server (used in `Procfile` for Render attempts, and in `Dockerfile` for Cloud Run attempts).
*   **Git & GitHub:** For version control and code hosting.
*   **Ngrok:** For tunneling the local API endpoint to a public URL.
*   **Postman:** For API testing.

## 7. How to Run the Project Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nishananzr/genai-shl-assessment-recommender
    cd genai-shl-assessment-recommender
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```
    *   Windows (PowerShell): `& .\.venv\Scripts\Activate.ps1`
    *   macOS/Linux: `source .venv/bin/activate`
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **To run the API server:**
    *   Set the `FLASK_APP` environment variable:
        *   PowerShell: `$env:FLASK_APP = "src.api"`
        *   CMD: `set FLASK_APP=src.api`
        *   Bash: `export FLASK_APP=src.api`
    *   Run Flask:
        ```bash
        python -m flask run --host=0.0.0.0 --port=5001
        ```
    *   The API will be available at `http://localhost:5001`.

5.  **To run the Streamlit Web UI (in a separate terminal, with venv active):**
    ```bash
    streamlit run src/app_ui.py
    ```
    *   The UI will be available at `http://localhost:8501`.

6.  **To run the evaluation script (in a separate terminal, with venv active):**
    ```bash
    python src/evaluation.py
    ```