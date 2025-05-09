import json
from sentence_transformers import SentenceTransformer, util
import torch 
import os


DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'assessments_data.json')
MODEL_NAME = 'paraphrase-MiniLM-L3-v2' 

assessments_data = []
assessment_embeddings = None
model = None


def load_data():
    """Loads assessment data from the JSON file."""
    global assessments_data
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            assessments_data = json.load(f)
        print(f"Successfully loaded {len(assessments_data)} assessments from {DATA_FILE_PATH}")
        if not assessments_data:
            print("Warning: No data loaded. assessments_data.json might be empty or incorrectly formatted.")


    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        print("Please ensure 'assessments_data.json' exists in the 'data' folder and is populated.")
        assessments_data = [] 
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATA_FILE_PATH}. Check for syntax errors in the file.")
        assessments_data = [] 
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        assessments_data = []


def initialize_model_and_embeddings():
    """Initializes the sentence transformer model and pre-computes embeddings for all assessments."""
    global model, assessment_embeddings, assessments_data

    if not assessments_data:
        print("No assessment data loaded. Skipping model initialization and embedding generation.")
        return

    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Ensure you have a working internet connection if the model needs to be downloaded.")
        print("Also, check if 'sentence-transformers' and 'torch' are installed correctly in your .venv.")
        return

    
    descriptions = []
    valid_assessments_indices = [] 

    for i, assessment in enumerate(assessments_data):
        desc = assessment.get('description')
        if isinstance(desc, str) and desc.strip():
            descriptions.append(desc)
            valid_assessments_indices.append(i)
        else:
            print(f"Warning: Assessment '{assessment.get('assessment_name', 'Unknown Assessment')}' has missing or invalid description. Skipping.")

    if not descriptions:
        print("No valid descriptions found in the assessment data. Cannot generate embeddings.")
        return

    print(f"Generating embeddings for {len(descriptions)} assessment descriptions...")
    try:
        assessment_embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
        print("Embeddings generated successfully.")

    
        original_assessments_data = assessments_data
        assessments_data = [original_assessments_data[i] for i in valid_assessments_indices]
        print(f"Updated assessments_data to {len(assessments_data)} items after filtering for valid descriptions.")

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        assessment_embeddings = None


def recommend_assessments(query_text, top_n=10):
    """
    Recommends assessments based on semantic similarity to the query_text.
    """
    global model, assessment_embeddings, assessments_data

    if model is None or assessment_embeddings is None:
        print("Error: Model or assessment embeddings not initialized. Cannot make recommendations.")
        print("Please run initialize_model_and_embeddings() first, likely after loading data.")
        return []

    if not assessments_data:
        print("No assessment data available to make recommendations.")
        return []
    
    if not isinstance(query_text, str) or not query_text.strip():
        print("Error: Query text must be a non-empty string.")
        return []

    print(f"\nReceived query: '{query_text}'")
    try:
        query_embedding = model.encode(query_text, convert_to_tensor=True)
    except Exception as e:
        print(f"Error encoding query text: {e}")
        return []


    try:
        cosine_scores = util.cos_sim(query_embedding, assessment_embeddings)[0]
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return []


    top_results = torch.topk(cosine_scores, k=min(top_n, len(assessments_data)))

    recommendations = []
    print("Top recommendations:")
    for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
        assessment_index = idx.item() 
        recommended_assessment = assessments_data[assessment_index]
        
    
        recommendation_output = {
            "assessment_name": recommended_assessment.get("assessment_name", "N/A"),
            "url": recommended_assessment.get("url", "N/A"),
            "description": recommended_assessment.get("description", "N/A"), 
            "remote_support": recommended_assessment.get("remote_support", "Not specified"),
            "adaptive_support": recommended_assessment.get("adaptive_support", "Not specified"),
            "duration": recommended_assessment.get("duration", "N/A"),
            "test_type": recommended_assessment.get("test_type", []),
            "similarity_score": score.item() 
        }
        recommendations.append(recommendation_output)
        print(f"  {i+1}. {recommendation_output['assessment_name']} (Score: {score.item():.4f})")
        
    return recommendations


if __name__ == '__main__':
    print("--- Initializing Recommender System ---")
    load_data()
    if assessments_data: 
        initialize_model_and_embeddings()

        if model and assessment_embeddings is not None:
            print("\n--- Testing Recommendations ---")
            test_queries = [
                "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
                "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
                "Content Writer required, expert in English and SEO."
            ]

            for t_query in test_queries:
                recommended = recommend_assessments(t_query, top_n=3) 
                
                print("-" * 30)
        else:
            print("Skipping recommendation tests as model or embeddings failed to initialize.")
    else:
        print("Skipping initialization and tests as no data was loaded.")