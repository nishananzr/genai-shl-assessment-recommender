from flask import Flask, request, jsonify
import recommender 

app = Flask(__name__)

print("Flask App: Initializing Recommender System...")
recommender.load_data()

if recommender.assessments_data: 
    recommender.initialize_model_and_embeddings()
    if recommender.model is not None and recommender.assessment_embeddings is not None:
        print("Flask App: Recommender System Initialized Successfully.")
    else:
        print("Flask App: ERROR - Model or embeddings failed to initialize in recommender module.")
else:
    print("Flask App: ERROR - No data loaded by recommender module.")



@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    
    if not (recommender.model is not None and 
            recommender.assessment_embeddings is not None and 
            recommender.assessments_data): 
        return jsonify({"error": "Recommender system not ready or data not loaded."}), 503

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request JSON body"}), 400
        
        query_text = data["query"]
        if not isinstance(query_text, str) or not query_text.strip():
            return jsonify({"error": "'query' must be a non-empty string"}), 400

        raw_recommendations = recommender.recommend_assessments(query_text, top_n=10)

        formatted_recommendations = []
        for rec in raw_recommendations:
            api_adaptive_support = "Yes" if rec.get("adaptive_support") == "Yes" else "No"
            api_remote_support = "Yes" if rec.get("remote_support") == "Yes" else "No"
            
            api_duration = rec.get("duration")
            
            if api_duration is None:
                api_duration = 0
            elif not isinstance(api_duration, int):
                try:
                    api_duration = int(api_duration)
                except (ValueError, TypeError):
                    api_duration = 0 

            formatted_recommendations.append({
                "assessment_name": rec.get("assessment_name", "N/A"),
                "url": rec.get("url", "N/A"),
                "adaptive_support": api_adaptive_support,
                "description": rec.get("description", "N/A"), 
                "duration": api_duration,
                "remote_support": api_remote_support,
                "test_type": rec.get("test_type", [])
            })

        return jsonify({"recommended_assessments": formatted_recommendations}), 200

    except Exception as e:
        app.logger.error(f"Error in /recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred processing your request."}), 500

if __name__ == "__main__":
    
    if not (recommender.model is not None and 
            recommender.assessment_embeddings is not None and 
            recommender.assessments_data): 
         print("**************************************************************")
         print("ERROR: Recommender model/data not loaded. API may not work correctly.")
         print("Please check the console output above for initialization errors from the recommender module.")
         print("Ensure recommender.py can load data and models successfully when imported.")
         print("**************************************************************")
    
    app.run(debug=True, host="0.0.0.0", port=5001)