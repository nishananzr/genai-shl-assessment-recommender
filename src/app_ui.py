import streamlit as st
import recommender 
import time 


st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")


@st.cache_resource 
def load_recommender_system():
    print("UI: Initializing Recommender System for Streamlit App...")
    start_time = time.time()
    recommender.load_data()
    if recommender.assessments_data:
        recommender.initialize_model_and_embeddings()
        if recommender.model is not None and recommender.assessment_embeddings is not None:
            end_time = time.time()
            print(f"UI: Recommender System Initialized Successfully in {end_time - start_time:.2f} seconds.")
            return True 
    end_time = time.time()
    print(f"UI: ERROR - Recommender System FAILED to initialize in {end_time - start_time:.2f} seconds.")
    return False 

initialization_success = load_recommender_system()



st.title("SHL Assessment Recommendation System")
st.markdown("Enter a job description or a query to find relevant SHL assessments.")

if not initialization_success:
    st.error("Failed to initialize the recommendation system. Please check the server logs.")
else:
    
    query = st.text_area("Your Query / Job Description:", height=150, placeholder="e.g., I need a test for a senior java developer with strong problem-solving skills...")


    if st.button("Get Recommendations", type="primary"):
        if query.strip():
            with st.spinner("Finding recommendations..."):
                recommendations_list = recommender.recommend_assessments(query, top_n=10)
            
            if recommendations_list:
                st.subheader("Top Recommendations:")
                for i, rec in enumerate(recommendations_list):
                    st.markdown(f"---")
                    col1, col2 = st.columns([3, 1]) 
                    with col1:
                        st.markdown(f"**{i+1}. {rec.get('assessment_name', 'N/A')}**")
                        st.markdown(f"*{rec.get('description', 'No description available.')}*")
                    with col2:
                        st.markdown(f"[View Assessment]({rec.get('url', '#')})", unsafe_allow_html=True)
                        st.caption(f"Duration: {rec.get('duration', 'N/A')} mins")
                        st.caption(f"Remote: {rec.get('remote_support', 'N/A')}")
                        st.caption(f"Adaptive: {rec.get('adaptive_support', 'N/A')}")
                        if rec.get('test_type'):
                            st.caption(f"Type(s): {', '.join(rec.get('test_type'))}")
        
                st.markdown(f"---")

            else:
                st.info("No recommendations found for your query, or an error occurred.")
        else:
            st.warning("Please enter a query or job description.")

st.sidebar.header("About")
st.sidebar.info(
    "This is a Gen AI project to recommend SHL assessments based on natural language queries."
    "It uses sentence embeddings for semantic search."
)