import streamlit as st
import requests

st.set_page_config("Opportunity Recommender", layout="wide")
st.title("🚀 Opportunity Recommendation System")

API_BASE = "http://127.0.0.1:8000"

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Recommendations", "Add Opportunity", "Add User", "Record Interaction"])

if page == "Recommendations":
    st.header("Get Personalized Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        query = st.text_input("🔍 Search Query (optional)", placeholder="e.g., 'Python developer', 'data science'")
        user_id = st.number_input("User ID (optional)", min_value=0, step=1, value=0)
        k = st.slider("Number of Recommendations", 1, 20, 5)
    
    with col2:
        alpha = st.slider("Hybrid Weight (Content ↔ Collaborative)", 0.0, 1.0, 0.5, 
                         help="0.0 = Pure Collaborative, 1.0 = Pure Content-Based")
        include_reasons = st.checkbox("Show Explanation", value=True)
    
    if st.button("Get Recommendations", type="primary"):
        payload = {
            "query": query if query.strip() else None,
            "user_id": int(user_id) if user_id > 0 else None,
            "k": k,
            "alpha": alpha,
            "include_reasons": include_reasons
        }
        
        try:
            resp = requests.post(f"{API_BASE}/recommend", json=payload, timeout=10)
            
            if resp.status_code == 200:
                results = resp.json()["results"]
                if results:
                    st.success(f"Found {len(results)} recommendations")
                    for i, r in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### {i}. {r['title']}")
                            st.write(r['description'])
                            if r.get('score') is not None:
                                st.progress(min(r['score'], 1.0))
                            if r.get('reason') and include_reasons:
                                st.info(f"💡 **Reason:** {r['reason']}")
                            st.divider()
                else:
                    st.warning("No recommendations found. Try adjusting your query or user ID.")
            else:
                st.error(f"API Error: {resp.status_code} - {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Add Opportunity":
    st.header("Add New Opportunity")
    
    with st.form("opportunity_form"):
        opp_id = st.number_input("Opportunity ID", min_value=1, step=1)
        title = st.text_input("Title *", placeholder="e.g., Senior ML Engineer")
        description = st.text_area("Description *", placeholder="Detailed description...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.text_input("Category", placeholder="e.g., Technology")
        with col2:
            location = st.text_input("Location", placeholder="e.g., Remote, New York")
        with col3:
            opp_type = st.selectbox("Type", ["", "job", "internship", "course", "grant", "event", "freelance"])
        
        skills = st.text_input("Skills Required (comma-separated)", placeholder="e.g., Python, TensorFlow, NLP")
        
        submitted = st.form_submit_button("Add Opportunity", type="primary")
        
        if submitted:
            if not title or not description:
                st.error("Title and Description are required")
            else:
                payload = {
                    "opportunity_id": int(opp_id),
                    "title": title,
                    "description": description,
                    "skills_required": [s.strip() for s in skills.split(",") if s.strip()],
                    "category": category,
                    "location": location,
                    "opportunity_type": opp_type
                }
                
                try:
                    resp = requests.post(f"{API_BASE}/opportunities", json=payload)
                    if resp.status_code == 200:
                        st.success(f"✅ Opportunity {opp_id} added! Note: Restart API or call /reload to refresh models.")
                    else:
                        st.error(f"Error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "Add User":
    st.header("Create User Profile")
    
    with st.form("user_form"):
        user_id = st.number_input("User ID", min_value=1, step=1)
        profile_text = st.text_area("Profile Text", placeholder="Describe your background, interests, goals...")
        skills = st.text_input("Skills (comma-separated)", placeholder="e.g., Python, Machine Learning, SQL")
        interests = st.text_input("Interests (comma-separated)", placeholder="e.g., AI, Data Science, Web Development")
        experience = st.text_area("Experience", placeholder="Your work/education experience...")
        
        submitted = st.form_submit_button("Create User", type="primary")
        
        if submitted:
            payload = {
                "user_id": int(user_id),
                "profile_text": profile_text,
                "skills": [s.strip() for s in skills.split(",") if s.strip()],
                "interests": [s.strip() for s in interests.split(",") if s.strip()],
                "experience": experience
            }
            
            try:
                resp = requests.post(f"{API_BASE}/users", json=payload)
                if resp.status_code == 200:
                    st.success(f"✅ User {user_id} created!")
                else:
                    st.error(f"Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "Record Interaction":
    st.header("Record User Interaction")
    
    with st.form("interaction_form"):
        user_id = st.number_input("User ID", min_value=1, step=1)
        opp_id = st.number_input("Opportunity ID", min_value=1, step=1)
        event_type = st.selectbox("Event Type", ["view", "click", "save", "apply", "implicit"])
        weight = st.slider("Weight", 0.1, 5.0, 1.0, 0.1, help="Higher weight = stronger signal")
        
        submitted = st.form_submit_button("Record Interaction", type="primary")
        
        if submitted:
            payload = {
                "user_id": int(user_id),
                "opportunity_id": int(opp_id),
                "event_type": event_type,
                "weight": float(weight)
            }
            
            try:
                resp = requests.post(f"{API_BASE}/interactions", json=payload)
                if resp.status_code == 200:
                    st.success("✅ Interaction recorded! Note: Restart API or call /reload to refresh models.")
                else:
                    st.error(f"Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

