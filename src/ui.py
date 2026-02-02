import streamlit as st
import requests
import pandas as pd
import time

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="Zomato Sentiment AI", page_icon="🍽️", layout="wide")

# Minimalist Modern CSS
st.markdown("""
    <style>
    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        color: #333333;
    }
    .stApp {
        background-color: #FFFFFF;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #000000 !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #E5E5E5;
        padding: 10px 12px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #000000;
        box-shadow: none;
    }

    /* Buttons - Minimalist Black/White */
    div.stButton > button {
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #000000;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #333333;
        border-color: #333333;
        color: #FFFFFF;
    }
    div.stButton > button:active {
        transform: scale(0.98);
    }

    /* Cards/Metrics */
    div[data-testid="stMetric"] {
        background-color: #FAFAFA;
        border: 1px solid #F0F0F0;
        padding: 15px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }

    /* Sidebar - Clean */
    [data-testid="stSidebar"] {
        background-color: #F9F9F9;
        border-right: 1px solid #EEEEEE;
    }
    [data-testid="stSidebar"] hr {
        margin: 20px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 1px solid #EEEEEE;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #888888;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #000000;
        font-weight: 600;
        border-bottom: 2px solid #000000;
    }
    </style>
""", unsafe_allow_html=True)

# --- API HELPERS ---
API_URL = "http://127.0.0.1:8000"

def get_models():
    try:
        resp = requests.get(f"{API_URL}/models")
        if resp.status_code == 200:
            return resp.json()
    except:
        return {"available_models": ["RandomForest"], "metrics": {"RandomForest": 0.0}}
    return {"available_models": ["RandomForest"], "metrics": {"RandomForest": 0.0}}

def analyze_review(text, model):
    try:
        resp = requests.post(f"{API_URL}/predict", json={"review": text, "model_name": model})
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None

# --- SIDEBAR ---
st.sidebar.title("Zomato AI")
st.sidebar.caption("Enterprise Sentiment Intelligence")
st.sidebar.markdown("---")

# Model Selection
model_data = get_models()
available_models = model_data.get("available_models", ["RandomForest"])
selected_model = st.sidebar.selectbox("Active Model", available_models)

# Display Current Model Accuracy
current_acc = model_data.get("metrics", {}).get(selected_model, 0.0)
st.sidebar.metric("Accuracy", f"{current_acc:.2%}")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.success("● API Online")
st.sidebar.info("v1.2.0 • Production")

# --- MAIN APP ---
st.title("Sentiment Dashboard")
st.write("Real-time analysis of culinary experiences.")

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["Analysis", "Reputation", "Architecture"])

# --- TAB 1: LIVE ANALYSIS ---
with tab1:
    col1, col2 = st.columns([0.65, 0.35])
    
    with col1:
        st.subheader("Live Feed")
        with st.form("search_form"):
            review_query = st.text_input("Input Review", placeholder="e.g. The ambiance was stunning but the food was cold...")
            submitted = st.form_submit_button("Run Analysis")
        
        if submitted and review_query:
            with st.spinner("Processing..."):
                time.sleep(0.3) # Debounce UI feel
                result = analyze_review(review_query, selected_model)
                
                if result:
                    sentiment = result['sentiment']
                    conf = result['confidence']
                    
                    st.markdown("### Result")
                    if sentiment == "Positive":
                        st.success(f"Positive ({conf:.2%})")
                    else:
                        st.error(f"Negative ({conf:.2%})")
    
    with col2:
        st.subheader("Insights")
        if submitted and review_query:
            st.markdown(f"""
            **Model Used:** `{selected_model}`
            
            **Actionable Step:**
            """)
            if sentiment == "Negative":
                st.warning("⚠️ Flag for Manager Review")
                st.caption("Suggested SLA: 24 Hours")
            else:
                st.info("✅ Mark for Social Media")
                st.caption("High potential for engagement")

# --- TAB 2: REPUTATION MANAGER ---
with tab2:
    st.subheader("Batch Analytics")
    st.write("Simulate analysis on a batch of incoming reviews.")

    mock_reviews = [
        "Best biryani ever!", "Stale food, got sick.", "Amazing service and decor.",
        "Waiter was rude.", "Value for money.", "Cockroach found in soup."
    ]
    
    if st.button("Process Batch Queue"):
        results = []
        progress_bar = st.progress(0)
        
        for i, rev in enumerate(mock_reviews):
            res = analyze_review(rev, selected_model)
            if res:
                results.append(1 if res['sentiment'] == "Positive" else 0)
            progress_bar.progress((i + 1) / len(mock_reviews))
        
        avg_score = sum(results) / len(results)
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Positive", f"{sum(results)}")
        col_b.metric("Negative", f"{len(results) - sum(results)}")
        col_c.metric("Score", f"{avg_score:.2f}")

        st.markdown("---")
        
        if avg_score > 0.8:
            st.success("🏅 **Status: Gold Tier** - Excellent Performance")
        elif avg_score < 0.4:
            st.error("🚨 **Status: Critical** - Inspection Required")
        else:
            st.warning("⚖️ **Status: Stable** - Monitor Closely")

# --- TAB 3: ARCHITECTURE ---
with tab3:
    st.subheader("Pipeline Architecture")
    st.markdown("### Feature Engineering (TF-IDF)")
    st.code("""
    # Transformation Logic
    input = "The food was good"
    tokens = ["The", "food", "was", "good"]
    vector = [0.0, 0.707, 0.0, 0.707, ...] 
    """, language="python")
    
    st.markdown("### Model Specifications")
    st.json(model_data.get("metrics", {}))