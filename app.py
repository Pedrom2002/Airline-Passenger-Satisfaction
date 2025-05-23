import streamlit as st
import pandas as pd
import joblib
import json
import time
import plotly.express as px
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Configure large dataframe styling
pd.set_option("styler.render.max_elements", 2_000_000)

# Page configuration
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #F5F5F5;
    }
    .header {
        color: #1E3D6B;
        font-size: 2.5em;
        padding: 20px;
    }
    .stProgress > div > div > div {
        background-color: #1E3D6B;
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .download-btn {
        background-color: #1E3D6B !important;
        color: white !important;
        border: none;
    }
    .pagination-controls {
        margin: 20px 0;
        display: flex;
        justify-content: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Resources ---
@st.cache_resource
def load_resources():
    resources = {
        'model': joblib.load("models/xgb_final_model.pkl"),
        'preprocessor': joblib.load("models/preprocessor.pkl"),
        'label_encoder': joblib.load("models/label_encoder.pkl"),
        'features': json.load(open("models/feature_columns.json", 'r'))
    }
    return resources

resources = load_resources()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1047/1047711.png", width=100)
    st.title("Settings")
    st.markdown("---")
    st.info("""
    ### Instructions:
    1. Upload a CSV file
    2. Data will be automatically processed
    3. View results and download predictions
    """)

# --- Main Interface ---
st.markdown('<div class="header">Airline Passenger Satisfaction Prediction</div>', unsafe_allow_html=True)

# File Upload Section
upload_col, stats_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader("", type="csv", key="file_uploader")

if uploaded_file:
    # Data Processing
    with st.spinner('Processing data...'):
        start_time = time.time()
        
        try:
            # Read and prepare data
            df = pd.read_csv(uploaded_file)
            raw_df = df.copy()
            
            # Preprocessing
            df = df.drop(columns=['satisfaction', 'Unnamed: 0', 'id'], errors='ignore')
            df_processed = resources['preprocessor'].transform(df)
            df_processed = pd.DataFrame(df_processed, columns=resources['features'])
            
            # Make predictions
            preds = resources['model'].predict(df_processed)
            probas = resources['model'].predict_proba(df_processed)[:, 1]
            preds_decoded = resources['label_encoder'].inverse_transform(preds)
            
            processing_time = time.time() - start_time
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.stop()

    # Results Section
    st.success(f"Processing completed in {processing_time:.2f} seconds!")
    
    # Key Metrics
    avg_proba = probas.mean()
    satisfaction_rate = (preds == 1).mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Passengers Analyzed", len(df))
    with col2:
        st.metric("Satisfaction Rate", f"{satisfaction_rate:.1%}")
    with col3:
        st.metric("Average Confidence", f"{avg_proba:.1%}")

    # Se o CSV tiver a coluna 'satisfaction', calcular métricas adicionais
    if 'satisfaction' in raw_df.columns:
        y_true = raw_df['satisfaction']

        # Classification report
        report = classification_report(y_true, preds_decoded, output_dict=True)
        st.markdown("### Classification Report")
        st.dataframe(pd.DataFrame(report).T)

        # Confusion matrix
        cm = confusion_matrix(y_true, preds_decoded, labels=resources['label_encoder'].classes_)
        fig_cm = px.imshow(
            cm, 
            x=resources['label_encoder'].classes_,
            y=resources['label_encoder'].classes_,
            color_continuous_scale='Blues',
            text_auto=True,
            title='Confusion Matrix'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Accuracy & ROC AUC
        accuracy = accuracy_score(y_true, preds_decoded)
        try:
            # Map classes para binário 0 e 1 para calcular ROC AUC
            class_map = {cls: i for i, cls in enumerate(resources['label_encoder'].classes_)}
            y_true_bin = y_true.map(class_map)
            roc_auc = roc_auc_score(y_true_bin, probas)
        except Exception:
            roc_auc = None

        st.markdown(f"**Accuracy:** {accuracy:.4f}")
        if roc_auc is not None:
            st.markdown(f"**ROC AUC:** {roc_auc:.4f}")

    # Distribution of probabilities
    st.markdown("### Distribution of Prediction Probabilities")
    fig_prob = px.histogram(probas, nbins=50, title="Histogram of Positive Class Probabilities")
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Visualization Tabs
    tab1, tab2, tab3 = st.tabs(["Distribution", "Details", "Raw Data"])
    
    with tab1:
        fig = px.pie(
            names=resources['label_encoder'].classes_,
            values=np.bincount(preds),
            title="Satisfaction Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        for i in range(min(3, len(raw_df))):
            with st.expander(f"Passenger Details #{i+1}", expanded=(i==0)):
                cols = st.columns([2,1])
                with cols[0]:
                    def safe_format(x):
                        try:
                            return f"{float(x):.2f}" if isinstance(x, (int, float)) else str(x)
                        except:
                            return str(x)
                    
                    styled_df = raw_df.iloc[[i]].T.style.format(safe_format)
                    st.dataframe(styled_df)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>Prediction:</h4>
                        <h2>{preds_decoded[i]}</h2>
                        <h4>Probability:</h4>
                        <h2>{probas[i]:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        # Pagination controls
        PAGE_SIZE = 1000
        total_pages = max(1, (len(raw_df) - 1) // PAGE_SIZE + 1)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            page_number = st.number_input(
                "Page Number", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                key="page_selector"
            )
        
        start_idx = (page_number - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        
        # Conditional formatting
        numeric_cols = raw_df.select_dtypes(include=np.number).columns
        non_numeric_cols = raw_df.select_dtypes(exclude=np.number).columns
        
        st.dataframe(
            raw_df.iloc[start_idx:end_idx].style
                .format("{:.2f}", subset=numeric_cols)
                .format("{}", subset=non_numeric_cols),
            height=400,
            use_container_width=True
        )

    # Download Section
    result_df = raw_df.copy()
    result_df['Prediction'] = preds_decoded
    result_df['Probability'] = probas
    
    csv = result_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Full Results",
        data=csv,
        file_name='satisfaction_predictions.csv',
        mime='text/csv',
        key='download_csv',
        help="Click to download all results in CSV format",
        use_container_width=True
    )

else:
    # Initial Screen
    st.markdown("""
    <div style="text-align: center; margin-top: 100px;">
        <h3>Welcome to Passenger Satisfaction Prediction System</h3>
        <p>Start by uploading a CSV file using the uploader above</p>
        <p>🔼 Click the upload button at the top</p>
    </div>
    """, unsafe_allow_html=True)
