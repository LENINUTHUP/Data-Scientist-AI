import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
import time
import tempfile
from datetime import datetime

# --- Time Series Model Imports ---
# Note: You may need to install these libraries: pip install statsmodels prophet fpdf
try:
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    from prophet.plot import plot_components as prophet_plot_components
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- Advanced Model Imports ---
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# --- Page Configuration ---
st.set_page_config(
    page_title="Data Scientist AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom PDF Class with Footer ---
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        # Generation date
        self.set_x(10)
        self.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'L')


# --- Custom CSS for Black Theme ---
def apply_custom_theme():
    """Applies a dark, clean theme to the Streamlit app."""
    st.markdown("""
    <style>
        /* Main background color */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }

        /* Sidebar styling */
        .st-emotion-cache-16txtl3 {
            background-color: #1F222B;
        }
        
        /* Text color in sidebar */
        .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16txtl3 .st-emotion-cache-1v0mbdj p {
            color: #FAFAFA;
        }

        /* Button styling */
        .stButton>button {
            border: 2px solid #8A2BE2; /* Purple */
            background-color: #8A2BE2; /* Purple */
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 0.5rem;
            transition-duration: 0.4s;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #6B21A8; /* Darker Purple */
            border-color: #6B21A8;
        }
        
        /* Header and subheader styling */
        h1, h2, h3 {
            color: #8A2BE2; /* Purple accent color */
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #8A2BE2; /* Purple */
            border-radius: 8px;
        }

        /* Custom Tab Styling */
        button[data-baseweb="tab"] {
            font-size: 16px !important;
            padding-top: 12px !important;
            padding-bottom: 12px !important;
        }
        button[data-baseweb="tab"] > div {
            gap: 8px !important;
        }

    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def get_column_types(df):
    """Identifies and categorizes columns in a DataFrame."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    
    categorical_cols = [col for col in categorical_cols if col not in date_cols]
    
    id_cols = []
    for col in numerical_cols:
        if df[col].nunique() == len(df):
            id_cols.append(col)
    
    numerical_cols = [col for col in numerical_cols if col not in id_cols]
    return numerical_cols, categorical_cols, date_cols, id_cols

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes all required keys in Streamlit's session state."""
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    if 'task' not in st.session_state:
        st.session_state.task = None
    
    if 'df' not in st.session_state: st.session_state.df = None
    if 'processed_df' not in st.session_state: st.session_state.processed_df = None
    if 'model' not in st.session_state: st.session_state.model = None
    if 'forecast_df' not in st.session_state: st.session_state.forecast_df = None

    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {}
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = {}
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = {}
    if 'showdown_results' not in st.session_state:
        st.session_state.showdown_results = None
    if 'date_converted' not in st.session_state:
        st.session_state.date_converted = False
    if 'report_data' not in st.session_state:
        st.session_state.report_data = {}


# --- Sidebar Navigation ---
def render_sidebar():
    """Renders the sidebar with progress indicators based on the selected task."""
    
    # --- Icon Definitions (Base64 Encoded SVGs) ---
    chart_svg_base64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM4QTJCRTIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48bGluZSB4MT0iMTIiIHkxPSIyMCIgeDI9IjEyIiB5Mj0iMTAiPjwvbGluZT48bGluZSB4MT0iMTgiIHkxPSIyMCIgeDI9IjE4IiB5Mj0iNCI+PC9saW5lPjxsaW5lIHgxPSI2IiB5MT0iMjAiIHgyPSI2IiB5Mj0iMTYiPjwvbGluZT48L3N2Zz4="
    check_svg_base64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YWRlODAiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMjIgMTEuMDhWMTJhMTAgMTAgMCAxIDEtNS45My05LjE0Ii8+PHBvbHlsaW5lIHBvaW50cz0iMjIgNCAxMiAxNC4wMSA5IDExLjAxIi8+PC9zdmc+"
    cog_svg_base64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM4QTJCRTIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0iZmVhdGhlciBmZWF0aGVyLXNldHRpbmdzIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIzIj48L2NpcmNsZT48cGF0aCBkPSJNMTkuNCAxNWExLjY1IDEuNjUgMCAwIDAgLjMzIDEuODJsLjA2LjA2YTIgMiAwIDAgMSAwIDIuODMgMiAyIDAgMCAxLTIuODMgMGwtLjA2LS4wNmExLjY1IDEuNjUgMCAwIDAtMS44Mi0uMzMgMS42NSAxLjY1IDAgMCAwLTEgMS41MVYyMWExIDIgMCAwIDEtMiAyIDIgMiAwIDAgMS0yLTJ2LS4wOUExLjY1IDEuNjUgMCAwIDAgOSAxOS40YTEuNjUgMS42NSAwIDAgMC0xLjgyLjMzbC0uMDYuMDZhMiAyIDAgMCAxLTIuODMgMCAyIDIgMCAwIDEgMC0yLjgzbC4wNi0uMDZhMS42NSAxLjY1IDEuNjUgMCAwIDAgLjMzLTEuODIgMS42NSAxLjY1IDAgMCAwLTEuNTEtMUgzYTIgMiAwIDAgMS0yLTIgMiAyIDAgMCAxIDItMmguMDlBMS42NSAxLjY1IDAgMCAwIDQuNiA5YTEuNjUgMS42NSAwIDAgMC0uMzMtMS44MmwtLjA2LS4wNmEyIDIgMCAwIDEgMC0yLjgzIDIgMiAwIDAgMSAyLjgzIDBsLjA2LjA2YTEuNjUgMS42NSAwIDAgMCAxLjgyLjMzSDlhMS42NSAxLjY1IDAgMCAwIDEtMS41MVYzYTIgMiAwIDAgMSAyLTIgMiAyIDAgMCAxIDIgMnYuMDlhMS42NSAxLjY1IDAgMCAwIDEgMS41MSAxLjY1IDEuNjUgMCAwIDAgMS44Mi0uMzNsLjA2LS4wNmEyIDIgMCAwIDEgMi44MyAwIDIgMiAwIDAgMSAwIDIuODNsLS4wNi4wNmExLjY1IDEuNjUgMCAwIDAtLjMzIDEuODJWOWExLjY1IDEuNjUgMCAwIDAgMS41MSAxSDIxYTIgMiAwIDAgMSAyIDIgMiAyIDAgMCAxLTIgMmgtLjA5YTEuNjUgMS42NSAwIDAgMC0xLjUxIDF6Ij48L3BhdGg+PC9zdmc+"
    square_svg_base64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM5NGExYjgiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB4PSIzIiB5PSIzIiB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHJ4PSIyIiByeT0iMiIvPjwvc3ZnPg=="

    title_html = f'<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;"><img src="{chart_svg_base64}" width="24" height="24"> <span style="font-size: 1.35rem; font-weight: bold;">Pipeline Progress</span></div>'
    st.sidebar.write(title_html, unsafe_allow_html=True)
    
    task = st.session_state.get('task')
    
    if not task:
        steps = ['Upload', 'Task Selection']
    elif task == "Predictive Modeling":
        algorithm = st.session_state.get('user_selections', {}).get('algorithm')
        if algorithm in ["Prophet", "ARIMA"]:
            steps = ['Upload', 'Task Selection', 'Preprocessing', 'EDA', 'Feature Engineering', 'Model Recommendation', 'Modeling', 'Evaluation']
        else:
            steps = ['Upload', 'Task Selection', 'Preprocessing', 'EDA', 'Feature Engineering', 'Model Recommendation', 'Modeling', 'Evaluation', 'Action & Export']
    elif task == "Clustering Analysis":
        steps = ['Upload', 'Task Selection', 'Preprocessing', 'EDA', 'Clustering', 'Cluster Analysis', 'Action & Export']
    else:
        steps = ['Upload']

    page_map = {
        'upload': 0, 'task_hub': 1, 'preprocessing': 2, 'eda': 3,
        'feature_engineering': 4, 'model_recommendation': 5, 'modeling': 6, 'evaluation': 7,
        'clustering': 4, 'cluster_analysis': 5, 'action_export': 8
    }
    
    if st.session_state.page == 'action_export':
         current_step_index = len(steps) - 1
    else:
        current_step_index = page_map.get(st.session_state.page, 0)

    for i, step in enumerate(steps):
        if i < current_step_index:
            icon_html = f'<img src="{check_svg_base64}" width="16" height="16" style="display:inline-block; vertical-align:middle; margin-right:8px;">'
            st.sidebar.write(f'{icon_html} <span style="color:#4ade80; font-size: 16px;">{step}</span>', unsafe_allow_html=True)
        elif i == current_step_index:
            icon_html = f'<img src="{cog_svg_base64}" width="16" height="16" style="display:inline-block; vertical-align:middle; margin-right:8px;">'
            st.sidebar.write(f'{icon_html} <strong style="color:white; font-size: 16px;">{step}</strong>', unsafe_allow_html=True)
        else:
            icon_html = f'<img src="{square_svg_base64}" width="16" height="16" style="display:inline-block; vertical-align:middle; margin-right:8px;">'
            st.sidebar.write(f'{icon_html} <span style="color:#94a3b8; font-size: 16px;">{step}</span>', unsafe_allow_html=True)
    
    st.sidebar.markdown("<hr style='margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)

    if st.sidebar.button("Start Over", key="start_over_sidebar"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Page Rendering Functions ---

def render_upload_page():
    st.markdown("""
        <style>
            .stApp > header {
                background-color: transparent;
            }
            .upload-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                height: 100vh;
                padding-top: 5rem;
            }
            .title {
                font-size: 3rem; 
                font-weight: 700; 
                color: #E2E8F0; 
                letter-spacing: 0.1em;
                margin-bottom: 4rem;
            }
            .upload-section {
                width: 100%;
                max-width: 896px; 
            }
            .upload-header {
                font-size: 2rem; 
                font-weight: bold;
                color: #8A2BE2;
                margin-bottom: 2rem;
            }
            .stFileUploader > div > div {
                border: 2px dashed #4A5568;
                background-color: rgba(45, 55, 72, 0.5);
                padding: 3rem; 
                border-radius: 0.75rem;
            }
            .stFileUploader > div > div > button {
                background-color: #8A2BE2;
                color: white;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
            }
            .stFileUploader > div > div > button:hover {
                background-color: #6B21A8;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title" style="text-align: center;">DATA SCIENTIST AI</div>', unsafe_allow_html=True)
    
    _ , col2, _ = st.columns([1, 3, 1]) 
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<div class="upload-header">Upload Data</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag and drop file here", 
            type="csv",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Read Instructions & Conditions"):
            st.markdown("""
            ### How to Use the AI Data Science Pipeline

            Welcome! This guide will walk you through the steps to analyze your data using this tool. Please read the following conditions to ensure the best results and avoid errors.

            **1. Upload Your Data**
            - **Format:** Your data must be in a **`.csv`** file.
            - **Structure:** The file should contain structured data with clear column headers.
            - **Date Columns (Crucial for Forecasting):** If your data includes a date column for time-series analysis, please ensure it is in a standard date format (e.g., `YYYY-MM-DD`, `MM/DD/YYYY`, `DD-MM-YYYY`). The application will attempt to convert it automatically, but a consistent format is best.

            **2. Select Your Task**
            - After uploading, you will be asked to choose a primary task.
            - **Data Suitability:** Ensure your data is appropriate for the selected task to avoid errors.
                - **Predictive Modeling:** Choose this if your goal is to forecast a future value (like sales) or classify an outcome (like customer churn). **Your data must have a clear target column that you want to predict.**
                - **Clustering Analysis:** Choose this if you want to discover natural groupings or segments in your data (e.g., identify different customer types). **This task works best when your dataset has several numerical features.**

            **3. Preprocess Your Data**
            - This is a critical step to clean your data for the model.
            - **Missing Values:** Choose a method to fill any empty cells (e.g., using the mean of the column).
            - **Scaling:** For numerical columns, you can scale the data to a standard range, which often improves model performance.
            - **Encoding:** Text-based columns will be automatically converted into a numerical format that models can understand.

            **4. Explore Your Data (EDA)**
            - Visualize your data to find patterns and insights before modeling.
            - **Univariate Analysis:** View the distribution of a single column.
            - **Bivariate Analysis:** Explore the relationship between two columns.
            - **Correlation Heatmap:** See how all numerical features relate to each other.

            **5. Modeling and Evaluation**
            - **Model Selection:** The application will recommend a suitable model, but you are free to choose another from the list.
            - **Training:** Click "Train Model" to start the automated training process.
            - **Evaluation:** Once trained, you will see a detailed evaluation of the model's performance, including charts and key metrics. For time-series forecasts, this will include trend and seasonality components.

            **6. Download Your Report**
            - On the final "Evaluation" page, navigate to the **"Automated Report"** tab.
            - Here you can download:
                - The processed data as a CSV file.
                - The trained model as a `.pkl` file for future use.
                - A comprehensive PDF report summarizing the entire analysis with all the charts you generated.
            """)

    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.page = 'task_hub'
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

def render_task_hub():
    st.header("Stage 2: Task Selection")
    st.write("Your data is loaded. What would you like to do next?")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predictive Modeling")
        st.info("Forecast future values, classify outcomes, or predict numerical targets.")
        if st.button("Start Predictive Modeling"):
            st.session_state.task = "Predictive Modeling"
            df = st.session_state.df.copy()
            # Automatic date conversion runs only once here
            for col in df.columns:
                if col.lower() == 'date':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                        st.toast(f"Automatically converted column '{col}' to datetime.", icon="📅")
                    except Exception:
                        pass
            st.session_state.processed_df = df
            st.session_state.page = 'preprocessing'
            st.rerun()
    with col2:
        st.subheader("Clustering Analysis")
        st.info("Automatically discover hidden groups or segments in your data.")
        if st.button("Start Clustering Analysis"):
            st.session_state.task = "Clustering Analysis"
            st.session_state.processed_df = st.session_state.df.copy()
            st.session_state.page = 'preprocessing'
            st.rerun()

    st.markdown("---")
    if st.button("Go Back to Upload"):
        st.session_state.page = 'upload'
        st.session_state.df = None # Allow re-upload
        st.rerun()

def render_preprocessing_page():
    st.header("Configure Cleaning & Feature Engineering")
    df = st.session_state.processed_df.copy()

    # --- NEW: Data Quality Section ---
    st.subheader("Data Quality Check")
    col1, col2 = st.columns(2)
    with col1:
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            st.write("Columns with Missing Values:")
            st.dataframe(missing_values.to_frame(name='Missing Count'))
        else:
            st.success("No missing values found.")
    
    with col2:
        num_duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows Found", num_duplicates)
        if num_duplicates > 0:
            primary_key_cols = st.multiselect("Select Primary Key column(s) to remove duplicates:", df.columns)
        else:
            primary_key_cols = None

    st.markdown("---")

    # Initialize config if not present or columns have changed
    if not st.session_state.preprocessing_config or set(st.session_state.preprocessing_config.keys()) != set(df.columns):
        config = {}
        numerical_cols, categorical_cols, date_cols, _ = get_column_types(df)
        for col in df.columns:
            col_type = 'Date' if col in date_cols else ('Numerical' if col in numerical_cols else 'Categorical')
            config[col] = {
                'type': col_type,
                'missing': 'None',
                'scaling': 'None' if col_type == 'Numerical' else 'N/A',
                'remove': False
            }
        st.session_state.preprocessing_config = config
    
    config = st.session_state.preprocessing_config
    
    header_cols = st.columns([3, 2, 2, 2, 1])
    headers = ["COLUMN", "TYPE", "MISSING", "SCALING", "REMOVE"]
    for col, header in zip(header_cols, headers):
        col.write(f"**{header}**")

    st.markdown("---")

    for col_name in df.columns:
        row_cols = st.columns([3, 2, 2, 2, 1])
        row_cols[0].write(col_name)
        row_cols[1].info(config[col_name]['type'])
        
        missing_options = ["None", "Mean", "Median"] if config[col_name]['type'] == 'Numerical' else ["None", "Mode"]
        if config[col_name]['type'] == 'Date': missing_options = ["None"]
        config[col_name]['missing'] = row_cols[2].selectbox("Missing", missing_options, key=f"missing_{col_name}", label_visibility="collapsed")
        
        scaling_options = ["None", "StandardScaler", "MinMaxScaler"]
        if config[col_name]['type'] == 'Numerical':
            config[col_name]['scaling'] = row_cols[3].selectbox("Scaling", scaling_options, key=f"scaling_{col_name}", label_visibility="collapsed")
        else:
            row_cols[3].write("N/A")
            
        config[col_name]['remove'] = row_cols[4].checkbox("", key=f"remove_{col_name}", value=config[col_name]['remove'], label_visibility="collapsed")

    st.markdown("---")
    st.subheader("Global Encoding Strategy")
    encoding_strategy = st.selectbox("Choose how to handle all categorical columns:", ["Label Encoding", "One-Hot Encoding"])

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'task_hub'
            st.rerun()
    with col2:
        if st.button("Apply Preprocessing & Proceed"):
            with st.spinner("Processing..."):
                proc_df = df.copy()
                
                if primary_key_cols:
                    original_rows = len(proc_df)
                    proc_df.drop_duplicates(subset=primary_key_cols, inplace=True)
                    st.toast(f"Removed {original_rows - len(proc_df)} duplicates.")

                cols_to_remove = [col for col, settings in config.items() if settings['remove']]
                proc_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

                for col, settings in config.items():
                    if col in proc_df.columns:
                        if settings['missing'] != 'None':
                            if settings['type'] == 'Numerical':
                                if settings['missing'] == 'Mean': proc_df[col].fillna(proc_df[col].mean(), inplace=True)
                                elif settings['missing'] == 'Median': proc_df[col].fillna(proc_df[col].median(), inplace=True)
                            elif settings['type'] == 'Categorical':
                                 if settings['missing'] == 'Mode': proc_df[col].fillna(proc_df[col].mode()[0], inplace=True)
                
                st.session_state.original_processed_df = proc_df.copy()

                categorical_cols_to_encode = [col for col, settings in config.items() if col in proc_df.columns and settings['type'] == 'Categorical']
                if encoding_strategy == 'Label Encoding':
                    for col in categorical_cols_to_encode:
                        proc_df[col] = LabelEncoder().fit_transform(proc_df[col].astype(str))
                elif encoding_strategy == 'One-Hot Encoding':
                    proc_df = pd.get_dummies(proc_df, columns=categorical_cols_to_encode)

                for col, settings in config.items():
                     if col in proc_df.columns and settings['type'] == 'Numerical' and settings['scaling'] != 'None':
                        scaler = StandardScaler() if settings['scaling'] == 'StandardScaler' else MinMaxScaler()
                        proc_df[[col]] = scaler.fit_transform(proc_df[[col]])
                
                proc_df.dropna(inplace=True)
                st.session_state.processed_df = proc_df
                st.session_state.page = 'eda'
                st.rerun()

def render_eda_page():
    st.header("Exploratory Data Analysis (EDA)")
    df = st.session_state.original_processed_df.copy()
    report_data = st.session_state.get('report_data', {})
    
    st.subheader("1. Univariate Analysis")
    col1_uni, col2_uni = st.columns([1,2])
    with col1_uni:
        feature_uni = st.selectbox("Select a feature to visualize its distribution:", df.columns)
    with col2_uni:
        is_numeric_uni = pd.api.types.is_numeric_dtype(df[feature_uni].dtype)
        if is_numeric_uni:
            fig = px.histogram(df, x=feature_uni, title=f"Distribution of {feature_uni}", template='plotly_dark')
        else:
            fig = px.bar(df[feature_uni].value_counts(), title=f"Count of {feature_uni}", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        report_data['univariate_data'] = (df[feature_uni], is_numeric_uni)

    st.markdown("---")
    st.subheader("2. Bivariate Analysis")
    col1_bi, col2_bi = st.columns(2)
    with col1_bi:
        x_axis = st.selectbox("Select X-Axis Feature:", df.columns, key="x_axis")
    with col2_bi:
        y_axis = st.selectbox("Select Y-Axis Feature:", df.columns, key="y_axis")
    
    x_is_num = pd.api.types.is_numeric_dtype(df[x_axis].dtype)
    y_is_num = pd.api.types.is_numeric_dtype(df[y_axis].dtype)
    
    if x_axis != y_axis:
        if x_is_num and y_is_num:
            fig_bi = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs. {y_axis}", template='plotly_dark')
        elif (x_is_num and not y_is_num):
            fig_bi = px.box(df, x=y_axis, y=x_axis, title=f"{y_axis} vs. {x_axis}", template='plotly_dark')
        elif (not x_is_num and y_is_num):
            fig_bi = px.box(df, x=x_axis, y=y_axis, title=f"{x_axis} vs. {y_axis}", template='plotly_dark')
        else:
            pivot = df.groupby([x_axis, y_axis]).size().reset_index(name='count')
            fig_bi = px.bar(pivot, x=x_axis, y='count', color=y_axis, barmode='group', title=f"{x_axis} vs. {y_axis}", template='plotly_dark')
        st.plotly_chart(fig_bi, use_container_width=True)
        report_data['bivariate_data'] = (df, x_axis, y_axis)

    st.markdown("---")
    st.subheader("3. Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix of Numerical Features", template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)
        report_data['correlation_data'] = corr
    
    st.session_state.report_data = report_data
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'preprocessing'
            st.rerun()
    with col2:
        if st.button("Proceed"):
            if st.session_state.task == "Predictive Modeling":
                st.session_state.page = 'feature_engineering'
            else:
                st.session_state.page = 'clustering'
            st.rerun()

def render_feature_engineering_page():
    st.header("Feature Engineering (for Predictive Modeling)")
    df = st.session_state.processed_df
    st.write("#### Current Data Preview"); st.dataframe(df.head())
    st.markdown("---")
    
    _, categorical_cols, date_cols, _ = get_column_types(st.session_state.original_processed_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time-Series Features")
        if date_cols:
            time_index = st.selectbox("Select Time Index Column:", ["None"] + date_cols)
            extract_parts = st.multiselect("Extract Date Parts:", ['Year', 'Quarter', 'Month', 'Week', 'Day', 'Dayofweek']) if time_index != "None" else []
            segment_by = st.selectbox("Segment Forecast By (Optional):", ["None"] + categorical_cols)
        else:
            st.info("No date columns were detected in the preprocessed data.")
            time_index, extract_parts, segment_by = "None", [], "None"
    with col2:
        st.subheader("Lag & Rolling Features")
        num_cols = [c for c in df.select_dtypes(include=np.number).columns]
        feature_col = st.selectbox("Select column for Lag/Rolling:", ["None"] + num_cols)
        if feature_col != "None":
            lag_periods = st.text_input("Lag Periods (comma-separated)", "1, 7")
            rolling_window = st.number_input("Rolling Window Size", min_value=0, value=7)

    st.markdown("---")
    st.subheader("Feature Selection")
    
    col_fs1, col_fs2 = st.columns(2)
    with col_fs1:
        st.write("#### Manual Selection")
        manual_features = st.multiselect("Manually select features to keep:", df.columns, help="These features will always be included.")
    
    with col_fs2:
        st.write("#### Automated Selection")
        enable_rfe = st.checkbox("Enable Automated Feature Selection (RFE)")
        if enable_rfe:
            num_features_to_select = st.slider("Number of top features to select:", 1, len(df.columns)-1, min(10, len(df.columns)-1))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'eda'
            st.rerun()
    with col2:
        if st.button("Apply Features & Proceed"):
            with st.spinner("Engineering and Selecting Features..."):
                proc_df = df.copy()
                original_df = st.session_state.original_processed_df.copy()
                st.session_state.user_selections.update({
                    'time_index': time_index, 'date_parts': extract_parts, 'segment_by': segment_by,
                    'feature_eng_col': feature_col,
                    'lag_periods': lag_periods if feature_col != "None" else "", 'rolling_window': rolling_window if feature_col != "None" else 0,
                    'manual_features': manual_features,
                    'rfe_enabled': enable_rfe, 'rfe_features': num_features_to_select if enable_rfe else 0
                })
                if time_index != "None" and time_index in original_df.columns:
                    proc_df[time_index] = pd.to_datetime(original_df[time_index])
                    if 'Year' in extract_parts: proc_df['Year'] = proc_df[time_index].dt.year
                    if 'Quarter' in extract_parts: proc_df['Quarter'] = proc_df[time_index].dt.quarter
                    if 'Month' in extract_parts: proc_df['Month'] = proc_df[time_index].dt.month
                    proc_df.drop(columns=[time_index], inplace=True, errors='ignore')

                if feature_col != "None":
                    if lag_periods:
                        periods = [int(p.strip()) for p in lag_periods.split(',') if p.strip()]
                        for p in periods:
                            proc_df[f'{feature_col}_lag_{p}'] = proc_df[feature_col].shift(p)
                    if rolling_window > 0:
                        proc_df[f'{feature_col}_roll_mean_{rolling_window}'] = proc_df[feature_col].rolling(window=rolling_window).mean()
                
                proc_df.dropna(inplace=True)
                st.session_state.processed_df = proc_df
                st.session_state.page = 'model_recommendation'
                st.rerun()

def render_recommendation_page():
    st.header("Model Recommendation")
    df = st.session_state.processed_df
    selections = st.session_state.user_selections
    
    is_time_series = selections.get('time_index') != 'None'
    
    if is_time_series:
        st.success("**Top Recommendation: Prophet**")
        st.info("Because you selected a time index, your goal is likely forecasting. Prophet is a specialized forecasting model that is robust to missing data and shifts in trend, and typically handles seasonality well.")
        st.warning("**Secondary Option: ARIMA**")
        st.info("ARIMA is another powerful statistical model for time-series data.")
    else:
        st.info("This is a data-driven model showdown. We train several models on a sample of your data to see which performs best, giving you an intelligent starting point.")
        target_column = st.selectbox("Select your target column for the showdown:", df.columns, index=len(df.columns)-1)

        if st.button("Find Best Model for My Data"):
            with st.spinner("Running model showdown... This may take a moment."):
                problem_type = "Regression" if pd.api.types.is_numeric_dtype(df[target_column].dtype) else "Classification"
                
                sample_df = df.select_dtypes(exclude=['datetime', 'datetimetz']).sample(n=min(len(df), 1000), random_state=42)
                
                for col in sample_df.select_dtypes(include=['object', 'category']).columns:
                    sample_df[col] = LabelEncoder().fit_transform(sample_df[col].astype(str))

                X = sample_df.drop(columns=[target_column])
                y = sample_df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                models_to_test = {
                    "Linear/Logistic Regression": LinearRegression() if problem_type == "Regression" else LogisticRegression(),
                    "Random Forest": RandomForestRegressor(random_state=42) if problem_type == "Regression" else RandomForestClassifier(random_state=42),
                }
                if XGB_AVAILABLE:
                    models_to_test["XGBoost"] = xgb.XGBRegressor(random_state=42) if problem_type == "Regression" else xgb.XGBClassifier(random_state=42)

                results = []
                for name, model in models_to_test.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    if problem_type == "Regression":
                        score = r2_score(y_test, preds)
                        metric_name = "R-Squared"
                    else:
                        score = accuracy_score(y_test, preds)
                        metric_name = "Accuracy"
                    results.append({"Model": name, metric_name: score})
                
                results_df = pd.DataFrame(results).sort_values(by=metric_name, ascending=False).reset_index(drop=True)
                st.session_state.showdown_results = results_df
                
    if st.session_state.showdown_results is not None:
        st.subheader("Model Showdown Leaderboard")
        st.dataframe(st.session_state.showdown_results)
        top_model = st.session_state.showdown_results.iloc[0]["Model"]
        st.success(f"**Top Recommendation:** Based on the showdown, **{top_model}** is the recommended model.")

    st.markdown("---")
    st.write("You can accept the recommendation or choose any model on the next page.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'feature_engineering'
            st.rerun()
    with col2:
        if st.button("Proceed to Modeling"):
            st.session_state.page = 'modeling'
            st.rerun()

def render_modeling_page():
    st.header("Modeling (Predictive)")
    df = st.session_state.processed_df
    selections = st.session_state.user_selections
    
    target_column = st.selectbox("Select Target Column:", df.columns, index=len(df.columns)-1, key="target_select")
    problem_type = "Regression" if pd.api.types.is_numeric_dtype(df[target_column].dtype) else "Classification"
    st.session_state.problem_type = problem_type

    if selections.get('rfe_enabled') and 'rfe_ran' not in st.session_state:
        with st.spinner("Running Recursive Feature Elimination..."):
            temp_df = df.select_dtypes(exclude=['datetime', 'datetimetz']).dropna()
            X_temp = temp_df.drop(columns=[target_column])
            y_temp = temp_df[target_column]
            estimator = RandomForestRegressor(n_estimators=50, random_state=42) if problem_type == "Regression" else RandomForestClassifier(n_estimators=50, random_state=42)
            rfe = RFE(estimator, n_features_to_select=selections['rfe_features'])
            rfe.fit(X_temp, y_temp)
            
            rfe_selected_features = list(X_temp.columns[rfe.support_])
            manual_features = selections.get('manual_features', [])
            
            final_features = list(set(rfe_selected_features + manual_features))
            
            df = df[final_features + [target_column]]
            st.session_state.processed_df = df
            st.session_state['rfe_ran'] = True
            st.success(f"RFE Complete. Final features selected: {', '.join(final_features)}")

    st.write("#### Final Data for Modeling"); st.dataframe(df.head())
    st.markdown("---")

    ml_algorithms = ["Random Forest", "Gradient Boosting", "Support Vector Machine (SVM)", "Linear/Logistic Regression"]
    if XGB_AVAILABLE: ml_algorithms.append("XGBoost")
    if CATBOOST_AVAILABLE: ml_algorithms.append("CatBoost")
    ts_algorithms = ["Prophet", "ARIMA"] if PROPHET_AVAILABLE else []
    
    algorithm = st.selectbox("Select Algorithm:", ml_algorithms + ts_algorithms, key="algo_select")
    
    is_ts_model = algorithm in ts_algorithms
    if is_ts_model:
        if selections.get('time_index') == 'None':
            st.error("ARIMA and Prophet require a time index column.")
            return
        if problem_type == "Classification":
            st.error("ARIMA and Prophet are for regression problems only.")
            return

    if st.button("Train Model"):
        st.session_state.user_selections['target'] = target_column
        st.session_state.user_selections['algorithm'] = algorithm
        
        with st.spinner(f"Training {algorithm} model..."):
            if is_ts_model:
                models = {}
                # Train overall model
                ts_df = st.session_state.df.copy()
                time_col = selections['time_index']
                ts_df['ds'] = pd.to_datetime(ts_df[time_col], dayfirst=True)
                ts_df.rename(columns={target_column: 'y'}, inplace=True)
                
                if algorithm == "Prophet":
                    model_overall = Prophet().fit(ts_df[['ds', 'y']])
                elif algorithm == "ARIMA":
                    model_overall = ARIMA(ts_df['y'].values, order=(5,1,0)).fit()
                models['Overall'] = model_overall

                # Train segmented models if applicable
                segment_by = selections.get('segment_by')
                if segment_by and segment_by != "None":
                    for segment_value in st.session_state.df[segment_by].unique():
                        segment_df = st.session_state.df[st.session_state.df[segment_by] == segment_value].copy()
                        segment_df['ds'] = pd.to_datetime(segment_df[time_col], dayfirst=True)
                        segment_df.rename(columns={target_column: 'y'}, inplace=True)
                        if algorithm == "Prophet":
                            model_segment = Prophet().fit(segment_df[['ds', 'y']])
                        elif algorithm == "ARIMA":
                            model_segment = ARIMA(segment_df['y'].values, order=(5,1,0)).fit()
                        models[segment_value] = model_segment
                st.session_state.model = models
            else:
                X = df.select_dtypes(exclude=['datetime', 'datetimetz']).drop(columns=[target_column])
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model_options = {
                    "Random Forest": RandomForestRegressor(random_state=42) if problem_type == "Regression" else RandomForestClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42) if problem_type == "Regression" else GradientBoostingClassifier(random_state=42),
                    "Support Vector Machine (SVM)": SVR() if problem_type == "Regression" else SVC(random_state=42, probability=True),
                    "Linear/Logistic Regression": LinearRegression() if problem_type == "Regression" else LogisticRegression(random_state=42, max_iter=1000),
                    "XGBoost": xgb.XGBRegressor(random_state=42) if problem_type == "Regression" else xgb.XGBClassifier(random_state=42),
                    "CatBoost": cb.CatBoostRegressor(random_state=42, verbose=0) if problem_type == "Regression" else cb.CatBoostClassifier(random_state=42, verbose=0)
                }
                model = model_options[algorithm]
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                st.subheader("Quick Test Set Performance")
                predictions = model.predict(X_test)
                if problem_type == "Regression":
                    plot_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions}).reset_index()
                    fig_pred = px.line(plot_df, x='index', y=['Actual', 'Predicted'], title='Actual vs. Predicted on Test Set', template='plotly_dark')
                    st.plotly_chart(fig_pred, use_container_width=True)
            
            st.success(f"{algorithm} model(s) trained successfully!")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'model_recommendation'
            st.rerun()
    with col2:
        if st.button("Proceed to Full Evaluation"):
            st.session_state.page = 'evaluation'
            st.rerun()

def render_evaluation_page():
    st.header("Evaluation & Explainability (Predictive)")
    model_or_models = st.session_state.model
    selections = st.session_state.user_selections
    algorithm = selections.get('algorithm', 'Unknown Model')
    is_ts_model = algorithm in ["Prophet", "ARIMA"]
    report_data = st.session_state.get('report_data', {})
    report_data['segment_forecasts'] = {}

    if is_ts_model:
        tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Components", "📄 Automated Report"])
        with tab1:
            st.subheader("Overall Forecast")
            overall_model = model_or_models['Overall']
            with st.spinner(f"Generating overall forecast plot..."):
                if algorithm == "Prophet":
                    horizon = selections.get('forecast_horizon', 365)
                    freq = selections.get('forecast_freq', 'D')
                    future = overall_model.make_future_dataframe(periods=horizon, freq=freq)
                    forecast = overall_model.predict(future)
                    st.session_state.forecast_df = forecast
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=overall_model.history['ds'].tail(365), y=overall_model.history['y'].tail(365), mode='lines', name='Historical (Last Year)'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)'))
                    st.plotly_chart(fig, use_container_width=True)
                    report_data['forecast_fig'] = (overall_model.history, forecast)
            
            if len(model_or_models) > 1:
                st.subheader("Forecast Summary by Segment")
                forecast_summaries = []
                for segment, model in model_or_models.items():
                    if segment != 'Overall' and algorithm == "Prophet":
                        future = model.make_future_dataframe(periods=selections.get('forecast_horizon', 365), freq=selections.get('forecast_freq', 'D'))
                        forecast = model.predict(future)
                        total_forecast = forecast['yhat'].sum()
                        forecast_summaries.append({'Segment': segment, 'Total Forecasted Value': total_forecast})
                
                if forecast_summaries:
                    summary_df = pd.DataFrame(forecast_summaries)
                    fig_bar = px.bar(summary_df, x='Segment', y='Total Forecasted Value', title='Total Forecasted Value by Segment', template='plotly_dark')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    report_data['segment_summary_bar'] = summary_df

                with st.expander("View Detailed Forecasts by Segment"):
                    for segment, model in model_or_models.items():
                        if segment != 'Overall':
                            st.write(f"### Forecast for: {segment}")
                            with st.spinner(f"Generating forecast plot for {segment}..."):
                                if algorithm == "Prophet":
                                    future = model.make_future_dataframe(periods=selections.get('forecast_horizon', 365), freq=selections.get('forecast_freq', 'D'))
                                    forecast = model.predict(future)
                                    
                                    fig_segment = go.Figure()
                                    fig_segment.add_trace(go.Scatter(x=model.history['ds'].tail(365), y=model.history['y'].tail(365), mode='lines', name='Historical'))
                                    fig_segment.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                                    fig_segment.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)'))
                                    fig_segment.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)'))
                                    st.plotly_chart(fig_segment, use_container_width=True)
                                    report_data['segment_forecasts'][segment] = (model.history, forecast)
        with tab2:
            st.subheader("Forecast Components")
            if algorithm == "Prophet":
                fig_components = plot_components_plotly(model_or_models['Overall'], st.session_state.forecast_df)
                st.plotly_chart(fig_components, use_container_width=True)
                report_data['components_fig'] = (model_or_models['Overall'], st.session_state.forecast_df)

        with tab3:
            st.subheader("Export Artifacts")
            if st.session_state.forecast_df is not None:
                forecast_csv = st.session_state.forecast_df.to_csv().encode('utf-8')
                st.download_button("Download Forecast Data (CSV)", forecast_csv, "forecast_data.csv", "text/csv")

            if st.session_state.model:
                model_pkl = pickle.dumps(st.session_state.model)
                st.download_button("Download Trained Model (PKL)", model_pkl, "model.pkl")
            
            st.download_button("Download Full Report (PDF)", create_pdf_report(report_data), "report.pdf")

    else:
        tab1, tab2 = st.tabs(["📊 Dashboard", "📄 Automated Report"])
        with tab1:
            st.subheader("Performance Metrics")
            X_test, y_test = st.session_state.X_test, st.session_state.y_test
            predictions = model_or_models.predict(X_test)
            metrics = {}
            if st.session_state.problem_type == "Regression":
                metrics['R-Squared'] = r2_score(y_test, predictions)
                metrics['MAE'] = mean_absolute_error(y_test, predictions)
            else:
                metrics['Accuracy'] = accuracy_score(y_test, predictions)
                metrics['F1-Score'] = f1_score(y_test, predictions, average='weighted')
            st.session_state.metrics = metrics
            metric_cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                metric_cols[i].metric(key, f"{value:.4f}")
            
            if st.session_state.problem_type == "Classification":
                st.subheader("ROC Curve")
                y_pred_proba = model_or_models.predict_proba(st.session_state.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
                st.plotly_chart(fig_roc, use_container_width=True)
                report_data['roc_curve'] = (fpr, tpr, roc_auc)
        
        with tab2:
            st.subheader("Export Artifacts")
            processed_csv = st.session_state.processed_df.to_csv().encode('utf-8')
            st.download_button("Download Processed Data (CSV)", processed_csv, "processed_data.csv", "text/csv")

            if st.session_state.model:
                model_pkl = pickle.dumps(st.session_state.model)
                st.download_button("Download Trained Model (PKL)", model_pkl, "model.pkl")
            
            st.download_button("Download Full Report (PDF)", create_pdf_report(report_data), "report.pdf")
    
    st.session_state.report_data = report_data

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'modeling'
            st.rerun()
    with col2:
        if st.button("Proceed to Action & Export"):
            st.session_state.page = 'action_export'
            st.rerun()

def render_clustering_page():
    st.header("Clustering Model (K-Means)")
    df = st.session_state.processed_df
    st.write("#### Data for Clustering"); st.dataframe(df.head())
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Find Optimal Number of Clusters (Elbow Method)")
        with st.spinner("Calculating inertia..."):
            inertia = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans.fit(df)
                inertia.append(kmeans.inertia_)
            fig = px.line(x=k_range, y=inertia, title="Elbow Method for Optimal k")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        k = st.number_input("Select Number of Clusters (k)", 2, 20, 3)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'eda'
            st.rerun()
    with col2:
        if st.button("Run Clustering & Analyze Results"):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df)
            st.session_state.processed_df = df
            st.session_state.model = kmeans
            st.session_state.page = 'cluster_analysis'
            st.rerun()

def render_cluster_analysis_page():
    st.header("Cluster Analysis")
    df = st.session_state.processed_df
    report_data = st.session_state.get('report_data', {})
    
    st.subheader("Cluster Population")
    fig_bar = px.bar(df['Cluster'].value_counts(), title="Cluster Population", template='plotly_dark')
    st.plotly_chart(fig_bar, use_container_width=True)
    report_data['cluster_population_data'] = df['Cluster'].value_counts()
    
    st.subheader("Cluster Scatter Plot")
    all_cols = [col for col in df.columns if col != 'Cluster']
    x_axis = st.selectbox("X-Axis", all_cols, index=0)
    y_axis = st.selectbox("Y-Axis", all_cols, index=1)
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color="Cluster", title=f"{x_axis} vs. {y_axis} by Cluster")
    st.plotly_chart(fig_scatter, use_container_width=True)
    report_data['cluster_scatter_data'] = (df, x_axis, y_axis)
    
    st.subheader("Cluster Centers")
    st.dataframe(df.groupby('Cluster').mean())
    st.session_state.report_data = report_data

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'clustering'
            st.rerun()
    with col2:
        if st.button("Proceed to Action & Export"):
            st.session_state.page = 'action_export'
            st.rerun()

def create_pdf_report(report_data={}):
    pdf = PDF()
    pdf.alias_nb_pages()
    
    # --- Title Page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 100, "AI Data Science Pipeline Report", 0, 1, 'C')
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Task: {st.session_state.task}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')

    # --- Executive Summary Page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Executive Summary", 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    selections = st.session_state.user_selections
    task_summary = f"The pipeline was configured to perform a {st.session_state.task} task."
    if st.session_state.task == "Predictive Modeling":
        task_summary += f" The goal was to predict '{selections.get('target', 'N/A')}' using the {selections.get('algorithm', 'N/A')} algorithm."
    pdf.multi_cell(0, 5, task_summary)
    pdf.ln(10)
    
    if 'metrics' in st.session_state and st.session_state.metrics:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Performance Metrics", 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        for key, value in st.session_state.metrics.items():
            pdf.cell(0, 8, f"  -  {key}: {value:.4f}", 0, 1)
        pdf.ln(5)

    if report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Visual Analysis", 0, 1, 'L')
        
        with plt.style.context('dark_background'):
            # --- Evaluation Charts ---
            if 'forecast_fig' in report_data:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Overall Forecast", 0, 1)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 5, "This chart displays the historical data along with the model's forecast. The shaded area represents the uncertainty interval for the predictions.")
                pdf.ln(5)
                history, forecast = report_data['forecast_fig']
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history['ds'].tail(365), history['y'].tail(365), label='Historical (Last Year)')
                ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name)
                    pdf.image(tmpfile.name, w=180)
                os.remove(tmpfile.name)
                plt.close(fig)
                pdf.ln(5)
            
            if 'components_fig' in report_data:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Forecast Components", 0, 1)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 5, "This plot shows the forecast components: the overall trend, and the weekly and yearly seasonal patterns discovered in the data. This helps to understand the underlying drivers of the forecast.")
                pdf.ln(5)
                model, forecast = report_data['components_fig']
                fig = prophet_plot_components(model, forecast)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name)
                    pdf.image(tmpfile.name, w=180)
                os.remove(tmpfile.name)
                plt.close(fig)
                pdf.ln(5)
            
            if 'segment_summary_bar' in report_data:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Forecast Summary by Segment", 0, 1)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 5, "This bar chart compares the total forecasted value across different segments, providing a high-level overview of the predictions for each group.")
                pdf.ln(5)
                summary_df = report_data['segment_summary_bar']
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(summary_df['Segment'], summary_df['Total Forecasted Value'])
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name)
                    pdf.image(tmpfile.name, w=160)
                os.remove(tmpfile.name)
                plt.close(fig)
                pdf.ln(5)
                
            if 'segment_forecasts' in report_data and report_data['segment_forecasts']:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Detailed Segment Forecasts", 0, 1)
                for segment, (history, forecast) in report_data['segment_forecasts'].items():
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, f"Segment: {segment}", 0, 1)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history['ds'].tail(365), history['y'].tail(365), label='Historical')
                    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
                    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
                    ax.legend()
                    plt.tight_layout()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name)
                        pdf.image(tmpfile.name, w=180)
                    os.remove(tmpfile.name)
                    plt.close(fig)
                    pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')

def render_action_export_page():
    # Conditionally skip this page for time-series models
    if st.session_state.task == "Predictive Modeling" and st.session_state.user_selections.get('algorithm') in ["Prophet", "ARIMA"]:
        return

    st.header("Action & Export")

    if st.button("Go Back"):
        if st.session_state.task == "Clustering Analysis":
            st.session_state.page = 'cluster_analysis'
        else: # Predictive Modeling
            st.session_state.page = 'evaluation'
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.task == "Clustering Analysis":
        st.subheader("Cluster Interpretation & Naming")
        df = st.session_state.processed_df
        cluster_means = df.groupby('Cluster').mean()
        population_means = df.drop(columns=['Cluster']).mean()

        if not st.session_state.cluster_labels:
            for cluster_id in df['Cluster'].unique():
                st.session_state.cluster_labels[cluster_id] = f"Cluster {cluster_id}"
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.write("#### Name Your Clusters")
            for cluster_id in sorted(df['Cluster'].unique()):
                 st.session_state.cluster_labels[cluster_id] = st.text_input(
                     f"Label for Cluster {cluster_id}:", 
                     value=st.session_state.cluster_labels[cluster_id]
                 )
        with col2:
            st.write("#### Explore a Cluster")
            selected_cluster = st.selectbox("Select a cluster to profile:", sorted(df['Cluster'].unique()))
            
            cluster_data = cluster_means.loc[selected_cluster]
            comparison_df = pd.DataFrame({'Cluster Average': cluster_data, 'Population Average': population_means}).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comparison_df['index'], y=comparison_df['Cluster Average'], name='Cluster Average'))
            fig.add_trace(go.Bar(x=comparison_df['index'], y=comparison_df['Population Average'], name='Population Average'))
            fig.update_layout(barmode='group', title=f"Profile of Cluster {selected_cluster}", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Export Artifacts for Clustering")
        labeled_df = st.session_state.original_processed_df.copy()
        labeled_df['Cluster ID'] = st.session_state.processed_df['Cluster']
        labeled_df['Cluster Label'] = labeled_df['Cluster ID'].map(st.session_state.cluster_labels)
        export_csv = labeled_df.to_csv().encode('utf-8')
        st.download_button("Download Labeled Data (CSV)", export_csv, "labeled_cluster_data.csv", "text/csv")
        if st.session_state.model:
            model_pkl = pickle.dumps(st.session_state.model)
            st.download_button("Download Trained Model (PKL)", model_pkl, "model.pkl")
        
        st.download_button("Download Full Report (PDF)", create_pdf_report(st.session_state.get('report_data', {})), "report.pdf")


    else: # Predictive Modeling
        st.subheader("Live Prediction Simulator")
        if st.session_state.model and not (st.session_state.user_selections.get('algorithm') in ["Prophet", "ARIMA"]):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_.flatten())
            else:
                importances = np.zeros(len(X_test.columns))

            feature_importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': importances}).sort_values('importance', ascending=False)
            
            base_features_df = feature_importance_df[~feature_importance_df['feature'].str.contains('_lag_|_roll_')]
            top_features = base_features_df.head(5)['feature'].tolist()

            input_data = {}
            st.info("Adjust the top controllable features to see how they affect the prediction.")
            for feature in top_features:
                if pd.api.types.is_numeric_dtype(X_test[feature].dtype):
                    min_val, max_val = float(X_test[feature].min()), float(X_test[feature].max())
                    input_data[feature] = st.slider(f"Adjust {feature}", min_val, max_val, float(X_test[feature].mean()))
                else: 
                    unique_vals = list(X_test[feature].unique())
                    input_data[feature] = st.selectbox(f"Select {feature}", options=unique_vals, index=0)

            for col in X_test.columns:
                if col not in input_data:
                    input_data[col] = X_test[col].mean() if pd.api.types.is_numeric_dtype(X_test[col].dtype) else X_test[col].mode()[0]
            
            input_df = pd.DataFrame([input_data])[X_test.columns] 
            
            prediction = model.predict(input_df)
            pred_proba = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None

            if st.session_state.problem_type == "Classification":
                st.success(f"**Predicted Outcome:** `{prediction[0]}`")
                if pred_proba is not None:
                    prob_df = pd.DataFrame(pred_proba, columns=model.classes_).T
                    prob_df.columns = ["Probability"]
                    st.bar_chart(prob_df)
            else:
                st.success(f"**Predicted Value:** `{prediction[0]:,.2f}`")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction[0],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Value Gauge"},
                    gauge = {'axis': {'range': [y_test.min(), y_test.max()]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

# --- Main App Logic ---
def main():
    apply_custom_theme()
    initialize_session_state()
    render_sidebar()
    page = st.session_state.page
    
    if page == 'upload': render_upload_page()
    elif page == 'task_hub': render_task_hub()
    elif page == 'preprocessing': render_preprocessing_page()
    elif page == 'eda': render_eda_page()
    elif page == 'action_export': render_action_export_page()
    elif st.session_state.task == "Predictive Modeling":
        if page == 'feature_engineering': render_feature_engineering_page()
        elif page == 'model_recommendation': render_recommendation_page()
        elif page == 'modeling': render_modeling_page()
        elif page == 'evaluation': render_evaluation_page()
    elif st.session_state.task == "Clustering Analysis":
        if page == 'clustering': render_clustering_page()
        elif page == 'cluster_analysis': render_cluster_analysis_page()

if __name__ == "__main__":
    main()

