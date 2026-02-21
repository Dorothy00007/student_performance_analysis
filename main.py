"""
Student Performance Analytics - Fixed Version (No statsmodels needed)
Save this as t.py and run: streamlit run t.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# ============================================
# Generate Complete Sample Data with ALL Features
# ============================================
@st.cache_data
def generate_complete_data(n_students=500):
    """Generate complete sample student data with all features"""
    np.random.seed(42)
    
    data = {
        'StudentID': range(1001, 1001 + n_students),
        'Age': np.random.choice([15, 16, 17, 18], n_students),
        'Gender': np.random.choice([0, 1], n_students),
        'Ethnicity': np.random.choice([0, 1, 2, 3], n_students),
        'ParentalEducation': np.random.choice([0, 1, 2, 3, 4], n_students),
        'StudyTimeWeekly': np.random.uniform(0, 20, n_students),
        'Absences': np.random.poisson(5, n_students),
        'Tutoring': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
        'ParentalSupport': np.random.choice([0, 1, 2, 3, 4], n_students),
        'Extracurricular': np.random.choice([0, 1], n_students, p=[0.6, 0.4]),
        'Sports': np.random.choice([0, 1], n_students, p=[0.55, 0.45]),
        'Music': np.random.choice([0, 1], n_students, p=[0.65, 0.35]),
        'Volunteering': np.random.choice([0, 1], n_students, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate GPA
    df['GPA'] = (
        2.0 + df['StudyTimeWeekly'] * 0.08 + df['ParentalSupport'] * 0.15 +
        df['Tutoring'] * 0.3 + (df['Extracurricular'] + df['Sports'] + 
        df['Music'] + df['Volunteering']) * 0.1 - df['Absences'] * 0.05 +
        np.random.normal(0, 0.3, n_students)
    )
    df['GPA'] = df['GPA'].clip(0, 4)
    
    # Calculate Grade Class
    conditions = [
        (df['GPA'] >= 3.5),
        (df['GPA'] >= 3.0) & (df['GPA'] < 3.5),
        (df['GPA'] >= 2.5) & (df['GPA'] < 3.0),
        (df['GPA'] >= 2.0) & (df['GPA'] < 2.5),
        (df['GPA'] < 2.0)
    ]
    choices = [0, 1, 2, 3, 4]
    df['GradeClass'] = np.select(conditions, choices, default=4)
    
    # Add labels
    df['Gender_Label'] = df['Gender'].map({0: 'Male', 1: 'Female'})
    df['Ethnicity_Label'] = df['Ethnicity'].map({0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'})
    df['ParentalEdu_Label'] = df['ParentalEducation'].map({0: 'None', 1: 'High School', 2: 'Some College', 3: 'Bachelor', 4: 'Higher'})
    df['TotalActivities'] = df[['Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering']].sum(axis=1)
    
    return df

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.title("🎓 Student Analytics")
    st.markdown("---")
    
    # Data Source
    st.subheader("📁 Data Source")
    if st.button("📊 Generate Sample Data", use_container_width=True):
        with st.spinner("Generating data..."):
            st.session_state.data = generate_complete_data(500)
        st.success("✅ Data generated!")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio("Go to:", [
        "🏠 Dashboard",
        "🔍 Data Explorer",
        "🤖 ML Models",
        "🎯 Risk Predictor"
    ])

# ============================================
# Main Content
# ============================================

# Check if data is loaded
if st.session_state.data is None:
    st.markdown("<h1 class='main-header'>🎓 Student Performance Analytics</h1>", unsafe_allow_html=True)
    st.info("Click 'Generate Sample Data' in the sidebar to get started!")
    st.stop()

df = st.session_state.data

# ============================================
# Dashboard Page (FIXED - removed trendline)
# ============================================
if page == "🏠 Dashboard":
    st.markdown("<h1 class='main-header'>📊 Dashboard</h1>", unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Average GPA", f"{df['GPA'].mean():.2f}")
    col3.metric("Avg Study Time", f"{df['StudyTimeWeekly'].mean():.1f}h")
    col4.metric("Avg Absences", f"{df['Absences'].mean():.1f}")
    
    # Charts - FIXED: removed trendline parameter
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='GPA', nbins=20, title='GPA Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        grade_counts = df['GradeClass'].value_counts().sort_index()
        fig = px.bar(x=grade_counts.index, y=grade_counts.values,
                    labels={'x': 'Grade Class', 'y': 'Count'},
                    title='Grade Class Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots - FIXED: removed trendline
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='StudyTimeWeekly', y='GPA', color='Gender_Label',
                        title='Study Time vs GPA',
                        opacity=0.6)  # Removed trendline
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='Absences', y='GPA', color='Gender_Label',
                        title='Absences vs GPA',
                        opacity=0.6)  # Removed trendline
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# Data Explorer Page
# ============================================
elif page == "🔍 Data Explorer":
    st.markdown("<h1 class='main-header'>🔍 Data Explorer</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 Data Table", "📊 Statistics"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name="student_data.csv")
    
    with tab2:
        st.dataframe(df.describe().round(3), use_container_width=True)

# ============================================
# ML Models Page
# ============================================
elif page == "🤖 ML Models":
    st.markdown("<h1 class='main-header'>🤖 ML Models</h1>", unsafe_allow_html=True)
    
    features = ['StudyTimeWeekly', 'Absences', 'ParentalSupport', 'TotalActivities']
    X = df[features]
    y = df['GPA']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                title='Feature Importance')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# Risk Predictor Page
# ============================================
elif page == "🎯 Risk Predictor":
    st.markdown("<h1 class='main-header'>🎯 Risk Predictor</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        study = st.slider("Study Time", 0, 20, 10)
        absences = st.slider("Absences", 0, 30, 5)
    
    with col2:
        support = st.slider("Parental Support", 0, 4, 2)
        activities = st.slider("Activities", 0, 5, 2)
    
    if st.button("Predict Risk", use_container_width=True):
        features = ['StudyTimeWeekly', 'Absences', 'ParentalSupport', 'TotalActivities']
        X = df[features]
        y = df['GPA']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        pred = model.predict([[study, absences, support, activities]])[0]
        
        if pred >= 3.0:
            st.success(f"✅ LOW RISK - Predicted GPA: {pred:.2f}")
        elif pred >= 2.0:
            st.warning(f"⚠️ MEDIUM RISK - Predicted GPA: {pred:.2f}")
        else:
            st.error(f"❌ HIGH RISK - Predicted GPA: {pred:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and scikit-learn")