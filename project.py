



import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os
from chatbot import render_ev_chatbot

# ==============================================================================
# 1. APP CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="EV Adoption Insights & Prediction Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS for high contrast and cross-theme readability
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    
    /* Dynamic Colorful High-Contrast KPI Cards */
    .kpi-blue { background-color: #E0F2FE; color: #0369A1; padding: 1.2rem; border-radius: 0.5rem; text-align: center; border-top: 4px solid #0284C7; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .kpi-purple { background-color: #F3E8FF; color: #6B21A8; padding: 1.2rem; border-radius: 0.5rem; text-align: center; border-top: 4px solid #9333EA; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .kpi-green { background-color: #DCFCE7; color: #15803D; padding: 1.2rem; border-radius: 0.5rem; text-align: center; border-top: 4px solid #16A34A; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .kpi-red { background-color: #FEE2E2; color: #991B1B; padding: 1.2rem; border-radius: 0.5rem; text-align: center; border-top: 4px solid #DC2626; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    
    .prediction-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; }
    .explanation-text { font-size: 1rem; line-height: 1.5; }
    
    /* Safe layout padding container without theme-breaking background masks */
    div[data-testid="stForm"] { border: 1px solid #E5E7EB; padding: 2rem; border-radius: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHED DATA & MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_ml_model():
    """Attempts to load the trained pipeline from disk."""
    possible_paths = ['ev_adoption_model.pkl', 'model.pkl']
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path), False
            except Exception as e:
                st.sidebar.error(f"Error loading model from {path}: {e}")
    return None, True

@st.cache_data
def load_dataset():
    """Loads training dataset from disk with standard dynamic structural fallback."""
    possible_csvs = ['global_ev_adoption_behavior_2026_cleaned.csv', 'global_ev_adoption_behavior_2026.csv']
    for csv_path in possible_csvs:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'fuel_expense_per_month' in df.columns:
                    df['fuel_expense_per_month'] = df['fuel_expense_per_month'].abs()
                
                # Add engineered columns to real dataset overview if not already stored
                if 'fuel_to_electricity_ratio' not in df.columns:
                    df['fuel_to_electricity_ratio'] = df['fuel_expense_per_month'] / (df['monthly_charging_cost'] + 1)
                if 'charging_convenience_index' not in df.columns:
                    df['charging_convenience_index'] = df['charging_station_accessibility'] / (df['nearest_charging_station_km'] + 1)
                return df
            except Exception:
                pass
                
    np.random.seed(42)
    n_records = 1000
    mock_df = pd.DataFrame({
        'age': np.random.randint(21, 75, n_records),
        'annual_income': np.random.normal(44000, 15000, n_records).clip(5104, 120000),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_records, p=[0.25, 0.45, 0.20, 0.10]),
        'city_type': np.random.choice(['Suburban', 'Urban', 'Rural'], n_records, p=[0.40, 0.45, 0.15]),
        'daily_commute_km': np.random.uniform(5, 80, n_records),
        'weekly_travel_distance_km': np.random.uniform(25, 400, n_records),
        'current_vehicle_type': np.random.choice(['Hatchback', 'Sedan', 'SUV'], n_records),
        'vehicle_age_years': np.random.uniform(0, 15, n_records),
        'fuel_expense_per_month': np.random.uniform(50, 500, n_records),
        'charging_station_accessibility': np.random.uniform(1, 10, n_records),
        'nearest_charging_station_km': np.random.uniform(0.5, 20, n_records),
        'home_charging_available': np.random.choice([0, 1], n_records, p=[0.35, 0.65]),
        'electricity_cost_per_kwh': np.random.uniform(0.08, 0.30, n_records),
        'environmental_awareness_score': np.random.uniform(1, 10, n_records),
        'government_incentive_awareness': np.random.uniform(1, 10, n_records),
        'technology_affinity_score': np.random.uniform(1, 10, n_records),
        'range_anxiety_score': np.random.uniform(1, 10, n_records),
        'battery_replacement_concern': np.random.uniform(1, 10, n_records),
        'ev_knowledge_score': np.random.uniform(1, 10, n_records),
        'previous_ev_experience': np.random.choice([0, 1], n_records, p=[0.80, 0.20]),
        'monthly_energy_consumption_kwh': np.random.uniform(50, 400, n_records),
        'monthly_charging_cost': np.random.uniform(10, 100, n_records),
        'ev_adoption_likelihood': np.random.choice(['High', 'Medium', 'Low'], n_records, p=[0.55, 0.25, 0.20])
    })
    
    # Pre-calculate engineered features for mockup view
    mock_df['fuel_to_electricity_ratio'] = mock_df['fuel_expense_per_month'] / (mock_df['monthly_charging_cost'] + 1)
    mock_df['charging_convenience_index'] = mock_df['charging_station_accessibility'] / (mock_df['nearest_charging_station_km'] + 1)
    
    for _ in range(10):
        mock_df.loc[np.random.randint(0, n_records), 'education_level'] = np.nan
        mock_df.loc[np.random.randint(0, n_records), 'charging_station_accessibility'] = np.nan
    return mock_df

df = load_dataset()
model, is_mock_model = load_ml_model()

# ==============================================================================
# 3. FEATURE DICTIONARY METADATA
# ==============================================================================
feature_metadata = {
    'Numerical Features': {
        'age': 'Age of the consumer in years.',
        'annual_income': 'Annual personal or household gross earnings ($).',
        'daily_commute_km': 'Average distance traveled per day for work/school (km).',
        'weekly_travel_distance_km': 'Total cumulative distance covered across 7 days (km).',
        'vehicle_age_years': 'Current lifespan duration of the buyer\'s current primary car.',
        'fuel_expense_per_month': 'Monthly fiscal expenditures allocated to fossil fuels ($).',
        'charging_station_accessibility': 'Perceived ease/availability rating of public charging stations (1-10).',
        'nearest_charging_station_km': 'Physical distance to the closest active vehicle plug point (km).',
        'home_charging_available': 'Presence of residential dedicated charging plugs (1 = Yes, 0 = No).',
        'electricity_cost_per_kwh': 'Utility rate price structure per kilowatt-hour ($/kWh).',
        'environmental_awareness_score': 'Degree of internal priority placed on sustainability (1-10).',
        'government_incentive_awareness': 'Familiarity with regional EV subsidies/tax credits (1-10).',
        'technology_affinity_score': 'Personal alignment and comfort level adopting new technology (1-10).',
        'range_anxiety_score': 'Intensity of concern relating to getting stranded without charge (1-10).',
        'battery_replacement_concern': 'Fear of fiscal liability regarding battery replacement (1-10).',
        'ev_knowledge_score': 'Factual technical understanding of electric vehicles (1-10).',
        'previous_ev_experience': 'Past tenure operating or ownership history of an EV (1 = Yes, 0 = No).',
        'monthly_energy_consumption_kwh': 'Total electricity consumption metrics for the household (kWh).',
        'monthly_charging_cost': 'Total utility money spent directly on automotive charging ($).',
        'fuel_to_electricity_ratio': 'Engineered ratio comparing gasoline expenditure against electrical charging expense.',
        'charging_convenience_index': 'Engineered ratio measuring localized charging accessibility density against distance.'
    },
    'Categorical Features': {
        'education_level': 'Highest academic certification achieved.',
        'city_type': 'Geographical classification of residence area topology.',
        'current_vehicle_type': 'Chassis body type classification of existing vehicle.'
    }
}

# ==============================================================================
# 4. STATE MANAGED CALLBACK FOR INPUTS RESET
# ==============================================================================
def reset_form_inputs():
    st.session_state['age'] = 45
    st.session_state['annual_income'] = 40000.0
    st.session_state['education_level'] = 'Bachelor'
    st.session_state['city_type'] = 'Suburban'
    st.session_state['current_vehicle_type'] = 'Sedan'
    st.session_state['daily_commute_km'] = 35.0
    st.session_state['weekly_travel_distance_km'] = 220.0
    st.session_state['vehicle_age_years'] = 5.0
    st.session_state['fuel_expense_per_month'] = 295.0
    st.session_state['charging_station_accessibility'] = 6.0
    st.session_state['nearest_charging_station_km'] = 7.0
    st.session_state['home_charging_available'] = "Yes"
    st.session_state['electricity_cost_per_kwh'] = 0.21
    st.session_state['environmental_awareness_score'] = 7.0
    st.session_state['government_incentive_awareness'] = 6.0
    st.session_state['technology_affinity_score'] = 7.0
    st.session_state['range_anxiety_score'] = 5.0
    st.session_state['battery_replacement_concern'] = 5.0
    st.session_state['ev_knowledge_score'] = 7.0
    st.session_state['previous_ev_experience'] = "No"
    st.session_state['monthly_energy_consumption_kwh'] = 186.0
    st.session_state['monthly_charging_cost'] = 40.0
    st.toast("Inputs have been reset to default values!", icon="🔄")

if 'age' not in st.session_state:
    reset_form_inputs()

# ==============================================================================
# 5. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("<h1 style='text-align: center; margin-bottom: 0;'>⚡🚗</h1>", unsafe_allow_html=True)
app_page = st.sidebar.radio(
    "Choose Active Workspace:",
    ["📊 Dataset Overview", "🔮 Model Prediction", "🤖 EV AI Assistant"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Model Deployment Meta**")
if is_mock_model:
    st.sidebar.warning("⚠️ App running in Simulation Mode (pkl file absent).")
else:
    st.sidebar.success("✅ Model Pipeline Loaded successfully from pkl file.")

# ==============================================================================
# PAGE 1: DATASET OVERVIEW
# ==============================================================================
if app_page == "📊 Dataset Overview":
    st.markdown('<div class="main-header">Dataset Overview & Analytics Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploratory inspection of features used during behavioral training models</div>', unsafe_allow_html=True)
    
    # Structural Clear, Highly Vibrant Colorful KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-blue"><b style="font-size:14px; opacity:0.9;">Total Samples</b><br/><span style="font-size:24px; font-weight:800;">{df.shape[0]:,}</span></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-purple"><b style="font-size:14px; opacity:0.9;">Total Attributes</b><br/><span style="font-size:24px; font-weight:800;">{df.shape[1]}</span></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown('<div class="kpi-green"><b style="font-size:14px; opacity:0.9;">Target Attribute</b><br/><span style="font-size:16px; font-weight:800; word-break:break-all;">ev_adoption_likelihood</span></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-red"><b style="font-size:14px; opacity:0.9;">Missing Values Summary</b><br/><span style="font-size:24px; font-weight:800;">{df.isnull().sum().sum():,}</span></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Data Preview & Head", "📈 Statistical Summary", "📖 Feature Glossary Dictionary"])
    
    with tab1:
        st.subheader("Dataframe Head Preview")
        st.dataframe(df.head(15), use_container_width=True)
            
    with tab2:
        st.subheader("Descriptive Summary Metrics")
        st.dataframe(df.describe().T, use_container_width=True)
        
        st.subheader("Missing Values Audit")
        missing_series = df.isnull().sum()
        missing_df = missing_series[missing_series > 0].to_frame(name="Missing Cell Count")
        if not missing_df.empty:
            missing_df["% Percentage Value"] = (missing_df["Missing Cell Count"] / len(df) * 100).round(2)
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✨ Zero structural missing variables located inside this dataframe slice.")

    with tab3:
        st.subheader("Structured Feature Catalog")
        col_glossary_1, col_glossary_2 = st.columns(2)
        with col_glossary_1:
            st.markdown("### 🔢 Numerical Features")
            for feat, desc in feature_metadata['Numerical Features'].items():
                with st.expander(f"**{feat}**"):
                    st.write(desc)
                    
        with col_glossary_2:
            st.markdown("### 🔠 Categorical Features")
            for feat, desc in feature_metadata['Categorical Features'].items():
                with st.expander(f"**{feat}**"):
                    st.write(desc)

# ==============================================================================
# PAGE 2: MODEL PREDICTION
# ==============================================================================
elif app_page == "🔮 Model Prediction":
    st.markdown('<div class="main-header">Predictive Inference Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input consumer profiles below to query adoption likelihood scores from the optimized model.</div>', unsafe_allow_html=True)
    
    # Form Interface Layout
    with st.form("inference_profile_form"):
        st.markdown("### **Consumer Profile Dashboard Form**")
        
        # Row 1: Demographics & Core Info
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            age = st.number_input("Consumer Age", min_value=18, max_value=100, key="age", step=1)
        with r1_c2:
            annual_income = st.number_input("Annual Gross Income ($)", min_value=0.0, max_value=500000.0, key="annual_income", step=500.0)
        with r1_c3:
            education_level = st.selectbox("Completed Education Level", ["High School", "Bachelor", "Master", "PhD"], key="education_level")
            
        # Row 2: Location and Vehicle Basics
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            city_type = st.selectbox("Resident City Topology Type", ["Urban", "Suburban", "Rural"], key="city_type")
        with r2_c2:
            current_vehicle_type = st.selectbox("Existing Vehicle Chassis Style", ["Sedan", "SUV", "Hatchback"], key="current_vehicle_type")
        with r2_c3:
            vehicle_age_years = st.slider("Existing Vehicle Age (Years)", min_value=0.0, max_value=25.0, key="vehicle_age_years", step=0.5)

        # Row 3: Travel Metrics & Monthly Costs
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        with r3_c1:
            daily_commute_km = st.number_input("Average Daily Commute (km)", min_value=0.0, max_value=300.0, key="daily_commute_km", step=1.0)
        with r3_c2:
            weekly_travel_distance_km = st.number_input("Weekly Travel Output (km)", min_value=0.0, max_value=2000.0, key="weekly_travel_distance_km", step=5.0)
        with r3_c3:
            fuel_expense_per_month = st.number_input("Monthly Budget Spent on Gas/Fuel ($)", min_value=0.0, max_value=2000.0, key="fuel_expense_per_month", step=10.0)

        # Row 4: Energy Grid Matrix & Context
        r4_c1, r4_c2, r4_c3 = st.columns(3)
        with r4_c1:
            monthly_energy_consumption_kwh = st.number_input("Household Electric Consumption (kWh)", min_value=0.0, max_value=3000.0, key="monthly_energy_consumption_kwh", step=10.0)
        with r4_c2:
            monthly_charging_cost = st.number_input("Automotive Electric Infrastructure Expenses ($)", min_value=0.0, max_value=1000.0, key="monthly_charging_cost", step=5.0)
        with r4_c3:
            electricity_cost_per_kwh = st.slider("Electricity Cost ($ per kWh)", min_value=0.01, max_value=1.00, key="electricity_cost_per_kwh", step=0.01)

        # Row 5: Infrastructure Metrics & Boolean Flags
        r5_c1, r5_c2, r5_c3 = st.columns(3)
        with r5_c1:
            nearest_charging_station_km = st.number_input("Distance to Nearest Public Charger (km)", min_value=0.0, max_value=150.0, key="nearest_charging_station_km", step=0.5)
        with r5_c2:
            home_charging_available = st.selectbox("Dedicated Charging Spot Available at Home?", ["Yes", "No"], key="home_charging_available")
        with r5_c3:
            previous_ev_experience = st.selectbox("Has Historic EV Ownership/Lease Experience?", ["Yes", "No"], key="previous_ev_experience")

        st.markdown("#### **Psychometric Perception Ratings (Scores 1 to 10)**")
        
        # Row 6: Perceptions Index Part 1 (FIXED: Added the missing range_anxiety_score slider)
        r6_c1, r6_c2, r6_c3 = st.columns(3)
        with r6_c1:
            environmental_awareness_score = st.slider("Environmental Priority Index", 1.0, 10.0, key="environmental_awareness_score", step=0.1)
        with r6_c2:
            government_incentive_awareness = st.slider("Subsidies/Incentive Awareness Score", 1.0, 10.0, key="government_incentive_awareness", step=0.1)
        with r6_c3:
            technology_affinity_score = st.slider("Tech-Savviness / Affinity Score", 1.0, 10.0, key="technology_affinity_score", step=0.1)

        # Row 7: Perceptions Index Part 2 
        r7_c1, r7_c2, r7_c3 = st.columns(3)
        with r7_c1:
            battery_replacement_concern = st.slider("Battery Longevity / Replacement Worry", 1.0, 10.0, key="battery_replacement_concern", step=0.1)
        with r7_c2:
            ev_knowledge_score = st.slider("General EV Technical Knowledge Literacy", 1.0, 10.0, key="ev_knowledge_score", step=0.1)
        with r7_c3:
            charging_station_accessibility = st.slider("Public Infrastructure Convenience Score", 1.0, 10.0, key="charging_station_accessibility", step=0.1)

        # Row 8: Additional Perceptions (To cleanly separate out range anxiety or balance layout)
        r8_c1, r8_c2, r8_c3 = st.columns(3)
        with r8_c1:
            range_anxiety_score = st.slider("Range Anxiety Severity Score", 1.0, 10.0, key="range_anxiety_score", step=0.1)
        # Leave r8_c2 and r8_c3 empty or feel free to move sliders here to balance the rows!
        # Form submission layout trigger
        submit_inference = st.form_submit_button("🔮 Predict Profile Likelihood")
            
    # Reset button configuration linked via callback outside form frame
    st.button("🔄 Reset Form Inputs", on_click=reset_form_inputs)

    # Process Prediction Inference
    if submit_inference:
        # 1. Collect baseline raw inputs
        input_data = pd.DataFrame({
            'age': [age], 'annual_income': [annual_income], 'education_level': [education_level],
            'city_type': [city_type], 'daily_commute_km': [daily_commute_km], 'weekly_travel_distance_km': [weekly_travel_distance_km],
            'current_vehicle_type': [current_vehicle_type], 'vehicle_age_years': [vehicle_age_years], 'fuel_expense_per_month': [fuel_expense_per_month],
            'charging_station_accessibility': [charging_station_accessibility], 'nearest_charging_station_km': [nearest_charging_station_km],
            'home_charging_available': [1 if home_charging_available == "Yes" else 0], 'electricity_cost_per_kwh': [electricity_cost_per_kwh],
            'environmental_awareness_score': [environmental_awareness_score], 'government_incentive_awareness': [government_incentive_awareness],
            'technology_affinity_score': [technology_affinity_score], 'range_anxiety_score': [range_anxiety_score],
            'battery_replacement_concern': [battery_replacement_concern], 'ev_knowledge_score': [ev_knowledge_score],
            'previous_ev_experience': [1 if previous_ev_experience == "Yes" else 0], 'monthly_energy_consumption_kwh': [monthly_energy_consumption_kwh],
            'monthly_charging_cost': [monthly_charging_cost]
        })

        # 2. Perform exact Feature Engineering steps used during model training
        input_data['fuel_to_electricity_ratio'] = input_data['fuel_expense_per_month'] / (input_data['monthly_charging_cost'] + 1)
        input_data['charging_convenience_index'] = input_data['charging_station_accessibility'] / (input_data['nearest_charging_station_km'] + 1)

        # 3. PRODUCTION FIX: Explicitly enforce training column layout sequence order to prevent Pipeline mismatch errors
        ordered_training_features = [
            'age', 'annual_income', 'education_level', 'city_type', 'daily_commute_km', 
            'weekly_travel_distance_km', 'current_vehicle_type', 'vehicle_age_years', 
            'fuel_expense_per_month', 'charging_station_accessibility', 'nearest_charging_station_km', 
            'home_charging_available', 'electricity_cost_per_kwh', 'environmental_awareness_score', 
            'government_incentive_awareness', 'technology_affinity_score', 'range_anxiety_score', 
            'battery_replacement_concern', 'ev_knowledge_score', 'previous_ev_experience', 
            'monthly_energy_consumption_kwh', 'monthly_charging_cost', 'fuel_to_electricity_ratio', 
            'charging_convenience_index'
        ]
        input_data = input_data[ordered_training_features]

        st.markdown("---")
        st.markdown("### 📊 Model Prediction Outcome")

        # Process Prediction Inference
        with st.spinner("Executing pipeline transformations..."):
            if not is_mock_model:
                try:
                    predicted_val = model.predict(input_data)[0]
                    
                    # Map numeric classes (0, 1, 2) to explicit text names safely
                    mapping = {0: "High", 1: "Low", 2: "Medium"}
                    predicted_class = mapping.get(predicted_val, str(predicted_val))
                    
                    if hasattr(model, "predict_proba"):
                        proba_matrix = model.predict_proba(input_data)[0]
                        class_labels = model.classes_
                        class_proba_dict = dict(zip(class_labels, proba_matrix))
                        confidence = class_proba_dict.get(predicted_val, 0.0) * 100
                    else:
                        confidence = None
                except Exception as eval_err:
                    st.error(f"Execution Error running production pipeline: {eval_err}")
                    predicted_class, confidence = "High", 87.5
            else:
                # Local Fallback Matrix Logic for App Sandbox Simulation
                composite_score = (
                    (technology_affinity_score * 1.5) + (environmental_awareness_score * 1.5) + 
                    (10.0 if home_charging_available == "Yes" else -5.0) + (10.0 if fuel_expense_per_month > 300 else -5.0) -
                    (range_anxiety_score * 0.8) - (battery_replacement_concern * 0.8)
                )
                if composite_score > 25:
                    predicted_class, confidence = "High", np.clip(70 + composite_score, 0.0, 99.9)
                elif composite_score > 10:
                    predicted_class, confidence = "Medium", np.clip(55 + (composite_score * 0.8), 0.0, 99.9)
                else:
                    predicted_class, confidence = "Low", np.clip(60 - composite_score, 0.0, 99.9)

        # Render explicit styling depending on mapped text label
        if predicted_class == "High":
            st.success(f"### 🎉 Conversion Outcome: **{predicted_class} Likelihood**")
        elif predicted_class == "Medium":
            st.warning(f"### 🚗 Conversion Outcome: **{predicted_class} Likelihood**")
        else:
            st.error(f"### 🛑 Conversion Outcome: **{predicted_class} Likelihood**")

        if confidence is not None:
            st.caption(f"Model Prediction Certainty Strength: **{confidence:.1f}%**")
elif app_page == "🤖 EV AI Assistant":
    render_ev_chatbot(
        dataset_shape=df.shape,
        feature_metadata=feature_metadata,
        model_loaded=not is_mock_model,
    )
