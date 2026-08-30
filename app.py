import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 India Housing Price Prediction App")
st.write("Predict property prices in Lakhs (INR) using dataset features.")

DATA_FILE = "cleaned_india_housing_prices.csv"
MODEL_FILE = "xgb_housing_model.json"
LOCALITY_MAP_FILE = "locality_mapping.csv"


# ─────────────────────────────────────────────────────────────────────────
# Load the pretrained model (no retraining — the model file already exists)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file '{MODEL_FILE}' not found in the repo.")
        st.stop()
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    return model


@st.cache_data
def load_reference_data():
    """Loads the raw dataset (for dropdown options) and the locality name mapping."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset file '{DATA_FILE}' not found in the repo.")
        st.stop()
    if not os.path.exists(LOCALITY_MAP_FILE):
        st.error(f"Locality mapping file '{LOCALITY_MAP_FILE}' not found in the repo.")
        st.stop()

    raw_df = pd.read_csv(DATA_FILE)
    locality_map = pd.read_csv(LOCALITY_MAP_FILE)
    locality_map = locality_map[['City', 'Original_Locality_ID', 'Locality']].drop_duplicates()
    return raw_df, locality_map


model = load_model()
feature_names = model.get_booster().feature_names
raw_df, locality_map = load_reference_data()

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"**Total Model Features:** {len(feature_names)}")
st.sidebar.write(f"**Reference Records:** {len(raw_df):,}")

# ─────────────────────────────────────────────────────────────────────────
# UI Inputs
# ─────────────────────────────────────────────────────────────────────────
st.subheader("📋 Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    state_list = sorted(raw_df['State'].unique().tolist())
    selected_state = st.selectbox("State", state_list)

    # Filter cities based on selected state
    available_cities = sorted(raw_df[raw_df['State'] == selected_state]['City'].unique().tolist())
    selected_city = st.selectbox("City", available_cities)

    # Show REAL locality names in the UI, map back to the original Locality_### id
    city_locality_map = locality_map[locality_map['City'] == selected_city]
    available_localities = sorted(city_locality_map['Locality'].dropna().unique().tolist())
    selected_locality_name = st.selectbox("Locality", available_localities)

    matching_ids = city_locality_map.loc[
        city_locality_map['Locality'] == selected_locality_name, 'Original_Locality_ID'
    ].tolist()
    selected_locality = matching_ids[0] if matching_ids else None

    property_type = st.selectbox("Property Type", sorted(raw_df['Property_Type'].unique().tolist()))

with col2:
    bhk = st.number_input("BHK", min_value=1, max_value=5, value=3)
    size_sqft = st.number_input("Size (SqFt)", min_value=500, max_value=5000, value=1500, step=50)
    price_per_sqft = st.number_input("Price per SqFt (Lakhs/SqFt)", min_value=0.01, max_value=1.0, value=0.10, step=0.01, format="%.2f")
    year_built = st.number_input("Year Built", min_value=1990, max_value=2023, value=2015)
    floor_no = st.number_input("Floor Number", min_value=0, max_value=30, value=5)

with col3:
    total_floors = st.number_input("Total Floors", min_value=1, max_value=30, value=10)
    age_of_property = st.number_input("Age of Property (Years)", min_value=0, max_value=35, value=8)
    nearby_schools = st.slider("Nearby Schools Count", min_value=1, max_value=10, value=5)
    nearby_hospitals = st.slider("Nearby Hospitals Count", min_value=1, max_value=10, value=5)

st.markdown("---")
st.subheader("🛠️ Specifications & Amenities")

col_a, col_b, col_c = st.columns(3)

with col_a:
    furnished_status = st.selectbox("Furnished Status", sorted(raw_df['Furnished_Status'].unique().tolist()))
    transport_acc = st.selectbox("Public Transport Accessibility", sorted(raw_df['Public_Transport_Accessibility'].unique().tolist()))
    facing = st.selectbox("Facing Direction", sorted(raw_df['Facing'].unique().tolist()))

with col_b:
    owner_type = st.selectbox("Owner Type", sorted(raw_df['Owner_Type'].unique().tolist()))
    availability_status = st.selectbox("Availability Status", sorted(raw_df['Availability_Status'].unique().tolist()))
    parking_space = st.selectbox("Parking Space Available?", sorted(raw_df['Parking_Space'].unique().tolist()))

with col_c:
    security = st.selectbox("24/7 Security?", sorted(raw_df['Security'].unique().tolist()))
    st.write("**Amenities Available:**")
    has_garden = st.checkbox("Garden")
    has_playground = st.checkbox("Playground")
    has_clubhouse = st.checkbox("Clubhouse")
    has_gym = st.checkbox("Gym")
    has_pool = st.checkbox("Pool")


# ─────────────────────────────────────────────────────────────────────────
# Feature Vector Construction
# ─────────────────────────────────────────────────────────────────────────
def construct_input_dataframe():
    input_dict = {feat: 0.0 for feat in feature_names}

    # Numeric features
    input_dict['BHK'] = float(bhk)
    input_dict['Size_in_SqFt'] = float(size_sqft)
    input_dict['Price_per_SqFt'] = float(price_per_sqft)
    input_dict['Year_Built'] = float(year_built)
    input_dict['Floor_No'] = float(floor_no)
    input_dict['Total_Floors'] = float(total_floors)
    input_dict['Age_of_Property'] = float(age_of_property)
    input_dict['Nearby_Schools'] = float(nearby_schools)
    input_dict['Nearby_Hospitals'] = float(nearby_hospitals)

    # Amenities
    input_dict['Amenity_Garden'] = 1.0 if has_garden else 0.0
    input_dict['Amenity_Playground'] = 1.0 if has_playground else 0.0
    input_dict['Amenity_Clubhouse'] = 1.0 if has_clubhouse else 0.0
    input_dict['Amenity_Gym'] = 1.0 if has_gym else 0.0
    input_dict['Amenity_Pool'] = 1.0 if has_pool else 0.0

    # Categorical one-hot columns — only set the ones the model actually has
    cat_mappings = [
        f"State_{selected_state}",
        f"City_{selected_city}",
        f"Locality_{selected_locality}",
        f"Property_Type_{property_type}",
        f"Furnished_Status_{furnished_status}",
        f"Public_Transport_Accessibility_{transport_acc}",
        f"Parking_Space_{parking_space}",
        f"Security_{security}",
        f"Facing_{facing}",
        f"Owner_Type_{owner_type}",
        f"Availability_Status_{availability_status}",
    ]

    unmatched = [key for key in cat_mappings if key not in input_dict]
    if unmatched:
        st.warning(
            "These selections don't match any column the model was trained on, "
            f"so they'll be ignored: {unmatched}"
        )

    for key in cat_mappings:
        if key in input_dict:
            input_dict[key] = 1.0

    return pd.DataFrame([input_dict], columns=feature_names)


# ─────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("Predict Property Price 🚀", type="primary", use_container_width=True):
    input_df = construct_input_dataframe()
    predicted_price = model.predict(input_df)[0]

    st.subheader("🎯 Estimated Price Result")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Estimated Price (Lakhs)", value=f"₹ {predicted_price:,.2f} Lakhs")
    with col_res2:
        st.metric(label="Estimated Price (INR)", value=f"₹ {predicted_price * 100000:,.0f}")
