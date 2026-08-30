# 🏠 Indian House Price Predictor

A machine learning web app that predicts residential property prices across India using an **XGBoost regression model**, deployed with **Streamlit**.

🔗 **Live App:** [Add your Streamlit Cloud link here](#)

---

## 📌 Overview

This project estimates the price of a house/apartment based on property details such as size, location, amenities, and construction info. The model was trained on a housing dataset covering **42 cities and 20 states across India**, and is served through a simple, interactive Streamlit interface — just fill in the property details and get an instant price prediction.

---

## ✨ Features

- Predicts property price from real-world housing attributes
- Covers 42+ Indian cities across 20 states, with 500 localities
- Accounts for amenities (garden, gym, pool, clubhouse, playground)
- Considers property type, furnishing, parking, security, and facing direction
- Simple, fast, browser-based UI — no installation needed for end users

---

## 🧠 Model Details

| | |
|---|---|
| **Algorithm** | XGBoost Regressor (`XGBRegressor`) |
| **Objective** | `reg:squarederror` |
| **Trees** | 100 |
| **Input features** | 598 (after one-hot encoding) |
| **Format** | XGBoost native JSON (`xgb_housing_model.json`) |

### Input features used

**Numeric / core**
`BHK`, `Size_in_SqFt`, `Price_per_SqFt`, `Year_Built`, `Floor_No`, `Total_Floors`, `Age_of_Property`, `Nearby_Schools`, `Nearby_Hospitals`

**Categorical (one-hot encoded)**
`Property_Type`, `Furnished_Status`, `Public_Transport_Accessibility`, `Parking_Space`, `Security`, `Facing`, `Owner_Type`, `Availability_Status`, `State`, `City`, `Locality`

**Amenities (binary flags)**
`Garden`, `Playground`, `Clubhouse`, `Gym`, `Pool`

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit application
├── xgb_housing_model.json    # Trained XGBoost model
└── README.md
```

---

## ⚙️ Installation & Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rickeysharma942-stack/house-price-predictor
   cd house-price-predictor
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## 🚀 Usage

1. Open the app (locally or via the live Streamlit link).
2. Enter the property details — size, BHK, location, floor, amenities, etc.
3. Click **Predict**.
4. View the estimated house price instantly.

---

## 🛠️ Tech Stack

- **Python**
- **XGBoost** – model training & inference
- **Streamlit** – web app framework
- **Pandas / NumPy** – data handling

---

## 📊 Dataset

The model was trained on an Indian housing dataset containing property listings with details like location, size, amenities, and pricing.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Rickey Sharma**
📧 rickeysharma942@gmail.com
🔗 [GitHub](#)
