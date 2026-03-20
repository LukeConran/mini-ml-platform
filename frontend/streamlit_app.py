import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict their likelihood of churning.")

# --- Personal Info ---
st.subheader("Personal Info")
c1, c2, c3, c4 = st.columns(4)
gender      = c1.selectbox("Gender", ["Male", "Female"])
senior      = c2.checkbox("Senior Citizen")
partner     = c3.checkbox("Has Partner")
dependents  = c4.checkbox("Has Dependents")

# --- Account Info ---
st.subheader("Account")
c1, c2, c3 = st.columns(3)
tenure          = c1.slider("Tenure (months)", 0, 72, 12)
contract        = c2.radio("Contract", ["Month-to-month", "One year", "Two year"])
payment_method  = c3.radio("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
])
paperless_billing = st.checkbox("Paperless Billing")

# --- Charges ---
st.subheader("Charges")
c1, c2 = st.columns(2)
monthly_charges = c1.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.01)
total_charges   = c2.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * 65), step=0.01)

# --- Services ---
st.subheader("Services")
phone_service = st.checkbox("Phone Service", value=True)

if phone_service:
    multiple_lines = st.radio("Multiple Lines", ["No", "Yes"], horizontal=True)
else:
    multiple_lines = "No phone service"
    st.caption("Multiple Lines: N/A (no phone service)")

internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"], horizontal=True)

if internet_service != "No":
    c1, c2, c3 = st.columns(3)
    online_security   = c1.radio("Online Security",   ["No", "Yes"], horizontal=True)
    online_backup     = c2.radio("Online Backup",     ["No", "Yes"], horizontal=True)
    device_protection = c3.radio("Device Protection", ["No", "Yes"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    tech_support     = c1.radio("Tech Support",     ["No", "Yes"], horizontal=True)
    streaming_tv     = c2.radio("Streaming TV",     ["No", "Yes"], horizontal=True)
    streaming_movies = c3.radio("Streaming Movies", ["No", "Yes"], horizontal=True)
else:
    online_security = online_backup = device_protection = "No internet service"
    tech_support = streaming_tv = streaming_movies = "No internet service"
    st.caption("Internet add-ons: N/A (no internet service)")


def one_hot(value, options):
    return {opt: (value == opt) for opt in options}


def build_payload():
    payload = {
        "gender":            gender == "Male",
        "SeniorCitizen":     senior,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone_service,
        "PaperlessBilling":  paperless_billing,
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
        **one_hot(multiple_lines,    ["No", "No phone service", "Yes"]),
        **one_hot(internet_service,  ["DSL", "Fiber optic", "No"]),
        **one_hot(online_security,   ["No", "No internet service", "Yes"]),
        **one_hot(online_backup,     ["No", "No internet service", "Yes"]),
        **one_hot(device_protection, ["No", "No internet service", "Yes"]),
        **one_hot(tech_support,      ["No", "No internet service", "Yes"]),
        **one_hot(streaming_tv,      ["No", "No internet service", "Yes"]),
        **one_hot(streaming_movies,  ["No", "No internet service", "Yes"]),
        **one_hot(contract,          ["Month-to-month", "One year", "Two year"]),
        **one_hot(payment_method,    [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ]),
    }
    # Rename keys to match the API's expected field names
    renames = {
        "No phone service":             "No_phone_service",
        "Fiber optic":                  "Fiber_optic",
        "No internet service":          "No_internet_service",
        "Month-to-month":               "Month_to_month",
        "One year":                     "One_year",
        "Two year":                     "Two_year",
        "Bank transfer (automatic)":    "Bank_transfer_automatic",
        "Credit card (automatic)":      "Credit_card_automatic",
        "Electronic check":             "Electronic_check",
        "Mailed check":                 "Mailed_check",
    }
    return {renames.get(k, k): v for k, v in payload.items()}


def prefix_payload(payload):
    """Prefix one-hot keys with their feature group to match the API schema."""
    prefixes = {
        "No_phone_service": "MultipleLines", "Yes": None, "No": None,
    }
    field_map = {
        # MultipleLines
        "MultipleLines_No":                 ("MultipleLines", "No"),
        "MultipleLines_No_phone_service":   ("MultipleLines", "No_phone_service"),
        "MultipleLines_Yes":                ("MultipleLines", "Yes"),
        # InternetService
        "InternetService_DSL":              ("InternetService", "DSL"),
        "InternetService_Fiber_optic":      ("InternetService", "Fiber_optic"),
        "InternetService_No":               ("InternetService", "No"),
        # OnlineSecurity
        "OnlineSecurity_No":                ("OnlineSecurity", "No"),
        "OnlineSecurity_No_internet_service": ("OnlineSecurity", "No_internet_service"),
        "OnlineSecurity_Yes":               ("OnlineSecurity", "Yes"),
        # OnlineBackup
        "OnlineBackup_No":                  ("OnlineBackup", "No"),
        "OnlineBackup_No_internet_service": ("OnlineBackup", "No_internet_service"),
        "OnlineBackup_Yes":                 ("OnlineBackup", "Yes"),
        # DeviceProtection
        "DeviceProtection_No":              ("DeviceProtection", "No"),
        "DeviceProtection_No_internet_service": ("DeviceProtection", "No_internet_service"),
        "DeviceProtection_Yes":             ("DeviceProtection", "Yes"),
        # TechSupport
        "TechSupport_No":                   ("TechSupport", "No"),
        "TechSupport_No_internet_service":  ("TechSupport", "No_internet_service"),
        "TechSupport_Yes":                  ("TechSupport", "Yes"),
        # StreamingTV
        "StreamingTV_No":                   ("StreamingTV", "No"),
        "StreamingTV_No_internet_service":  ("StreamingTV", "No_internet_service"),
        "StreamingTV_Yes":                  ("StreamingTV", "Yes"),
        # StreamingMovies
        "StreamingMovies_No":               ("StreamingMovies", "No"),
        "StreamingMovies_No_internet_service": ("StreamingMovies", "No_internet_service"),
        "StreamingMovies_Yes":              ("StreamingMovies", "Yes"),
        # Contract
        "Contract_Month_to_month":          ("Contract", "Month_to_month"),
        "Contract_One_year":                ("Contract", "One_year"),
        "Contract_Two_year":                ("Contract", "Two_year"),
        # PaymentMethod
        "PaymentMethod_Bank_transfer_automatic": ("PaymentMethod", "Bank_transfer_automatic"),
        "PaymentMethod_Credit_card_automatic":   ("PaymentMethod", "Credit_card_automatic"),
        "PaymentMethod_Electronic_check":        ("PaymentMethod", "Electronic_check"),
        "PaymentMethod_Mailed_check":            ("PaymentMethod", "Mailed_check"),
    }
    _ = prefixes  # unused, field_map drives everything
    return field_map


# Build and remap the full payload to match CustomerData field names exactly
def get_api_payload():
    raw = build_payload()
    field_map = prefix_payload(raw)

    result = {
        "gender":           raw["gender"],
        "SeniorCitizen":    raw["SeniorCitizen"],
        "Partner":          raw["Partner"],
        "Dependents":       raw["Dependents"],
        "tenure":           raw["tenure"],
        "PhoneService":     raw["PhoneService"],
        "PaperlessBilling": raw["PaperlessBilling"],
        "MonthlyCharges":   raw["MonthlyCharges"],
        "TotalCharges":     raw["TotalCharges"],
    }
    for api_key, (group, option) in field_map.items():
        source_key = option  # the unprefixed key from build_payload
        result[api_key] = raw.get(source_key, False)

    return result


# --- Predict ---
st.divider()
if st.button("Predict Churn", type="primary", use_container_width=True):
    try:
        payload = get_api_payload()
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        prob = data["probability"]
        churned = data["churn"]

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{prob:.1%}")
        col2.metric("Prediction", "Will Churn" if churned else "Will Stay")
        st.progress(prob)

        if prob >= 0.7:
            st.error("High churn risk.")
        elif prob >= 0.4:
            st.warning("Moderate churn risk.")
        else:
            st.success("Low churn risk.")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the server is running (`./run.sh`).")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            st.error("API is running but the model isn't loaded yet. Check that MLflow has a registered Production model.")
        else:
            st.error(f"API error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
