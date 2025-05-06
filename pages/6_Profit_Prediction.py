from typing import Any, List, Dict
import streamlit as st
import pandas as pd
import plotly.express as px
from fredapi import Fred
import pickle
import urllib.request
import tempfile
import os
import numpy as np

st.set_page_config(page_title="Profit Prediction Model", layout="wide")

st.markdown("<h1 style='text-align: center;'>Profit Prediction Model</h1>", unsafe_allow_html=True)

# --- Constants ---
GCS_PUBLIC_MODEL_URL = "https://storage.googleapis.com/adta5410-public-bucket/final_model.pkl"
LOCAL_DATA_PATH = './pages/clean_retail_data.csv'  # Path to your dataset

# Assuming your FRED API key is stored in Streamlit secrets
FRED_API_KEY = st.secrets.get("FED_API") # Use .get() for safer access

# Define the exact order of features the model expects for prediction
# THIS IS CRUCIAL - ENSURE THIS MATCHES YOUR MODEL'S TRAINING DATA COLUMN ORDER
MODEL_FEATURE_ORDER = [
    "Price", "Units Ordered", "Advertising Spend",
    "Managed Stock Level", "DI Last Item", "PPI Last Item",
    "Profit Margin", "Average Pricing", "Average Competitor Pricing"
]

# Define which of these features are suitable for varying in the plot
FEATURES_TO_VARY = [
    "Price", "Units Ordered", "Advertising Spend",
    "Managed Stock Level", "Profit Margin",
    "Average Pricing", "Average Competitor Pricing"
]

@st.cache_data()
def load_dataset(data_path: str) -> pd.DataFrame:
    """Loads the dataset from a specified path."""
    try:
        df = pd.read_csv(data_path)
        # st.success("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {data_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()


@st.cache_resource
def load_model_from_gcs(model_url: str) -> Any | None:
    """Downloads a pickle model from a public GCS URL and loads it."""

    # st.info(f"Attempting to download model from {model_url}...")
    local_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            local_file_path = tmp_file.name

        urllib.request.urlretrieve(model_url, local_file_path)
        # st.success(f"Model downloaded to temporary file: {local_file_path}")

        # st.info("Loading model...")
        with open(local_file_path, 'rb') as f:
            model = pickle.load(f)
        # st.success("Model loaded successfully!")

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    finally:
        if local_file_path and os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
                # st.info(f"Cleaned up temporary file: {local_file_path}")
            except Exception as e:
                st.warning(f"Could not clean up temporary file {local_file_path}: {e}")


@st.cache_data()
def get_fred_indices(api_key: str | None) -> tuple[float, float] | None:
    """Fetches the latest FRED indices (PPI and DI). Returns (ppi, di)."""
    if not api_key:
        st.error("FRED API key not found in Streamlit secrets.")
        return None

   #  st.info("Fetching FRED indices (PPI and DI)...")
    try:
        fred = Fred(api_key=api_key)
        ppi_series = fred.get_series('WPSFD4111')
        di_series = fred.get_series('DTWEXBGS')

        ppi_last = ppi_series.dropna().iloc[-1] if not ppi_series.dropna().empty else None
        di_last = di_series.dropna().iloc[-1] if not di_series.dropna().empty else None

        if ppi_last is None or di_last is None:
             st.error("Could not fetch the latest FRED index data (PPI or DI is missing).")
             return None

        # st.success("FRED indices fetched successfully.")
        return float(ppi_last), float(di_last)

    except Exception as e:
        st.error(f"Error fetching FRED indices: {e}")
        return None

# --- Helper Function to Get Default Values ---
def get_default_values_for_sku(sku: str, df: pd.DataFrame) -> Dict[str, float]:
    """Fetch default values for the selected SKU from the dataset."""
    sku_row = df[df['sku'] == sku]
    if sku_row.empty:
        return {}
    return {
        "Price": sku_row["price"].iloc[0],
        "Units Ordered": sku_row["units_ordered"].iloc[0],
        "Advertising Spend": sku_row["advertising_spend"].iloc[0],
        "Managed Stock Level": sku_row["managed_stock_level"].iloc[0],
        "Profit Margin": sku_row["profit_margin"].iloc[0],
        "Average Pricing": sku_row["average_pricing"].iloc[0],
        "Average Competitor Pricing": sku_row["average_competitor_pricing"].iloc[0]
    }

    # --- Restore Defaults Button Logic ---
    if selected_sku:
        default_values = get_default_values_for_sku(selected_sku, df)

        if default_values:
            if st.button("Restore Defaults"):
                for key, value in default_values.items():
                    if key in input_strings:
                        st.session_state[key] = str(value)

            # Pre-fill input fields with default values if not already set
            for key, value in default_values.items():
                if key in input_strings and key not in st.session_state:
                    st.session_state[key] = str(value)
# --- Load Data, Model, and FRED Indices (Cached) ---

df = load_dataset(LOCAL_DATA_PATH)
model = load_model_from_gcs(GCS_PUBLIC_MODEL_URL)
fred_indices = get_fred_indices(FRED_API_KEY)

if model is None:
    st.error("Prediction model could not be loaded.")
    st.stop()

if fred_indices is None:
     st.error("FRED indices needed for prediction could not be fetched.")
     st.stop()

ppi_last_item, di_last_item = fred_indices



st.sidebar.header("Product Selection")

subgroup_options = sorted(df['subgroup'].unique())
selected_subgroup = st.sidebar.selectbox(
    label="Step 1: Choose a Group",
    options=subgroup_options
)

subgroup_df = df[df['subgroup'] == selected_subgroup].copy()

sku_options = sorted(subgroup_df['sku'].unique())

if not sku_options:
    st.warning(f"No SKUs found for Subgroup '{selected_subgroup}'. Please select a different subgroup.")
    st.sidebar.warning("No SKUs available.")
    selected_sku = None
else:
    selected_sku = st.sidebar.selectbox(
        label="Step 2: Choose a SKU",
        options=sku_options
    )


st.subheader("Select a Product and Enter Financial Metrics for Single Prediction")

# Collect USER INPUT values as strings
input_strings: Dict[str, str] = {
    "Price": st.text_input("Step 3: Enter Price"),
    "Units Ordered": st.text_input("Step 4: Enter Units"),
    "Advertising Spend": st.text_input("Step 5: Enter Advertising Spend"),
    "Managed Stock Level": st.text_input("Step 6: Enter Managed Stock Level"),
    "Profit Margin": st.text_input("Step 7: Enter Profit Margin"),
    "Average Pricing": st.text_input("Step 8: Enter Average Pricing"),
    "Average Competitor Pricing": st.text_input("Step 9: Enter Average Competitor Pricing")
}

# Display the fetched FRED indices
# st.subheader("Current Economic Indices (Fetched from FRED)")
# st.write(f"Latest Producer Price Index (PPI): `{ppi_last_item:.4f}`")
# st.write(f"Latest Broad Trade Weighted US Dollar Index (DI): `{di_last_item:.4f}`")


# Add a button for the single prediction
predict_button = st.button("Predict Single Profit")


# --- Single Prediction Logic ---

if predict_button and selected_sku is not None:
    st.subheader("Single Prediction Result")

    user_input_numeric: List[float] = []
    all_user_inputs_valid = True
    error_messages: List[str] = []

    text_input_order_keys = [
        "Price", "Units Ordered", "Advertising Spend",
        "Managed Stock Level", "Profit Margin",
        "Average Pricing", "Average Competitor Pricing"
    ]

    for key in text_input_order_keys:
        value_str = input_strings.get(key, "")

        if value_str == "":
            all_user_inputs_valid = False
            error_messages.append(f"'{key}' is required.")
            continue

        try:
            value_numeric = float(value_str)
            user_input_numeric.append(value_numeric)
        except ValueError:
            all_user_inputs_valid = False
            error_messages.append(f"'{key}' must be a valid number.")
            continue

    if all_user_inputs_valid:
        try:
            final_data_to_predict = []
            text_input_index = 0

            for feature_name in MODEL_FEATURE_ORDER:
                if feature_name == "DI Last Item":
                     final_data_to_predict.append(di_last_item)
                elif feature_name == "PPI Last Item":
                     final_data_to_predict.append(ppi_last_item)
                elif feature_name in text_input_order_keys:
                     final_data_to_predict.append(user_input_numeric[text_input_index])
                     text_input_index += 1

            input_array = np.array([final_data_to_predict])
            prediction = model.predict(input_array)

            st.success("Prediction successful!")
            if isinstance(prediction, (list, np.ndarray)):
                 predicted_value = prediction[0] if len(prediction) > 0 else "N/A"
                 if isinstance(predicted_value, (int, float)):
                     st.metric(label="Predicted Profit", value=f"${predicted_value:.2f}")
                 else:
                      st.write(f"Raw Prediction Output: {prediction}")
            else:
                 st.write(f"Raw Prediction Output: {prediction}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please check your inputs and ensure the model expects the features in the correct order.")

    else:
        st.warning("Please fix the following issues before predicting:")
        for msg in error_messages:
            st.write(f"- {msg}")

elif predict_button and selected_sku is None:
     st.warning("Please select a SKU first.")


# --- Prediction Range Visualization ---

st.subheader("Analyze Profit Across a Range")

current_user_inputs_valid = True
current_user_input_numeric_values: List[float] = []

text_input_order_keys = [
    "Price", "Units Ordered", "Advertising Spend",
    "Managed Stock Level", "Profit Margin",
    "Average Pricing", "Average Competitor Pricing"
]

for key in text_input_order_keys:
    value_str = input_strings.get(key, "")
    if value_str == "":
        current_user_inputs_valid = False
        break
    try:
        value_numeric = float(value_str)
        current_user_input_numeric_values.append(value_numeric)
    except ValueError:
        current_user_inputs_valid = False
        break


if not current_user_inputs_valid or selected_sku is None:
    st.info("Please select a SKU and enter valid numbers for all financial metrics above to enable the range analysis.")
else:
    # --- Range Selection Widgets ---
    st.write("Select a feature to vary and define the range.")

    feature_to_vary = st.selectbox(
        label="Select Feature to Vary",
        options=FEATURES_TO_VARY
    )

    # Find the current value of the selected feature to help set default range
    try:
        text_input_idx_to_vary = text_input_order_keys.index(feature_to_vary)
        current_value_of_feature_to_vary = current_user_input_numeric_values[text_input_idx_to_vary]

        default_min = current_value_of_feature_to_vary * 0.5
        default_max = current_value_of_feature_to_vary * 1.5

        if current_value_of_feature_to_vary <= 0:
             default_min = 0
             default_max = max(10.0, current_value_of_feature_to_vary + 10.0)

        if default_min > default_max:
             default_min, default_max = default_max, default_min

        default_min = float(f"{default_min:.2f}")
        default_max = float(f"{default_max:.2f}")

    except (ValueError, IndexError):
         current_value_of_feature_to_vary = 0.0
         default_min = 0.0
         default_max = 100.0


    range_min = st.number_input(
        f"Minimum value for {feature_to_vary}",
        value=default_min # Use cleaned default
    )

    range_max = st.number_input(
        f"Maximum value for {feature_to_vary}",
         value=default_max # Use cleaned default
    )

    num_steps = st.slider(
        "Number of steps in range",
        min_value=20, max_value=200, value=50, step=10
    )

    plot_button = st.button(f"Generate Plot for {feature_to_vary} Range")

    # --- Range Plot Generation Logic ---
    if plot_button:
        if range_min >= range_max:
            st.warning("Minimum value must be less than Maximum value for the range plot.")
        else:
            st.info(f"Generating predictions for {feature_to_vary} from {range_min} to {range_max} in {num_steps} steps...")

            try:
                varying_values = np.linspace(range_min, range_max, num_steps)

                plot_data: List[Dict[str, float]] = []

                feature_vary_index = MODEL_FEATURE_ORDER.index(feature_to_vary)

                base_data_point: List[float] = []
                current_text_input_index = 0

                for feature_name in MODEL_FEATURE_ORDER:
                    if feature_name == "DI Last Item":
                         base_data_point.append(di_last_item)
                    elif feature_name == "PPI Last Item":
                         base_data_point.append(ppi_last_item)
                    elif feature_name in text_input_order_keys:
                        base_data_point.append(current_user_input_numeric_values[current_text_input_index])
                        current_text_input_index += 1
                    # No else needed here

                for value in varying_values:
                    data_point_for_prediction = base_data_point.copy()
                    data_point_for_prediction[feature_vary_index] = value

                    input_array = np.array([data_point_for_prediction])
                    predicted_profit = model.predict(input_array)[0]

                    plot_data.append({
                        feature_to_vary: value,
                        "Predicted Profit": predicted_profit
                    })

                plot_df = pd.DataFrame(plot_data)

                chart = px.line(
                    plot_df,
                    x=feature_to_vary,
                    y="Predicted Profit",
                    title=f"Predicted Profit vs. {feature_to_vary}",
                    labels={"Predicted Profit": "Predicted Profit ($)"} # Optional: improve label
                )

                st.plotly_chart(chart, use_container_width=True)
                st.success("Plot generated!")

            except Exception as e:
                st.error(f"An error occurred while generating the plot: {e}")
                st.error("Please check the range values and model compatibility.")

# --- Optional: Display selected SKU details ---
if selected_sku:
    st.sidebar.subheader(f"Selected SKU: {selected_sku}")
    # Add SKU details display here if desired
    pass