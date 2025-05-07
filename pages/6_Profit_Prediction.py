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
LOCAL_DATA_PATH = './pages/clean_retail_data.csv'

FRED_API_KEY = st.secrets.get("FED_API")

MODEL_FEATURE_ORDER = [
    "Price", "Units Ordered", "Advertising Spend",
    "Managed Stock Level", "DI Last Item", "PPI Last Item",
    "Profit Margin", "Average Pricing", "Average Competitor Pricing"
]


FEATURES_TO_VARY = [
    "Price", "Units Ordered", "Advertising Spend",
    "Managed Stock Level", "Profit Margin",
    "Average Pricing", "Average Competitor Pricing"
]

# --- List of keys for the input fields that will be linked to session state ---
INPUT_KEYS = [
    "Price",
    "Units Ordered",
    "Advertising Spend",
    "Managed Stock Level",
    "Profit Margin",
    "Average Pricing",
    "Average Competitor Pricing"
]


# --- Session State Initialization for Input Fields and SKU Tracking ---
for key in INPUT_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

if 'last_selected_sku' not in st.session_state:
    st.session_state.last_selected_sku = None


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


# --- Helper Function to Get Default Values (Your existing code, modified for rounding) ---
def get_default_values_for_sku(sku: str, df: pd.DataFrame) -> Dict[str, float]:
    """Fetch default values for the selected SKU from the dataset."""
    sku_row = df[df['sku'] == sku]
    if sku_row.empty:
        return {}
    return {
        "Price": round(sku_row["price"].iloc[0], 2),
        "Units Ordered": round(sku_row["unitsordered"].iloc[0], 2),
        "Advertising Spend": round(sku_row["adspend"].iloc[0], 2),
        "Managed Stock Level": round(sku_row["managed_fba_stock_level"].iloc[0], 0),
        "Profit Margin": round(sku_row["profit_margin"].iloc[0], 2),
        "Average Pricing": round(sku_row["avg_pricing"].iloc[0], 2),
        "Average Competitor Pricing": round(sku_row["avg_comp_pricing"].iloc[0], 2)
    }

# --- Callback function for the "Restore Defaults" button ---
def restore_defaults_callback(selected_sku: str, df: pd.DataFrame):
    """Fetches default values and updates session state."""
    default_values = get_default_values_for_sku(selected_sku, df)
    if default_values:
        for key, value in default_values.items():
            if key in INPUT_KEYS:
                # Update session state with default values (as strings)
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
        options=sku_options,
        key='selected_sku_selectbox'
    )

# --- Load Defaults into Session State when SKU Changes ---
if selected_sku and selected_sku != st.session_state.last_selected_sku:
    # SKU has changed, load new defaults into state
    default_values = get_default_values_for_sku(selected_sku, df)

    if default_values:
        for key, value in default_values.items():
            if key in INPUT_KEYS:
                 st.session_state[key] = str(value)

    st.session_state.last_selected_sku = selected_sku

elif not selected_sku and st.session_state.last_selected_sku is not None:
     for key in INPUT_KEYS:
         st.session_state[key] = None
     st.session_state.last_selected_sku = None


st.subheader("Select a Product and Enter Financial Metrics for Single Prediction")

# --- Define Input Fields (linked to Session State) ---
label_mapping = {
    "Price": "Step 3: Enter Price",
    "Units Ordered": "Step 4: Enter Units",
    "Advertising Spend": "Step 5: Enter Advertising Spend",
    "Managed Stock Level": "Step 6: Enter Managed Stock Level",
    "Profit Margin": "Step 7: Enter Profit Margin",
    "Average Pricing": "Step 8: Enter Average Pricing",
    "Average Competitor Pricing": "Step 9: Enter Average Competitor Pricing"
}

num_cols = 2
cols = st.columns(num_cols)
col_index = 0

input_strings: Dict[str, str] = {}

for key in INPUT_KEYS:
    label = label_mapping.get(key, key)

    current_value_from_state = st.session_state.get(key)
    display_value = "" if current_value_from_state is None else str(current_value_from_state)

    with cols[col_index]:
         st.text_input(
             label=label,
             value=display_value,
             key=key
         )
    input_strings[key] = str(st.session_state.get(key, ""))

    col_index = (col_index + 1) % num_cols


# --- Restore Defaults Button Logic (Using Callback) ---
if selected_sku:
     st.markdown("---")

     col1, col2 = st.columns([3, 1])

     with col2:
         st.button(
             "Restore Defaults",
             on_click=restore_defaults_callback,
             args=(selected_sku, df)
         )


# Add a button for the single prediction
predict_button = st.button("Predict Single Profit")

# --- Single Prediction Logic ---
if predict_button and selected_sku is not None:
    st.subheader("Single Prediction Result")

    user_input_numeric: List[float] = []
    all_user_inputs_valid = True
    error_messages: List[str] = []

    for key in INPUT_KEYS:
        value_str = str(st.session_state.get(key, "")) # Ensure it's a string

        if value_str == "":
            all_user_inputs_valid = False
            error_messages.append(f"'{label_mapping.get(key, key)}' is required.")
            continue

        try:
            value_numeric = float(value_str)
            user_input_numeric.append(value_numeric)
        except ValueError:
            all_user_inputs_valid = False
            error_messages.append(f"'{label_mapping.get(key, key)}' must be a valid number.")
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
                elif feature_name in INPUT_KEYS:
                     try:
                         input_key_index = INPUT_KEYS.index(feature_name)
                         final_data_to_predict.append(user_input_numeric[input_key_index])
                     except ValueError:
                         st.error(f"Internal Error: Feature '{feature_name}' from MODEL_FEATURE_ORDER not found in INPUT_KEYS.")
                         st.stop()

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
current_user_input_numeric_values: Dict[str, float] = {} # Use a dict for easier lookup

for key in INPUT_KEYS:
    value_str = str(st.session_state.get(key, ""))

    if value_str == "":
        current_user_inputs_valid = False
        break
    try:
        value_numeric = float(value_str)
        current_user_input_numeric_values[key] = value_numeric
    except ValueError:
        current_user_inputs_valid = False
        break


if not current_user_inputs_valid or selected_sku is None:
    st.info("Please select a SKU and enter valid numbers for all financial metrics above to enable the range analysis.")
else:
    st.write("Select a feature to vary and define the range.")

    feature_to_vary = st.selectbox(
        label="Select Feature to Vary",
        options=FEATURES_TO_VARY,
        key='feature_to_vary_selectbox'
    )

    current_value_of_feature_to_vary = current_user_input_numeric_values.get(feature_to_vary, 0.0)

    default_min = current_value_of_feature_to_vary * 0.5
    default_max = current_value_of_feature_to_vary * 1.5

    if current_value_of_feature_to_vary <= 0:
         default_min = 0
         default_max = max(10.0, current_value_of_feature_to_vary + 10.0)

    if default_min > default_max:
         default_min, default_max = default_max, default_min

    default_min = float(f"{default_min:.2f}")
    default_max = float(f"{default_max:.2f}")


    range_min = st.number_input(
        f"Minimum value for {feature_to_vary}",
        value=default_min,
        key=f'range_min_{feature_to_vary}'
    )

    range_max = st.number_input(
        f"Maximum value for {feature_to_vary}",
         value=default_max,
         key=f'range_max_{feature_to_vary}'
    )

    num_steps = st.slider(
        "Number of steps in range",
        min_value=20, max_value=200, value=50, step=10,
        key=f'num_steps_{feature_to_vary}'
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

                base_data_point: List[float] = []

                for feature_name in MODEL_FEATURE_ORDER:
                    if feature_name == "DI Last Item":
                         base_data_point.append(di_last_item)
                    elif feature_name == "PPI Last Item":
                         base_data_point.append(ppi_last_item)
                    elif feature_name in INPUT_KEYS:
                        base_data_point.append(current_user_input_numeric_values.get(feature_name, 0.0))

                try:
                    feature_vary_index_in_model_order = MODEL_FEATURE_ORDER.index(feature_to_vary)
                except ValueError:
                     st.error(f"Internal Error: Feature '{feature_to_vary}' not found in MODEL_FEATURE_ORDER.")
                     st.stop()


                for value in varying_values:
                    data_point_for_prediction = base_data_point.copy()
                    data_point_for_prediction[feature_vary_index_in_model_order] = value

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
                    labels={"Predicted Profit": "Predicted Profit ($)"}
                )

                st.plotly_chart(chart, use_container_width=True)
                st.success("Plot generated!")

            except Exception as e:
                st.error(f"An error occurred while generating the plot: {e}")
                st.error("Please check the range values and model compatibility.")

# --- Optional: Display selected SKU details ---
if selected_sku:
    st.sidebar.subheader(f"Selected SKU: {selected_sku}")
    pass
