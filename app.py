import ast
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="India Real Estate Dashboard",
    page_icon="🏠",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

CITY_COORDINATES = {
    ("Andhra Pradesh", "Vijayawada"): (16.5062, 80.6480),
    ("Andhra Pradesh", "Vishakhapatnam"): (17.6868, 83.2185),
    ("Assam", "Guwahati"): (26.1445, 91.7362),
    ("Assam", "Silchar"): (24.8333, 92.7789),
    ("Bihar", "Gaya"): (24.7914, 85.0002),
    ("Bihar", "Patna"): (25.5941, 85.1376),
    ("Chhattisgarh", "Bilaspur"): (22.0797, 82.1391),
    ("Chhattisgarh", "Raipur"): (21.2514, 81.6296),
    ("Delhi", "Dwarka"): (28.5921, 77.0460),
    ("Delhi", "New Delhi"): (28.6139, 77.2090),
    ("Gujarat", "Ahmedabad"): (23.0225, 72.5714),
    ("Gujarat", "Surat"): (21.1702, 72.8311),
    ("Haryana", "Faridabad"): (28.4089, 77.3178),
    ("Haryana", "Gurgaon"): (28.4595, 77.0266),
    ("Jharkhand", "Jamshedpur"): (22.8046, 86.2029),
    ("Jharkhand", "Ranchi"): (23.3441, 85.3096),
    ("Karnataka", "Bangalore"): (12.9716, 77.5946),
    ("Karnataka", "Mangalore"): (12.9141, 74.8560),
    ("Karnataka", "Mysore"): (12.2958, 76.6394),
    ("Kerala", "Kochi"): (9.9312, 76.2673),
    ("Kerala", "Trivandrum"): (8.5241, 76.9366),
    ("Madhya Pradesh", "Bhopal"): (23.2599, 77.4126),
    ("Madhya Pradesh", "Indore"): (22.7196, 75.8577),
    ("Maharashtra", "Mumbai"): (19.0760, 72.8777),
    ("Maharashtra", "Nagpur"): (21.1458, 79.0882),
    ("Maharashtra", "Pune"): (18.5204, 73.8567),
    ("Odisha", "Bhubaneswar"): (20.2961, 85.8245),
    ("Odisha", "Cuttack"): (20.4625, 85.8830),
    ("Punjab", "Amritsar"): (31.6340, 74.8723),
    ("Punjab", "Ludhiana"): (30.9010, 75.8573),
    ("Rajasthan", "Jaipur"): (26.9124, 75.7873),
    ("Rajasthan", "Jodhpur"): (26.2389, 73.0243),
    ("Tamil Nadu", "Chennai"): (13.0827, 80.2707),
    ("Tamil Nadu", "Coimbatore"): (11.0168, 76.9558),
    ("Telangana", "Hyderabad"): (17.3850, 78.4867),
    ("Telangana", "Warangal"): (17.9689, 79.5941),
    ("Uttar Pradesh", "Lucknow"): (26.8467, 80.9462),
    ("Uttar Pradesh", "Noida"): (28.5355, 77.3910),
    ("Uttarakhand", "Dehradun"): (30.3165, 78.0322),
    ("Uttarakhand", "Haridwar"): (29.9457, 78.1642),
    ("West Bengal", "Durgapur"): (23.5204, 87.3119),
    ("West Bengal", "Kolkata"): (22.5726, 88.3639),
}


def inject_animated_background() -> None:
    st.markdown(
        """
        <style>
        html, body {
            min-height: 100%;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0b3d91, #0b86d0, #2bc0e4);
            background-size: 400% 400%;
            animation: gradientAnimation 20s ease infinite;
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background: radial-gradient(circle at 25% 25%, rgba(255,255,255,0.18), transparent 18%),
                        radial-gradient(circle at 75% 20%, rgba(255,255,255,0.12), transparent 16%),
                        radial-gradient(circle at 50% 75%, rgba(255,255,255,0.10), transparent 20%);
            pointer-events: none;
            opacity: 0.6;
            filter: blur(4px);
            z-index: -1;
            animation: floatLights 30s linear infinite;
        }

        [data-testid="stAppViewContainer"],
        .main,
        .block-container,
        .css-18e3th9,
        .css-1d391kg,
        .css-1y0tads,
        .css-1v3fvcz {
            background: transparent !important;
            box-shadow: none !important;
        }

        [data-testid="stAppViewContainer"] > section,
        .css-1y0tads,
        .css-1d391kg,
        .css-1v3fvcz {
            background: rgba(255,255,255,0.12) !important;
            backdrop-filter: blur(14px) !important;
            box-shadow: 0 0 60px rgba(0, 0, 0, 0.10) !important;
            border-radius: 18px !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
        }

        .css-18e3th9, .css-1v3fvcz, .streamlit-expander {
            background: rgba(255,255,255,0.08) !important;
        }

        .css-18e3th9 {
            filter: blur(0) !important;
        }

        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatLights {
            0% { transform: translate(0, 0); }
            50% { transform: translate(20px, -30px); }
            100% { transform: translate(0, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600)
def load_property_data(state: Optional[str] = None, city: Optional[str] = None) -> pd.DataFrame:
    """Load property data from `india_housing_prices.csv`.
    """
    base = Path(__file__).resolve().parent
    india_path = base / "india_housing_prices.csv"

    if india_path.exists():
        df = pd.read_csv(india_path, encoding="utf-8", low_memory=False)
        
        # map common columns to the pipeline's expected names
        if "State" in df.columns:
            df["State"] = df["State"].astype(str)
        if "City" in df.columns:
            df["City"] = df["City"].astype(str)
        if "Price_per_SqFt" in df.columns:
            df["Price per SqFt"] = pd.to_numeric(df["Price_per_SqFt"], errors="coerce")
        if "Size_in_SqFt" in df.columns:
            df["Area SqFt"] = pd.to_numeric(df["Size_in_SqFt"], errors="coerce")
        if "BHK" in df.columns:
            df["Beds"] = pd.to_numeric(df["BHK"], errors="coerce")
        if "Property_Type" in df.columns:
            df["Property Type"] = df["Property_Type"].astype(str)

        
        # map additional numeric features from India dataset
        if "Year_Built" in df.columns:
            df["Year Built"] = pd.to_numeric(df["Year_Built"], errors="coerce")
        if "Floor_No" in df.columns:
            df["Floor Number"] = pd.to_numeric(df["Floor_No"], errors="coerce")
        if "Total_Floors" in df.columns:
            df["Total Floor"] = pd.to_numeric(df["Total_Floors"], errors="coerce")
        if "Age_of_Property" in df.columns:
            df["Age"] = pd.to_numeric(df["Age_of_Property"], errors="coerce")
        if "Nearby_Schools" in df.columns:
            df["Nearby Schools"] = pd.to_numeric(df["Nearby_Schools"], errors="coerce")
        if "Nearby_Hospitals" in df.columns:
            df["Nearby Hospitals"] = pd.to_numeric(df["Nearby_Hospitals"], errors="coerce")
        if "Parking_Space" in df.columns:
            df["Parking Space"] = df["Parking_Space"].astype(str)
        
        # map categorical features from India dataset
        if "Furnished_Status" in df.columns:
            df["Furnished Status"] = df["Furnished_Status"].astype(str)
        if "Public_Transport_Accessibility" in df.columns:
            df["Public Transport"] = df["Public_Transport_Accessibility"].astype(str)
        if "Security" in df.columns:
            df["Security"] = df["Security"].astype(str)
        if "Amenities" in df.columns:
            df["Amenities"] = df["Amenities"].astype(str)
        if "Facing" in df.columns:
            df["Facing"] = df["Facing"].astype(str)
        if "Owner_Type" in df.columns:
            df["Owner Type"] = df["Owner_Type"].astype(str)
        if "Locality" in df.columns:
            df["Locality"] = df["Locality"].astype(str)

        # map dataset-specific availability/status to the pipeline's Transaction Type
        if "Availability_Status" in df.columns:
            df["Transaction Type"] = df["Availability_Status"].astype(str)

        # basic derived fields
        if "Price_in_Lakhs" in df.columns:
            df["Price_in_Lakhs"] = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")

    else:
        # india dataset missing — return empty so UI can warn
        return pd.DataFrame()

    # apply optional state/city filters
    if state and "State" in df.columns:
        df = df[df["State"] == state].copy()
    if city and "City" in df.columns:
        df = df[df["City"] == city].copy()
    return df


def prepare_listing_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.copy()
    if df.empty:
        return df

    numeric_columns = [
        "Price_in_Lakhs",
        "Price per SqFt",
        "Area SqFt",
        "Beds",
        "Year Built",
        "Floor Number",
        "Total Floor",
        "Age",
        "Nearby Schools",
        "Nearby Hospitals",
        "BHK",
        "Size_in_SqFt",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    text_columns = df.select_dtypes(include=["object"]).columns
    for column in text_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
        df[column] = df[column].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    if "Amenities" in df.columns:
        df["Amenity Count"] = df["Amenities"].apply(
            lambda value: 0
            if pd.isna(value) or str(value).strip().lower() in {"", "unknown", "nan"}
            else len([item for item in str(value).split(",") if item.strip()])
        )

    df = df.dropna(subset=["Price_in_Lakhs"])
    return df


def apply_filter_panel(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.copy()
    with st.sidebar:
        st.header("Filter panel")

        if "Property Type" in filtered.columns:
            property_types = sorted(filtered["Property Type"].dropna().unique())
            selected_types = st.multiselect("Property type", property_types, default=property_types)
            if selected_types:
                filtered = filtered[filtered["Property Type"].isin(selected_types)]

        if "Beds" in filtered.columns and filtered["Beds"].notna().any():
            min_bhk = int(filtered["Beds"].min())
            max_bhk = int(filtered["Beds"].max())
            if min_bhk == max_bhk:
                st.write(f"BHK: {min_bhk}")
            else:
                selected_bhk = st.slider("BHK", min_bhk, max_bhk, (min_bhk, max_bhk))
                filtered = filtered[filtered["Beds"].between(selected_bhk[0], selected_bhk[1])]

        if "Area SqFt" in filtered.columns and filtered["Area SqFt"].notna().any():
            min_area = int(filtered["Area SqFt"].min())
            max_area = int(filtered["Area SqFt"].max())
            if min_area == max_area:
                st.write(f"Area SqFt: {min_area}")
            else:
                selected_area = st.slider("Area SqFt", min_area, max_area, (min_area, max_area))
                filtered = filtered[filtered["Area SqFt"].between(selected_area[0], selected_area[1])]

        for column, label in [
            ("Age", "Age"),
            ("Nearby Schools", "Nearby schools"),
            ("Nearby Hospitals", "Nearby hospitals"),
        ]:
            if column in filtered.columns and filtered[column].notna().any():
                min_value = int(filtered[column].min())
                max_value = int(filtered[column].max())
                if min_value == max_value:
                    st.write(f"{label}: {min_value}")
                else:
                    selected_range = st.slider(label, min_value, max_value, (min_value, max_value))
                    filtered = filtered[filtered[column].between(selected_range[0], selected_range[1])]

        for column, label in [
            ("Furnished Status", "Furnished status"),
            ("Public Transport", "Public transport"),
            ("Parking Space", "Parking space"),
        ]:
            if column in filtered.columns:
                options = sorted(filtered[column].dropna().unique())
                selected = st.multiselect(label, options, default=options)
                if selected:
                    filtered = filtered[filtered[column].isin(selected)]
            else:
                st.write(f"{label}: not available")

    return filtered


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    # Keep the prediction form consistent across cities with only required inputs.
    preferred_features = [
        "Property Type",
        "Beds",
        "Area SqFt",
        "Furnished Status",
        "Age",
        "Nearby Schools",
        "Nearby Hospitals",
        "Public Transport",
        "Parking Space",
    ]
    return [column for column in preferred_features if column in data.columns]


def relative_absolute_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def build_preprocessor(feature_data: pd.DataFrame) -> ColumnTransformer:
    numeric_features = feature_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in feature_data.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


@st.cache_resource(show_spinner=False)
def train_city_models(model_data: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline, str, list[str]]:
    feature_columns = get_feature_columns(model_data)
    X = model_data[feature_columns]
    y = model_data["Price_in_Lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    trained_models = {}
    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        rme = float(np.mean(y_test - predictions))
        rae = relative_absolute_error(y_test, predictions)
        r2 = float(r2_score(y_test, predictions))
        results.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "RME": rme,
                "RAE": rae,
                "Accuracy (R2 %)": r2 * 100,
            }
        )
        trained_models[model_name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(["RMSE", "RAE"], ascending=True)
    best_model_name = str(metrics_df.iloc[0]["Model"])
    return metrics_df, trained_models[best_model_name], best_model_name, feature_columns


def explain_best_model(metrics_df: pd.DataFrame, best_model_name: str) -> str:
    best_row = metrics_df[metrics_df["Model"] == best_model_name].iloc[0]
    next_best = metrics_df[metrics_df["Model"] != best_model_name].iloc[0]
    return (
        f"{best_model_name} is selected because it has the lowest RMSE "
        f"({best_row['RMSE']:.2f} lakhs) on this selected dataset. "
        f"Lower RMSE means its test predictions stay closer to the actual house prices. "
        f"It also compares well on RAE ({best_row['RAE']:.3f}) against the next best model, "
        f"{next_best['Model']} ({next_best['RAE']:.3f})."
    )


def prediction_form(data: pd.DataFrame, model: Pipeline, feature_columns: list[str]) -> None:
    st.subheader("Predict house price")
    st.caption("This form uses the best model trained on the currently selected and filtered city dataset.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        prediction_input = {}

        for index, column in enumerate(feature_columns):
            container = [col1, col2, col3][index % 3]
            series = data[column] if column in data.columns else pd.Series(dtype="object")

            with container:
                if pd.api.types.is_numeric_dtype(series):
                    clean_series = pd.to_numeric(series, errors="coerce").dropna()
                    default = float(clean_series.median()) if not clean_series.empty else 0.0
                    min_value = float(clean_series.min()) if not clean_series.empty else 0.0
                    max_value = float(clean_series.max()) if not clean_series.empty else max(default, 1.0)
                    if min_value == max_value:
                        max_value = min_value + 1.0
                    prediction_input[column] = st.number_input(
                        column,
                        min_value=min_value,
                        max_value=max_value,
                        value=min(max(default, min_value), max_value),
                    )
                else:
                    options = sorted(series.dropna().astype(str).unique())
                    if not options:
                        options = ["Unknown"]
                    prediction_input[column] = st.selectbox(column, options)

        submitted = st.form_submit_button("Predict price")

    if submitted:
        input_df = pd.DataFrame([prediction_input])
        predicted_price = float(model.predict(input_df)[0])
        st.success(f"Predicted house price: ₹{predicted_price:,.2f} lakhs")


def render_city_map(state: str, city: str, data: pd.DataFrame) -> None:
    coordinates = CITY_COORDINATES.get((state, city))
    if not coordinates:
        st.warning(f"Map coordinates are not available for {city}, {state}.")
        return

    latitude, longitude = coordinates
    map_data = pd.DataFrame(
        [
            {
                "State": state,
                "City": city,
                "Latitude": latitude,
                "Longitude": longitude,
                "Listings": len(data),
                "Avg Price": round(float(data["Price_in_Lakhs"].mean()), 2),
            }
        ]
    )

    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[Longitude, Latitude]",
        get_radius=8500,
        get_fill_color=[220, 38, 38, 180],
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
        pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=map_data,
        get_position="[Longitude, Latitude]",
        get_text="City",
        get_size=18,
        get_color=[17, 24, 39],
        get_angle=0,
        get_text_anchor='"middle"',
        get_alignment_baseline='"bottom"',
        get_pixel_offset=[0, -22],
    )

    st.subheader("Selected city map")
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=latitude,
                longitude=longitude,
                zoom=10,
                pitch=0,
            ),
            layers=[marker_layer, text_layer],
            tooltip={
                "html": (
                    "<b>{City}, {State}</b><br/>"
                    "Listings: {Listings}<br/>"
                    "Avg price: ₹{Avg Price} lakhs"
                ),
                "style": {"backgroundColor": "white", "color": "#111827"},
            },
        ),
        height=420,
    )


def render_visualizations(data: pd.DataFrame) -> None:
    st.subheader("Market visualizations")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Listings", f"{len(data):,}")
    metric_col2.metric("Avg price", f"₹{data['Price_in_Lakhs'].mean():,.2f} L")
    metric_col3.metric("Median area", f"{data['Area SqFt'].median():,.0f} sqft")
    metric_col4.metric("Avg BHK", f"{data['Beds'].mean():.1f}")

    chart_data = data.copy()
    if len(chart_data) > 5000:
        chart_data = chart_data.sample(5000, random_state=42)
        st.caption("Scatter and distribution charts use a 5,000-row sample for smoother rendering.")

    price_distribution = (
        alt.Chart(chart_data)
        .mark_bar(color="#2563eb", opacity=0.82)
        .encode(
            alt.X("Price_in_Lakhs:Q", bin=alt.Bin(maxbins=35), title="Price in lakhs"),
            alt.Y("count():Q", title="Number of listings"),
            tooltip=[alt.Tooltip("count():Q", title="Listings")],
        )
        .properties(height=320)
    )

    property_summary = (
        data.groupby("Property Type", as_index=False)
        .agg(
            Avg_Price_Lakhs=("Price_in_Lakhs", "mean"),
            Listings=("Price_in_Lakhs", "size"),
        )
        .sort_values("Avg_Price_Lakhs", ascending=False)
    )
    property_price = (
        alt.Chart(property_summary)
        .mark_bar(color="#059669")
        .encode(
            alt.X("Avg_Price_Lakhs:Q", title="Average price in lakhs"),
            alt.Y("Property Type:N", sort="-x", title="Property type"),
            tooltip=[
                alt.Tooltip("Property Type:N", title="Property type"),
                alt.Tooltip("Avg_Price_Lakhs:Q", title="Avg price", format=".2f"),
                alt.Tooltip("Listings:Q", title="Listings"),
            ],
        )
        .properties(height=320)
    )

    bhk_summary = (
        data.groupby("Beds", as_index=False)
        .agg(
            Avg_Price_Lakhs=("Price_in_Lakhs", "mean"),
            Listings=("Price_in_Lakhs", "size"),
        )
        .sort_values("Beds")
    )
    bhk_price = (
        alt.Chart(bhk_summary)
        .mark_line(point=True, color="#dc2626")
        .encode(
            alt.X("Beds:O", title="BHK"),
            alt.Y("Avg_Price_Lakhs:Q", title="Average price in lakhs"),
            tooltip=[
                alt.Tooltip("Beds:O", title="BHK"),
                alt.Tooltip("Avg_Price_Lakhs:Q", title="Avg price", format=".2f"),
                alt.Tooltip("Listings:Q", title="Listings"),
            ],
        )
        .properties(height=320)
    )

    area_vs_price = (
        alt.Chart(chart_data)
        .mark_circle(size=54, opacity=0.58)
        .encode(
            alt.X("Area SqFt:Q", title="Area SqFt"),
            alt.Y("Price_in_Lakhs:Q", title="Price in lakhs"),
            color=alt.Color("Property Type:N", title="Property type"),
            tooltip=[
                alt.Tooltip("Property Type:N", title="Property type"),
                alt.Tooltip("Beds:Q", title="BHK"),
                alt.Tooltip("Area SqFt:Q", title="Area", format=",.0f"),
                alt.Tooltip("Price_in_Lakhs:Q", title="Price", format=".2f"),
                alt.Tooltip("Furnished Status:N", title="Furnished"),
            ],
        )
        .interactive()
        .properties(height=380)
    )

    tab1, tab2, tab3 = st.tabs(["Price", "Property mix", "Area vs price"])
    with tab1:
        st.altair_chart(price_distribution, width="stretch")
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(property_price, width="stretch")
        with col2:
            st.altair_chart(bhk_price, width="stretch")
    with tab3:
        st.altair_chart(area_vs_price, width="stretch")


def render_dataset_cards(data: pd.DataFrame, city: str, state: str) -> None:
    st.subheader(f"Dataset overview for {city}, {state}")

    card_style = """
        <style>
        .dataset-card {
            padding: 18px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(17, 24, 39, 0.10);
            box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
            min-height: 118px;
            margin: 12px 0 20px 0;
        }
        .dataset-card-label {
            color: #475569;
            font-size: 0.88rem;
            margin-bottom: 8px;
        }
        .dataset-card-value {
            color: #111827;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .dataset-card-note {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 8px;
        }
        </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    cards = [
        ("Total records", f"{len(data):,}", "Listings after selected filters"),
        ("Average price", f"₹{data['Price_in_Lakhs'].mean():,.2f} L", "Mean house price"),
        ("Median price", f"₹{data['Price_in_Lakhs'].median():,.2f} L", "Middle listing price"),
        ("Average area", f"{data['Area SqFt'].mean():,.0f} sqft", "Mean property size"),
        ("Median BHK", f"{data['Beds'].median():.0f}", "Typical bedroom count"),
        ("Avg property age", f"{data['Age'].mean():.1f} years", "Mean age of listings"),
        ("Nearby schools", f"{data['Nearby Schools'].mean():.1f}", "Average nearby schools"),
        ("Nearby hospitals", f"{data['Nearby Hospitals'].mean():.1f}", "Average nearby hospitals"),
    ]

    for row_start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, note) in zip(cols, cards[row_start : row_start + 4]):
            with col:
                st.markdown(
                    f"""
                    <div class="dataset-card">
                        <div class="dataset-card-label">{label}</div>
                        <div class="dataset-card-value">{value}</div>
                        <div class="dataset-card-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.subheader("Dataset records")
    st.dataframe(
        data.sort_values("Price_in_Lakhs", na_position="last"),
        width="stretch",
    )


def render_dashboard_page(data: pd.DataFrame, selected_state: str, selected_city: str) -> None:
    st.subheader(f"Dashboard for {selected_city}, {selected_state}")
    st.write(f"{len(data):,} records available after filtering.")
    render_city_map(selected_state, selected_city, data)
    render_visualizations(data)


def render_model_page(data: pd.DataFrame) -> None:
    if len(data) < 10:
        st.warning("At least 10 records are needed after filtering to train and compare models reliably.")
        return

    st.subheader("Model comparison")
    with st.spinner("Preprocessing city data, splitting train/test data, and training models..."):
        metrics_df, best_model, best_model_name, feature_columns = train_city_models(data)

    st.dataframe(
        metrics_df.style.format(
            {
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}",
                "RME": "{:.2f}",
                "RAE": "{:.3f}",
                "Accuracy (R2 %)": "{:.2f}",
            }
        ),
        width="stretch",
    )
    st.info(explain_best_model(metrics_df, best_model_name))
    prediction_form(data, best_model, feature_columns)


def render_page_navigation() -> str:
    st.markdown(
        """
        <style>
        
        div[data-testid="stRadio"] {
            border-radius: 0;
            border: 0;
            padding: 0;
            box-shadow: none;
        }
        div[data-testid="stRadio"] > label {
            display: none;
        }
       
        div[data-testid="stRadio"] label {
            color: #374151;
            font-weight: 500;
        }
        div[data-testid="stRadio"] input {
            display: none;
        }
        div[data-testid="stRadio"] p {
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    selected_page = st.radio(
        "Page",
        ["Dashboard", "Dataset", "Model Comparison & Predict"],
        horizontal=True,
        label_visibility="collapsed",
        key="page_navigation",
    )
    st.markdown(
        """
        <div class="app-navbar-title">
            <h3 style="margin-bottom: 0;">India Real Estate Market Dashboard</h3>
            <p style="margin-top: 4px; color: #475569;">
                Select a city, filter listings, explore visuals, then compare models and predict price.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_page


def main() -> None:
    inject_animated_background()
    selected_page = render_page_navigation()

    all_data = load_property_data()
    if all_data.empty:
        st.warning("The dataset could not be loaded. Ensure `india_housing_prices.csv` exists in the project folder.")
        return

    state_options = sorted(all_data["State"].dropna().unique()) if "State" in all_data.columns else []
    selected_state = None
    selected_city = None
    with st.sidebar:
        st.header("Location selector")
        selected_state = st.selectbox("Select state", state_options, index=0 if state_options else 0)
        # dependent city list
        city_options = []
        if selected_state and "State" in all_data.columns and "City" in all_data.columns:
            city_options = sorted(all_data[all_data["State"] == selected_state]["City"].dropna().unique())

        selected_city = st.selectbox("Select city", city_options, index=0 if city_options else 0)
        load_city = st.button("Fetch City Data")
        st.write("---")
        st.write("Data is loaded from `india_housing_prices.csv`, then filtered by the selected state/city.")

    if load_city:
        st.session_state["selected_state"] = selected_state
        st.session_state["selected_city"] = selected_city

    selected_state = st.session_state.get("selected_state", selected_state)
    selected_city = st.session_state.get("selected_city", selected_city)

    data = load_property_data(state=selected_state, city=selected_city)
    if data.empty:
        st.warning(f"No listings found for {selected_city}. Try another city.")
        return

    # Full preprocessing and transformation before showing listings
    clean_data = prepare_listing_data(data)
    if clean_data.empty:
        st.warning("No listing data available after full preprocessing.")
        return

    filtered_data = apply_filter_panel(clean_data)
    if filtered_data.empty:
        st.warning("No listings match the selected filters.")
        return

    if selected_page == "Dashboard":
        render_dashboard_page(filtered_data, selected_state, selected_city)
    elif selected_page == "Dataset":
        render_dataset_cards(filtered_data, selected_city, selected_state)
    else:
        render_model_page(filtered_data)



if __name__ == "__main__":
    main()
