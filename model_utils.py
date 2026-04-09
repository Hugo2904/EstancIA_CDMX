import joblib
import pandas as pd
import streamlit as st

def load_artifacts():
    return {
        "model_price": joblib.load("rforest_price.pkl"),
        "model_occupancy": joblib.load("rforest_occupancy.pkl"),
        "ohe_price": joblib.load("ohe_price.pkl"),
        "scaler_price": joblib.load("scaler_price.pkl"),
        "ohe_occupancy": joblib.load("ohe_occupancy.pkl"),
        "scaler_occupancy": joblib.load("scaler_occupancy.pkl"),
        "price_feature_order": joblib.load("price_feature_order.pkl"),
        "occupancy_feature_order": joblib.load("occupancy_feature_order.pkl"),
        "price_numeric_cols": joblib.load("price_numeric_cols.pkl"),
        "occupancy_numeric_cols": joblib.load("occupancy_numeric_cols.pkl"),
        "market_data": joblib.load("market_data.pkl"),
    }


def build_base_input(
    property_type,
    neighbourhood_cleansed,
    host_age,
    host_verifications,
    host_has_profile_pic,
    host_identity_verified,
    accommodates,
    bathrooms,
    bedrooms,
    amenities,
    minimum_nights,
    maximum_nights,
    number_of_reviews,
    instant_bookable,
    availability_30,
):
    trust_level = (
        host_verifications +
        host_has_profile_pic +
        host_identity_verified
    )

    return pd.DataFrame([{
        "property_type": property_type,
        "neighbourhood_cleansed": neighbourhood_cleansed,
        "host_age": host_age,
        "trust_level": int(trust_level),
        "accommodates": accommodates,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "amenities": amenities,
        "minimum_nights": minimum_nights,
        "maximum_nights": maximum_nights,
        "number_of_reviews": number_of_reviews,
        "instant_bookable": instant_bookable,
        "availability_30": availability_30,
    }])


def build_price_input(df_base: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "property_type",
        "neighbourhood_cleansed",
        "host_age",
        "trust_level",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "amenities",
        "minimum_nights",
        "maximum_nights",
        "number_of_reviews",
        "instant_bookable",
        "availability_30",
    ]
    return df_base[cols].copy()


def build_occupancy_input(df_base: pd.DataFrame, price_value: float) -> pd.DataFrame:
    cols = [
        "property_type",
        "neighbourhood_cleansed",
        "host_age",
        "trust_level",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "amenities",
        "minimum_nights",
        "maximum_nights",
        "number_of_reviews",
        "instant_bookable",
    ]
    df_occ = df_base[cols].copy()
    df_occ["price"] = price_value
    return df_occ


def apply_partial_scaling(df: pd.DataFrame, scaler, numeric_cols: list) -> pd.DataFrame:
    df = df.copy()

    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def transform_for_price(df_price: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    X = artifacts["ohe_price"].transform(df_price)

    expected_cols = artifacts["price_feature_order"]
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_cols]
    X = apply_partial_scaling(
        X,
        artifacts["scaler_price"],
        artifacts["price_numeric_cols"]
    )
    return X


def transform_for_occupancy(df_occ: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    X = artifacts["ohe_occupancy"].transform(df_occ)

    expected_cols = artifacts["occupancy_feature_order"]
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_cols]
    X = apply_partial_scaling(
        X,
        artifacts["scaler_occupancy"],
        artifacts["occupancy_numeric_cols"]
    )
    return X


def predict_price(df_base: pd.DataFrame, artifacts: dict) -> float:
    df_price = build_price_input(df_base)
    X = transform_for_price(df_price, artifacts)
    pred = artifacts["model_price"].predict(X)[0]
    return float(pred)


def predict_occupancy(df_base: pd.DataFrame, price_value: float, artifacts: dict) -> float:
    df_occ = build_occupancy_input(df_base, price_value)
    X = transform_for_occupancy(df_occ, artifacts)
    pred = artifacts["model_occupancy"].predict(X)[0]
    return max(0.0, min(1.0, float(pred)))


def calculate_monthly_income(price_value: float, occupancy_rate: float, days: int = 30) -> float:
    return float(price_value * days * occupancy_rate)


def market_comparison(
    df_base: pd.DataFrame,
    user_price: float,
    artifacts: dict
) -> dict:
    market = artifacts["market_data"].copy()
    row = df_base.iloc[0]

    required_cols = [
        "neighbourhood_cleansed",
        "property_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "price",
        "amenities",
        "host_age"
    ]

    missing_cols = [col for col in required_cols if col not in market.columns]

    if missing_cols:
        return {
            "n_comparables": 0,
            "position": "No disponible",
            "missing_cols": missing_cols
        }

    # 1) filtro base: misma zona + mismo tipo
    comparables = market[
        (market["neighbourhood_cleansed"] == row["neighbourhood_cleansed"]) &
        (market["property_type"] == row["property_type"])
    ].copy()

    if comparables.empty:
        return {
            "n_comparables": 0,
            "position": "Sin comparables",
            "missing_cols": []
        }

    # 2) score de similitud dentro de los comparables
    comparables["sim_accommodates"] = 1 - (
        abs(comparables["accommodates"] - row["accommodates"]) /
        max(1, market["accommodates"].max() - market["accommodates"].min())
    )

    comparables["sim_bathrooms"] = 1 - (
        abs(comparables["bathrooms"] - row["bathrooms"]) /
        max(1, market["bathrooms"].max() - market["bathrooms"].min())
    )

    comparables["sim_bedrooms"] = 1 - (
        abs(comparables["bedrooms"] - row["bedrooms"]) /
        max(1, market["bedrooms"].max() - market["bedrooms"].min())
    )

    comparables["sim_score"] = (
        0.4 * comparables["sim_accommodates"] +
        0.3 * comparables["sim_bathrooms"] +
        0.3 * comparables["sim_bedrooms"]
    ).clip(lower=0.01)

    weights = comparables["sim_score"]

    # 3) métricas ponderadas del mercado
    market_avg_price = float((comparables["price"] * weights).sum() / weights.sum())
    avg_amenities = float((comparables["amenities"] * weights).sum() / weights.sum())
    avg_host_age = float((comparables["host_age"] * weights).sum() / weights.sum())
    avg_bathrooms = float((comparables["bathrooms"] * weights).sum() / weights.sum())
    avg_bedrooms = float((comparables["bedrooms"] * weights).sum() / weights.sum())
    avg_accommodates = float((comparables["accommodates"] * weights).sum() / weights.sum())

    price_diff_abs = float(user_price - market_avg_price)
    price_diff_pct = float((user_price / market_avg_price - 1) * 100) if market_avg_price != 0 else None

    amenities_diff = float(row["amenities"] - avg_amenities)
    host_age_diff = float(row["host_age"] - avg_host_age)
    bathrooms_diff = float(row["bathrooms"] - avg_bathrooms)
    bedrooms_diff = float(row["bedrooms"] - avg_bedrooms)
    accommodates_diff = float(row["accommodates"] - avg_accommodates)

    if user_price < market_avg_price * 0.95:
        position = "Por debajo del mercado"
    elif user_price > market_avg_price * 1.05:
        position = "Por encima del mercado"
    else:
        position = "En línea con el mercado"

    # percentil simple de precio dentro de comparables base
    price_percentile = float((comparables["price"] < user_price).mean() * 100)

    # insights
    insights = []

    if price_diff_pct is not None:
        if price_diff_pct > 10:
            insights.append("Tu precio se encuentra por encima del mercado para propiedades similares.")
        elif price_diff_pct < -10:
            insights.append("Tu precio se encuentra por debajo del mercado y podría haber margen para aumentarlo.")
        else:
            insights.append("Tu precio se encuentra en un rango cercano al mercado.")

    if amenities_diff < -1:
        insights.append("Tu propiedad tiene menos amenidades que el promedio de los comparables.")
    elif amenities_diff > 1:
        insights.append("Tu propiedad ofrece más amenidades que el promedio de los comparables.")

    if bathrooms_diff < -0.5:
        insights.append("Tienes menos baños que propiedades similares.")
    elif bathrooms_diff > 0.5:
        insights.append("Tienes más baños que propiedades similares.")

    if bedrooms_diff < -0.5:
        insights.append("Tienes menos recámaras que propiedades similares.")
    elif bedrooms_diff > 0.5:
        insights.append("Tienes más recámaras que propiedades similares.")

    if host_age_diff < -1:
        insights.append("Tu inmueble tiene menor antigüedad en plataforma que el promedio del mercado.")
    elif host_age_diff > 1:
        insights.append("Tu inmueble tiene mayor antigüedad en plataforma que el promedio del mercado.")

    if not insights:
        insights.append("Tu propiedad está bastante alineada con el mercado en las variables principales.")

    return {
        "n_comparables": int(len(comparables)),
        "position": position,
        "missing_cols": [],

        "market_avg_price": market_avg_price,
        "price_diff_abs": price_diff_abs,
        "price_diff_pct": price_diff_pct,
        "price_percentile": price_percentile,

        "user_amenities": float(row["amenities"]),
        "avg_amenities": avg_amenities,
        "amenities_diff": amenities_diff,

        "user_host_age": float(row["host_age"]),
        "avg_host_age": avg_host_age,
        "host_age_diff": host_age_diff,

        "user_bathrooms": float(row["bathrooms"]),
        "avg_bathrooms": avg_bathrooms,
        "bathrooms_diff": bathrooms_diff,

        "user_bedrooms": float(row["bedrooms"]),
        "avg_bedrooms": avg_bedrooms,
        "bedrooms_diff": bedrooms_diff,

        "user_accommodates": float(row["accommodates"]),
        "avg_accommodates": avg_accommodates,
        "accommodates_diff": accommodates_diff,

        "insights": insights
    }