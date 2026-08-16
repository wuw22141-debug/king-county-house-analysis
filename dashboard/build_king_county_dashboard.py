#!/usr/bin/env python3
"""Build the self-contained King County House Sales HTML dashboard.

Examples
--------
1. Download the data from OpenML and build the dashboard:

   python build_king_county_dashboard.py

2. Rebuild from an existing CSV file:

   python build_king_county_dashboard.py --csv king_county_house_sales.csv

3. Put the output in another directory:

   python build_king_county_dashboard.py --output-dir output

The script uses Python for data preparation and model training, then injects
JSON data into an HTML/CSS/JavaScript template. Plotly.js is embedded directly,
so the resulting HTML can be opened offline without running a web server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline.offline import get_plotlyjs
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


OPENML_DATA_ID = 42092
DEFAULT_TEMPLATE = Path(__file__).with_name("king_county_dashboard_template.html")
DEFAULT_OUTPUT_NAME = "king_county_house_sales_dashboard.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the King County interactive HTML dashboard."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional local CSV. If omitted, download OpenML data_id=42092.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="HTML template containing {{PLACEHOLDER}} values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for the generated files.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Generated HTML filename.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save the downloaded/raw data as a CSV.",
    )
    return parser.parse_args()


def load_data(csv_path: Path | None) -> pd.DataFrame:
    """Load an existing CSV or obtain the dataset from OpenML."""
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print(f"[1/7] Reading local CSV: {csv_path}")
        return pd.read_csv(csv_path)

    print(f"[1/7] Downloading OpenML dataset {OPENML_DATA_ID}...")
    bunch = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
    )
    return bunch.frame.copy()


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean types and construct analysis features used by the dashboard."""
    print("[2/7] Cleaning data and creating derived features...")
    df = raw.copy()

    required = {
        "id", "date", "price", "bedrooms", "bathrooms", "sqft_living",
        "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # OpenML stores this date as a string such as 20141013T000000.
    # pandas can also parse the YYYY-MM-DD version saved by this project.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [c for c in required if c not in {"date"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows unusable by the main visualizations.
    df = df.dropna(
        subset=[
            "date", "price", "sqft_living", "sqft_lot", "yr_built",
            "zipcode", "lat", "long",
        ]
    ).copy()

    # Time features.
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.quarter
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Interpretable features shown in charts or filters.
    df["house_age"] = df["sale_year"] - df["yr_built"]
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    df["built_decade"] = (df["yr_built"] // 10 * 10).astype(int)
    df["sqft_per_bedroom"] = (
        df["sqft_living"] / df["bedrooms"].replace(0, np.nan)
    )
    df["bathrooms_per_bedroom"] = (
        df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
    )
    df["living_lot_ratio"] = (
        df["sqft_living"] / df["sqft_lot"].replace(0, np.nan)
    )

    # Use integer ZIP codes in controls and labels.
    df["zipcode"] = df["zipcode"].astype(int)
    return df


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    """Train comparison models and create compact model-output datasets."""
    print("[3/7] Training and evaluating regression models...")

    feature_columns = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "sqft_above",
        "sqft_basement", "yr_built", "yr_renovated", "zipcode",
        "lat", "long", "sqft_living15", "sqft_lot15", "sale_month",
    ]

    model_df = df[feature_columns + ["price"]].dropna().copy()
    X = model_df[feature_columns]
    y = model_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=140,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    metrics: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"      fitting {name}...")
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        predictions[name] = prediction
        metrics.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(y_test, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "r2": float(r2_score(y_test, prediction)),
            }
        )

    metrics_df = pd.DataFrame(metrics).sort_values("mae").reset_index(drop=True)
    best_model_name = str(metrics_df.iloc[0]["model"])
    best_prediction = predictions[best_model_name]
    best_residual = y_test.to_numpy() - best_prediction

    # The full test set is not necessary for one scatter plot. A deterministic
    # sample keeps the HTML smaller and the browser responsive.
    rng = np.random.default_rng(42)
    sample_size = min(1500, len(y_test))
    sample_indices = rng.choice(len(y_test), size=sample_size, replace=False)
    model_scatter = [
        {
            "actual": float(y_test.iloc[i]),
            "predicted": float(best_prediction[i]),
            "residual": float(best_residual[i]),
        }
        for i in sample_indices
    ]

    random_forest = models["Random forest"]
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(15)

    return {
        "metrics_df": metrics_df,
        "importance_df": importance_df,
        "model_scatter": model_scatter,
        "best_model_name": best_model_name,
        "best_mae": float(metrics_df.iloc[0]["mae"]),
        "best_r2": float(metrics_df.iloc[0]["r2"]),
    }


def build_correlation_data(df: pd.DataFrame) -> tuple[list[str], list[list[float]]]:
    """Create the fixed correlation matrix shown on the overview tab."""
    columns = [
        "price", "sqft_living", "grade", "bathrooms", "bedrooms",
        "view", "waterfront", "condition", "house_age", "lat",
        "sqft_living15", "price_per_sqft",
    ]
    corr = df[columns].corr().round(3)
    return corr.columns.tolist(), corr.values.tolist()


def dashboard_records(df: pd.DataFrame) -> pd.DataFrame:
    """Select only columns required by browser-side filtering and charts."""
    columns = [
        "id", "date", "price", "bedrooms", "bathrooms", "sqft_living",
        "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
        "year_month", "house_age", "renovated", "has_basement",
        "price_per_sqft", "built_decade",
    ]
    out = df[columns].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def json_records(frame: pd.DataFrame) -> str:
    """Serialize DataFrame rows as compact JSON accepted directly by JS."""
    return frame.to_json(orient="records")


def fill_template(
    template_text: str,
    df: pd.DataFrame,
    model_output: dict[str, Any],
) -> str:
    """Inject Plotly, records and precomputed model outputs into the template."""
    print("[4/7] Serializing data and filling the HTML template...")

    labels, corr_z = build_correlation_data(df)
    dash_df = dashboard_records(df)

    zipcodes = sorted(df["zipcode"].unique().tolist())
    zipcode_options = "\n".join(
        f'<option value="{int(z)}">{int(z)}</option>' for z in zipcodes
    )

    metrics_df: pd.DataFrame = model_output["metrics_df"]
    importance_df: pd.DataFrame = model_output["importance_df"]

    replacements = {
        "{{PLOTLY_JS}}": get_plotlyjs(),
        "{{N_FORMATTED}}": f"{len(df):,}",
        "{{ZIP_N}}": str(df["zipcode"].nunique()),
        "{{DATE_MIN}}": df["date"].min().strftime("%Y-%m-%d"),
        "{{DATE_MAX}}": df["date"].max().strftime("%Y-%m-%d"),
        "{{MIN_PRICE}}": str(int(df["price"].min())),
        "{{MAX_PRICE}}": str(int(df["price"].max())),
        "{{MIN_YEAR}}": str(int(df["yr_built"].min())),
        "{{MAX_YEAR}}": str(int(df["yr_built"].max())),
        "{{ZIPCODE_OPTIONS}}": zipcode_options,
        "{{BEST_MODEL}}": str(model_output["best_model_name"]),
        "{{BEST_MAE_FORMATTED}}": f'{model_output["best_mae"]:,.0f}',
        "{{BEST_R2}}": f'{model_output["best_r2"]:.3f}',
        "{{HOUSES_JSON}}": json_records(dash_df),
        "{{METRICS_JSON}}": json_records(metrics_df),
        "{{IMPORTANCE_JSON}}": json_records(importance_df),
        "{{MODEL_SCATTER_JSON}}": json.dumps(
            model_output["model_scatter"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "{{CORR_Z_JSON}}": json.dumps(corr_z, separators=(",", ":")),
        "{{CORR_LABELS_JSON}}": json.dumps(labels, separators=(",", ":")),
    }

    result = template_text
    for placeholder, value in replacements.items():
        if placeholder not in result:
            raise ValueError(f"Template placeholder missing: {placeholder}")
        result = result.replace(placeholder, value)

    # Detect unresolved project placeholders. Plotly's minified source may
    # legitimately contain double braces, so only match our all-caps tokens.
    import re
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", result)))
    if unresolved:
        raise ValueError(f"Unresolved template placeholders: {unresolved[:10]}")

    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.csv)
    df = prepare_data(raw)
    print(f"      prepared shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if args.save_csv:
        csv_out = args.output_dir / "king_county_house_sales.csv"
        df.to_csv(csv_out, index=False)
        print(f"      saved CSV: {csv_out}")

    model_output = train_models(df)

    if not args.template.exists():
        raise FileNotFoundError(f"Dashboard template not found: {args.template}")
    template_text = args.template.read_text(encoding="utf-8")
    html = fill_template(template_text, df, model_output)

    output_path = args.output_dir / args.output_name
    print(f"[5/7] Writing HTML: {output_path}")
    output_path.write_text(html, encoding="utf-8")

    print("[6/7] Performing basic output checks...")
    checks = {
        "HTML begins correctly": html.lstrip().startswith("<!DOCTYPE html>"),
        "Plotly library embedded": "plotly.js" in html.lower(),
        "House records embedded": "const houses = [" in html,
        "Dashboard initialization present": (
            "renderAll();" in html or "renderActiveTab();" in html
        ),
        "No unresolved project placeholders": not __import__("re").search(
            r"\{\{[A-Z0-9_]+\}\}", html
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"      {'OK' if ok else 'FAIL'}: {name}")
    if failed:
        raise RuntimeError(f"Output validation failed: {failed}")

    print("[7/7] Complete")
    print(f"      rows embedded: {len(df):,}")
    print(f"      output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"      best model: {model_output['best_model_name']}")
    print(f"      MAE: ${model_output['best_mae']:,.0f}")
    print(f"      R²: {model_output['best_r2']:.3f}")
    print("\nOpen the generated HTML file in Chrome, Edge, Firefox or Safari.")


if __name__ == "__main__":
    main()
