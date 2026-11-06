
import pandas as pd


def group_icd9(code):
    """Map ICD-9-CM diagnosis codes to broad diagnostic groups."""

    if pd.isna(code):
        return "Missing"

    code = str(code).strip().upper()

    if code in {"?", "", "NULL", "NAN", "NONE"}:
        return "Missing"

    if code.startswith("V"):
        return "Supplementary_V"

    if code.startswith("E"):
        return "External_Cause_E"

    try:
        numeric_code = float(code)
    except ValueError:
        return "Unknown"

    # Diabetes requires a separate category
    if 250 <= numeric_code < 251:
        return "Diabetes"

    icd_ranges = [
        (1, 139, "Infectious"),
        (140, 239, "Neoplasms"),
        (240, 279, "Endocrine_Metabolic"),
        (280, 289, "Blood"),
        (290, 319, "Mental"),
        (320, 389, "Nervous_Sensory"),
        (390, 459, "Circulatory"),
        (460, 519, "Respiratory"),
        (520, 579, "Digestive"),
        (580, 629, "Genitourinary"),
        (630, 679, "Pregnancy"),
        (680, 709, "Skin"),
        (710, 739, "Musculoskeletal"),
        (740, 759, "Congenital"),
        (760, 779, "Perinatal"),
        (780, 799, "Symptoms"),
        (800, 999, "Injury_Poisoning"),
    ]

    for lower, upper, category in icd_ranges:
        if lower <= numeric_code < upper + 1:
            return category

    return "Other_Unknown"



def engineer_features(X):
    """Create deterministic features from discharge-time information."""

    X = X.copy()

    # Overall healthcare utilization in the preceding year
    X["total_prior_visits"] = (
        X["number_inpatient"]
        + X["number_emergency"]
        + X["number_outpatient"]
    )

    # Frequent previous hospitalizations
    X["frequent_inpatient"] = (
        X["number_inpatient"] >= 3
    ).astype("int8")

    # Group ICD-9 diagnoses and identify recorded diabetes diagnoses
    diagnosis_cols = ["diag_1", "diag_2", "diag_3"]

    for col in diagnosis_cols:
        X[f"{col}_group"] = X[col].apply(group_icd9)

    X["diabetes_diagnosis_flag"] = (
        X[[f"{col}_group" for col in diagnosis_cols]]
        .eq("Diabetes")
        .any(axis=1)
        .astype("int8")
    )

    # Drop raw high-cardinality diagnosis codes
    X = X.drop(columns=diagnosis_cols)

    return X



import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer


LAB_COLUMNS = ["A1Cresult", "max_glu_serum"]

HOSPITAL_ID_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

BINARY_COLUMNS = [
    "frequent_inpatient",
    "diabetes_diagnosis_flag",
]


def clean_features(X):
    """Normalize missing values and categorical hospital IDs."""
    X = X.copy()

    # Missing lab result means the test was not measured
    for col in LAB_COLUMNS:
        X[col] = X[col].fillna("Not_Measured")
        X[col] = X[col].replace(
            {"None": "Not_Measured", "?": "Unknown"}
        )

    # Preserve hospital ID semantics
    unknown_ids = {
        "admission_type_id": {5, 6, 8},
        "discharge_disposition_id": {18, 25, 26},
        "admission_source_id": {9, 15, 17, 20, 21},
    }

    for col in HOSPITAL_ID_COLUMNS:
        codes = pd.to_numeric(X[col], errors="coerce")
        X[col] = codes.map(
            lambda value: (
                "Unknown"
                if pd.isna(value) or value in unknown_ids[col]
                else str(int(value))
            )
        )

    # Other categorical fields: preserve unavailable information
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:
        X[col] = (
            X[col]
            .astype("string")
            .str.strip()
            .replace({"?": "Unknown", "": "Unknown"})
            .fillna("Unknown")
            .astype(str)
        )

    return X


def build_preprocessing_pipeline():
    """Build a reusable, unfitted feature preprocessing pipeline."""

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(
                    dtype_include=["object", "string", "category"]
                ),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline([
        (
            "feature_engineering",
            FunctionTransformer(
                engineer_features,
                validate=False,
            ),
        ),
        (
            "semantic_cleaning",
            FunctionTransformer(
                clean_features,
                validate=False,
            ),
        ),
        ("preprocessing", column_transformer),
    ])


