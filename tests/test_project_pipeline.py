import pandas as pd

from src.components.data_ingestion import load_data
from src.pipeline.predict_pipeline import predict_pipeline


def test_load_data_returns_dataframe():
    df = load_data("notebook/data/cleaned_diabetic_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert "readmitted" in df.columns


def test_predict_pipeline_returns_risk_payload():
    sample = {
        "age": "[40-50)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 5,
        "race": "Caucasian",
        "gender": "Female",
        "diag_1": "250",
        "diag_2": "401",
        "diag_3": "276",
        "A1Cresult": ">8",
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "Steady",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "insulin": "No",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "metformin-pioglitazone": "No",
        "change": "No",
        "diabetesMed": "Yes",
    }

    result = predict_pipeline(sample)
    assert set(result.keys()) >= {"readmit_probability", "prediction", "threshold_used"}
    assert isinstance(result["prediction"], str)
