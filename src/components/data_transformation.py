import sys
import os

import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException

from src.logger import logging
from src.utils import save_object


# Diagnosis code mapping — raw ICD-9 codes -> 9 broad clinical categories

def map_diagnosis_codes(code):
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "other"

    try:
        code_num = float(code)
    except ValueError:
        return "other"
    if 250 <= code_num < 251:
        return "Diabetes"
    elif (390 <=  code_num <= 459) or code_num == 785:
        return "Circulatory"
    elif (460 <= code_num <= 519) or code_num == 786:
        return "Respiratory"
    elif (520 <= code_num <= 579) or code_num == 787:
        return "Digestive"
    elif (580 <= code_num <= 629) or code_num == 788:
        return "Genitourinary"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    else:
        return "other"

# Age bracket -> ordinal number (order is meaningful, so not one-hot)


AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
    "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
    "[80-90)": 8, "[90-100)": 9,
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot_encoder = None
        self.categorical_cols = None
        self.feature_columns = None

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["diag_1", "diag_2", "diag_3"]:
            if col in df.columns:
                df[col] = df[col].apply(map_diagnosis_codes)

        df["age"] = df["age"].map(AGE_MAP)

        df["total_prior_visits"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )
        return df

    
    def fit(self, X: pd.DataFrame, y=None):
        df = self._engineer(X)
        self.categorical_cols = df.select_dtypes(include="object").columns.tolist()

        self.onehot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.onehot_encoder.fit(df[self.categorical_cols])

        encoded_cols = self.onehot_encoder.get_feature_names_out(self.categorical_cols)
        self.feature_columns = [c for c in df.columns if c not in self.categorical_cols] + list(encoded_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._engineer(X)

        encoded_array = self.onehot_encoder.transform(df[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.onehot_encoder.get_feature_names_out(self.categorical_cols),
            index=df.index,
        )

        df_final = df.drop(columns=self.categorical_cols)
        df_final = pd.concat([df_final, encoded_df], axis=1)
        df_final = df_final.reindex(columns=self.feature_columns, fill_value=0)
        return df_final

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data for transformation")

            target_column = "readmitted"
            X_train_raw = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test_raw = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Fitting FeatureEngineer on training data")
            feature_engineer = FeatureEngineer()
            X_train = feature_engineer.fit_transform(X_train_raw)
            X_test = feature_engineer.transform(X_test_raw)

            logging.info(f"Transformed shapes - Train: {X_train.shape}, Test: {X_test.shape}")

            save_object(self.data_transformation_config.preprocessor_obj_file_path, feature_engineer)
            logging.info("Saved preprocessor object to artifacts/preprocessor.pkl")


            return (
                X_train,
                y_train,
                X_test,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


        


    

