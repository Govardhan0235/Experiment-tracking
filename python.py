import pathlib
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow
import xgboost as xgb
from prefect import flow, task 
from scipy.stats import ks_2samp

@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df

@task
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dv = DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task
def detect_drift(df_train: pd.DataFrame, df_val: pd.DataFrame, feature: str = "duration"):
    train_duration = df_train[feature].values
    val_duration = df_val[feature].values
    ks_stat, p_value = ks_2samp(train_duration, val_duration)
    mlflow.log_metric("ks_statistic", ks_stat)
    mlflow.log_metric("p_value", p_value)
    print(f"Drift KS statistic: {ks_stat}, p-value: {p_value}")
    drift_threshold = 0.001  # Threshold for KS statistic
    needs_retraining = ks_stat > drift_threshold
    print(f"Model needs retraining: {needs_retraining}")
    return needs_retraining

@task
def train_best_model(X_train, X_val, y_train, y_val, dv):
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }
        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

@flow
def main_flow(
    train_path: str = "https://github.com/fenago/datasets/raw/main/green_tripdata_2024-01.parquet",
    val_path: str = "https://github.com/fenago/datasets/raw/main/green_tripdata_2023-12.parquet",
):
    mlflow.set_tracking_uri(uri="sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_id="3")
    mlflow.autolog()
    df_train = read_data(train_path)
    df_val = read_data(val_path)
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    needs_retraining = detect_drift(df_train, df_val)
    if needs_retraining:
        print("Retraining model due to drift.")
        train_best_model(X_train, X_val, y_train, y_val, dv)
    else:
        print("No retraining required at this time.")

if __name__ == "__main__":
    main_flow()