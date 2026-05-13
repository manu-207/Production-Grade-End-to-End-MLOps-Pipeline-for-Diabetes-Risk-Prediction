import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise EnvironmentError("MLFLOW_TRACKING_URI env var is not set")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    ## log metrics to MLFLOW

    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])
