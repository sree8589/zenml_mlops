from zenml import pipeline
from steps.injest_data import ingest_df
from steps.cleaning_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def training_pipe(data_path: str):
    df=ingest_df(data_path)
    xtrain,xtest,ytrain,ytest=clean_data(df)
    model =train_model(xtrain,xtest,ytrain,ytest)
    r2_Score,rmse=evaluate_model(model,xtest,ytest)
    

