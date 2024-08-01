from pipelines.training_pipeline import training_pipe
from zenml.client import Client
if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipe(data_path="/home/mahesh/mlopsproject/data/olist_customers_dataset1.csv")
    

# mlflow ui --backend-store-uri "file:/home/mahesh/.config/zenml/local_stores/fa9cc8fa-ab1f-4aac-b458-20c2c2757fbd/mlruns"