import pandas as pd
import logging
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker
import mlflow
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: RegressorMixin,
        xtest: pd.DataFrame,
        ytest: pd.Series,  # Correct type here
    ) -> Tuple[
            Annotated[float, "r2"],
            Annotated[float, "rmse"],
        ]:
    
    try:
        # Get predictions
        predictions = model.predict(xtest)
        
        # Initialize evaluation classes
        mse_class = MSE()
        r2_class = R2()
        rmse_class = RMSE()
        
        # Calculate metrics
        mse = mse_class.calculate_scores(ytest, predictions)
        mlflow.log_metric("mse",mse)
        r2 = r2_class.calculate_scores(ytest, predictions)
        mlflow.log_metric('r2',r2)
        rmse = rmse_class.calculate_scores(ytest, predictions)

        return r2, rmse
    except Exception as e:
        logging.error('Error in evaluating model: {}'.format(e))
        raise e
