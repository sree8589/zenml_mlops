import logging
import pandas as pd
from zenml import step
from src.data_clening  import Datacleaning,DatatPreProccesing,DataDividestrategy
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "xtrain"],
    Annotated[pd.DataFrame, 'xtest'],
    Annotated[pd.Series, 'ytrain'],
    Annotated[pd.Series, 'ytest']
]:
    """
    Cleans the data and divides it into training and testing sets.
    Returns xtrain, xtest, ytrain, and ytest.
    """
    try:
        preprocessing_strategy = DatatPreProccesing()
        data_cleaning = Datacleaning(df, preprocessing_strategy)
        cleaned_data = data_cleaning.handle_data()

        divide_strategy = DataDividestrategy()
        data_cleaning = Datacleaning(cleaned_data, divide_strategy)
        xtrain, xtest, ytrain, ytest = data_cleaning.handle_data()
        logging.info('Data cleaning completed')
        return xtrain, xtest, ytrain, ytest
    except Exception as e:
        logging.error(f'Error in cleaning data: {e}')
        raise e
