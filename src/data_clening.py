import logging
from abc import  ABC,abstractmethod
from typing import Union,Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    abstract class defining startegy for handling data
    """
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DatatPreProccesing(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            print('column to drop',data.columns.to_list())
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data['product_weight_g'].fillna(data['product_weight_g'].median(),inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(),inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(),inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(),inplace=True)
            data["review_comment_message"].fillna('No review',inplace=True)

            data=data.select_dtypes(include=[np.number])
            cols_to_drop=['customer_zip_code_prefix','order_item_id']
            data=data.drop(cols_to_drop,axis=1)
            return data
        except Exception as e:
            logging.error("error in oreprocessing data:{}".format(e))
            raise e
        
class DataDividestrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data:{}".format(e))
            raise e

        
class Datacleaning:
    """
    class for cleaning data and splitting

    """
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        "handle data"
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("error in handling data:{}".format(e))
            raise e
    


        


