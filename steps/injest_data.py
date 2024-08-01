import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    ingesting data from the data_path
    """
    def __init__(self,data_path: str):
        """
        Args:
            path to the data
        """
        self.data_path=data_path

    def get_data(self):
        """
        ingestig  the data from the path.
        """
        logging.info(f"injesting data from{self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    injesting the data.
    Args:
        data_path: path to the data
    returns:
        pd.Dataframe: the injested data
    """
    try:
        ingest_data = IngestData(data_path)
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"error while ingesting data: {e}")
        raise e