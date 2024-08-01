import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
class Evaluation(ABC):
    """abstarct method for evaluation"""
    @abstractmethod
    def calculate_scores(self,ytrue:np.ndarray,ypred:np.ndarray):
        pass


class MSE(Evaluation):
    """
    evaluation startegy using mse
    """
    def calculate_scores(self, ytrue: np.ndarray, ypred: np.ndarray):
        try:
            logging.info('calculating mse')
            mse=mean_squared_error(ytrue,ypred)
            logging.info('mse:{}'.format(mse))
            return mse
        except Exception as e:
            logging.error('error in calculating mse,{}'.format(e))
            raise e
        

class R2(Evaluation):
    """
    evaluation startegy using r2
    """
    def calculate_scores(self, ytrue: np.ndarray, ypred: np.ndarray):
        try:
            logging.info('calculating r2')
            r2=r2_score(ytrue,ypred)
            logging.info('mse:{}'.format(r2))
            return r2
        except Exception as e:
            logging.error('error in calculating mse,{}'.format(e))
            raise e
        
class RMSE(Evaluation):
    """
    evaluation startegy using rmse
    """
    def calculate_scores(self, ytrue: np.ndarray, ypred: np.ndarray):
        try:
            logging.info('calculating rmse')
            rmse=mean_squared_error(ytrue,ypred,squared=False)
            logging.info('rmse:{}'.format(rmse))
            return rmse
        except Exception as e:
            logging.error('error in calculating mse,{}'.format(e))
            raise e
        