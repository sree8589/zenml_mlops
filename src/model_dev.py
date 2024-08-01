import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    abstarct method for all class
    """
    @abstractmethod
    def train(self,xtrain,ytrain):
        pass

class LinearRegressionModel(Model):
    def train(self, xtrain, ytrain,**kwargs):
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(xtrain,ytrain)
            logging.info('model training completed')
            return reg
        except Exception as e:
            logging.error('error in training model:{}'.format(e))
            raise(e)
