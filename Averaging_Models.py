from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import numpy as np
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
  """
  Average the prediction models to have one final weighted output
  Inputs :
      X : features ;
      y : target variable ;
      models : models to be averaged
  Output :
      mean of the predictions
  """
  def __init__(self, X, y, models):
      self.X = X
      self.y = y
      self.models = models
  def fit(self, X, y):
      self.models_ = [clone(x) for x in self.models]  
        # Train cloned base models
      for model in self.models_:
            model.fit(self.X, self.y)
      return self
      
      #Now we do the predictions for cloned models and average them
  def predict(self, X):
      predictions = np.column_stack([ model.predict(X) for model in self.models_ ])
      return np.mean(predictions, axis=1)
