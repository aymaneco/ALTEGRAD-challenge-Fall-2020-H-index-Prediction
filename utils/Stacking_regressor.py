
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
def Stacking_regressor(model_1, model_2, model_3):
    """
    Stack models to enhance the performance.
    Inputs :
        model_1 : First model ;
        model_2 : Second model ;
        model_3 : Third model
    Output :
        regressor : The Stacked three models
    """
    estimators = [
    ('lgb', model_1),
    ('cat', model_2),
    ('xgb', model_3)
 ]
    regressor = StackingRegressor(
     estimators=estimators,
     final_estimator=LinearRegression(),
 )
    return regressor
