import pandas as pd   
from sklearn.linear_model import LogisticRegression

def planningAI():
    data = pd.read_csv("../../assets/planning.csv")
    xData = data[""].values
    yData = data[""].values
    linearModel = LogisticRegression()
    linearModel.fit(X=xData,y=yData)
    result = linearModel.predict(xData)
    return result[0]