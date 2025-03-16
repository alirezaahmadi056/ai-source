import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class ClusterPlanningAI:
    def __init__(self):
        self.data = pd.read_csv("../assets/planning.csv")
        self.encoder = LabelEncoder()
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.features = ["age", "educationalStatus", "fieldOfStudy", "maritalStatus", "gender",
                         "militaryStatus", "freeTime", "targetIncome", "intentionToMigrate",
                         "interestInMathematics", "computerExperience", "whichOneDoYouLikeMore",
                         "whichCaseIsmoreRelevant", "doYouWorkOnHolidays", "disability",
                         "addictionred"]

        self.data['result'] = self.encoder.fit_transform(self.data['result'])
        X = self.data[self.features].values
        Y = self.data['result']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, new_data):
        return self.encoder.inverse_transform(self.model.predict(np.array(new_data).reshape(1, -1)))
