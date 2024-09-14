import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Accuracy  # Import the correct Accuracy metric

class NeuralPlanningAI:
    def __init__(self):
        self.x_test = None
        self.x_train = None
        self.y_train = None
        self.y_test = None
        self.data = pd.read_csv("../assets/planning.csv")
        self.X = None
        self.Y = None
        self.encoder = LabelEncoder()
        self.model = None
        print("working")

    def __prepare_data(self):
        self.features = ["age", "educationalStatus", "fieldOfStudy", "maritalStatus", "gender",
                         "militaryStatus", "freeTime", "targetIncome", "intentionToMigrate",
                         "interestInMathematics", "computerExperience", "whichOneDoYouLikeMore",
                         "whichCaseIsmoreRelevant", "doYouWorkOnHolidays", "disability",
                         "addictionred"]

        self.data['result'] = self.encoder.fit_transform(self.data['result'])  # Encode target variable
        self.X = self.data[self.features].values
        self.Y = self.data['result']

    def __create_model(self):
        self.model = Sequential([
            layers.Input(shape=(16,)),  # Change shape to (16,) to match your data
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

    def __compile_model(self):
        self.model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[Accuracy()])

    def __fit_model(self, x_train, y_train, epochs=10):
        self.model.fit(x_train, y_train, epochs=epochs)

    def predict(self, new_data):
        new_data = np.array([new_data]).reshape(1, -1)  # Reshape for prediction
        prediction = self.model.predict(new_data)
        return self.encoder.inverse_transform(np.argmax(prediction[0]).ravel())
    def create_and_train_model(self):
        self.__prepare_data()
        self.__create_model()
        self.__compile_model()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.__fit_model(self.x_train, self.y_train, epochs=10)



