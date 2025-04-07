import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

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
        self.scaler = StandardScaler()
        self.model = None
        

    def __prepare_data(self):
        self.features = ["age", "educationalStatus", "fieldOfStudy", "maritalStatus", "gender",
                         "militaryStatus", "freeTime", "targetIncome", "intentionToMigrate",
                         "interestInMathematics", "computerExperience", "whichOneDoYouLikeMore",
                         "whichCaseIsmoreRelevant", "doYouWorkOnHolidays", "disability",
                         "addictionred"]

     
        self.data['result'] = self.encoder.fit_transform(self.data['result'])
        
        
        for feature in self.features:
            if self.data[feature].dtype == 'object':
                self.data[feature] = LabelEncoder().fit_transform(self.data[feature])
        
        
        self.X = self.data[self.features].values
        self.Y = self.data['result'].values  

        
        self.X = self.scaler.fit_transform(self.X)

    def __create_model(self):
        output_dim = len(np.unique(self.Y))
        self.model = Sequential([
            layers.Input(shape=(16,)), 
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim, activation='softmax')
        ])

    def __compile_model(self):
        self.model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['sparse_categorical_accuracy'])

    def __fit_model(self, x_train, y_train, epochs=20, batch_size=16):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, new_data):
        new_data = np.array([new_data]).reshape(1, -1)  
        new_data = self.scaler.transform(new_data)
        prediction = self.model.predict(new_data)
        predicted_class = np.argmax(prediction, axis=1)
        return self.encoder.inverse_transform(predicted_class)  

    def create_and_train_model(self):
        self.__prepare_data()
        self.__create_model()
        self.__compile_model()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.__fit_model(self.x_train, self.y_train, epochs=20)
