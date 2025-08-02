import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# importing the datasets
datasets = pd.read_csv("../datasets/diabetes_dataset.csv")
X = datasets.iloc[:, :-1].values 
y = datasets.iloc[:, -1].values

# printing the first few rows of the dataset
print("First few rows of the dataset:")
print(datasets.head())



# exploratory data analysis
print("Descriptive statistics of the dataset:")
print(datasets.describe())


# checking for missing values
print("Checking for missing values:")
print(datasets.isnull().sum())
# checking the shape of the dataset
print("Shape of the dataset:", datasets.shape)  
# checking the data types of the columns
print("Data types of the columns:")
print(datasets.dtypes)

#splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Making predictions
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), axis=1))

# making a single prediction
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)  # Example input data
input_data = np.asarray(input_data)
# Reshaping the input data
input_data = input_data.reshape(1, -1)
# Scaling the input data
input_data = scaler.transform(input_data)
# Predicting the result
prediction = classifier.predict(input_data)

if prediction[0] == 1:
    print("The person is likely to have diabetes.")
else:
    print("The person is unlikely to have diabetes.")
    
# Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Saving the model
import pickle
import os

# Save model to 'models/diabetes_model.pkl'
model_path = os.path.join('../models', 'diabetes_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)


