# importing the necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing the datasets
datasets = pd.read_csv("../datasets/heart.csv")
datasets = datasets.drop('sex', axis=1)  # dropping

# printing the first 5 rows
print(datasets.head())

# droping wrong data
datasets = datasets[datasets['ca'] < 4] 
datasets = datasets[datasets['thal'] > 0]
print(f"The dataset after dropping wrong data:\nlength: {len(datasets)}\n")

# Renaming columns for clarity
datasets.rename(columns={
    'cp': 'chest_pain_type',
    'thalach': 'max_heart_rate',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'depression_induced_by_exercise',
    'slope': 'slope_of_st_segment',
    'ca': 'number_of_major_vessels',
    'thal': 'thalassemia',
    'trestbps': 'resting_blood_pressure',
    'chol': 'serum_cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_ecg_results'
}, inplace=True)

# Converting categorical variables to more readable formats
datasets['chest_pain_type'] = datasets['chest_pain_type'].map({
    0: 'typical_angina',
    1: 'atypical_angina',
    2: 'non_anginal_pain',
    3: 'asymptomatic'
})
datasets['thalassemia'] = datasets['thalassemia'].map({
    1: 'normal',
    2: 'fixed_defect',
    3: 'reversible_defect'
})
datasets['exercise_induced_angina'] = datasets['exercise_induced_angina'].map({
    0: 'no',
    1: 'yes'
})
datasets['slope_of_st_segment'] = datasets['slope_of_st_segment'].map({
    0: 'upsloping',
    1: 'flat',
    2: 'downsloping'
})
datasets['resting_ecg_results'] = datasets['resting_ecg_results'].map({
    0: 'normal',
    1: 'ST_T_wave_abnormality',
    2: 'left_ventricular_hypertrophy'
})
datasets['fasting_blood_sugar'] = datasets['fasting_blood_sugar'].map({
    0: 'less_than_120_mg_per_dl',
    1: 'greater_than_120_mg_per_dl'
})

# printing the first few rows of the dataset
print("First few rows of the dataset:")
print(datasets.head())

#checking for missing values
print("Checking for missing values:")
print(datasets.isnull().sum())
# checking the shape of the dataset
print("Shape of the dataset:", datasets.shape)  
# checking the data types of the columns
print("Data types of the columns:")
print(datasets.dtypes)

# Separating the features into numerical and categorical
categorical_cols = datasets.select_dtypes(include=['object']).columns.tolist()
numeric_cols = datasets.select_dtypes(exclude=['object']).columns.tolist()
# Separating the features and target variable
X = datasets.drop('target', axis=1)
y = datasets['target']

print(X.head())

# splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# applying one-hot encoding to categorical features
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# ✅ ColumnTransformer for One-Hot Encoding categorical & scaling numerical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)


#  Create a pipeline (preprocessing + model)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])


#  Train the model
model.fit(X_train, y_train)
y_test_array = y_test.values.reshape(-1, 1)  # convert Series to numpy and reshape
y_pred_array = model.predict(X_test).reshape(-1, 1)
print(np.concatenate((y_test_array, y_pred_array), axis=1))



#  Make predictions
y_pred = model.predict(X_test)

#  Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 1️⃣4️⃣ Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease","Disease"], yticklabels=["No Disease","Disease"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Saving the model
import pickle
import os

# Save model to 'models/heart_model.pkl'
model_path = os.path.join('../models', 'heart_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
