import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Simulated dataset for training
data = {
    'Age': [25, 30, 45, 50, 60],
    'BMI': [22.0, 25.5, 28.0, 30.0, 35.5],
    'Glucose_Level': [85, 90, 110, 130, 150],
    'Diabetes_Risk': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Splitting data
X = df[['Age', 'BMI', 'Glucose_Level']]
y = df['Diabetes_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, './models/diabetes_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
