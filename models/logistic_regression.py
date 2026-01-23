### Logistic Regression Model Implementation
#Sai Tej#
#######


## Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("Logistic Regression Model Implementation")

class LogisticRegression:
    def __init__(self, alpha=0.09, num_iters=2000):
        self.alpha = alpha
        self.num_iters = num_iters
        self.theta = None
        self.loss = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.random.rand(n)
        self.loss = []
        for i in range(self.num_iters):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            self.theta -= (self.alpha / m) * np.dot(X.T, (h - y))
            cost = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
            self.loss.append(cost)
        return self.loss, self.theta

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in h]

# Data Loading and Preprocessing
data = pd.read_csv('../data/Breast_cancer_data.csv')

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Standardizing the data
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# Create and train the model
model = LogisticRegression(alpha=0.09, num_iters=2000)
loss, theta = model.fit(X_train, y_train)
print(f"The cost after training loss: {np.squeeze(loss)}")
print(f"The resulting vector of weights is: {theta}")

# Make predictions
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))