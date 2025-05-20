<H3>Naveen Raja N R</H3>
<H3>212222230093</H3>
<H3>EX.NO.6</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>

## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python

## Algorithm:
<b>Step 1:</b> Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
<b>Step 2:</b> Load the heart disease dataset from a file using pd.read_csv().<BR>
<b>Step 3:</b> Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
<b>Step 4:</b> Split the dataset into training and testing sets using train_test_split().<BR>
<b>Step 5:</b> Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
<b>Step 6:</b> Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
<b>Step 7:</b> Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
<b>Step 8:</b> Make predictions on the testing set using mlp.predict(X_test).<BR>
<b>Step 9:</b> Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
<b>Step 10:</b> Print the accuracy of the model.<BR>
<b>Step 11:</b> Plot the error convergence during training using plt.plot() and plt.show().<BR>

## Program: 

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset (assuming it's stored in a file)
data = pd.read_csv('heart.csv')

# Separate features and labels
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train, y_train).loss_curve_

# Make predictions on the testing set
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the error convergence
plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()

conf_matrix=confusion_matrix(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

```
## Output:
![image](https://github.com/user-attachments/assets/6ad2cd78-730e-4504-b9a9-99475a3fa81a)
![image](https://github.com/user-attachments/assets/8825d092-806c-42df-a0b4-70c7a6e7b9b1)

## Results:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
