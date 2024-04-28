# MEEN-423 Final Project
# Vincent Irani (830004153)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import Adam

# Load data from the CSV file
df = pd.read_csv("star_classification.csv")

# Defining Features and response
features = df.loc[:, df.columns != "class"]
response = df['class']

# Encode labels to convert from catagorical to numerical feature outputs
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(response)

# Scaling the data
scaler = StandardScaler()
scaler.fit(features)
X_scaled = scaler.transform(features) # This is the entire data set for PCA testing purposes

# Creating PCA object
pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

# Doing analysis to determine the n_components to keep
plt.figure(figsize=(15, 6))

# Making Scree Plot
plt.subplot(1, 2, 1)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel("Number of Principal Components", fontsize=16)
plt.ylabel("Eigenvalues", fontsize=16)

# Making Explained Variance Plot
plt.subplot(1, 2, 2)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
plt.plot(exp_var_cumul)
plt.xlabel("Number of Principal Components", fontsize=16)
plt.ylabel("Explained Variance", fontsize=16)

plt.show()

# Based on PCA testing, n_components to keep should be ~8-10

# Splitting test and training data
x_train, x_test, y_train, y_test = train_test_split(features, response, test_size=0.2, random_state = 53)

# scaling data
scaler = StandardScaler()
scaler.fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)

# PCA
pca = PCA(n_components=8)
pca.fit(x_train_sc)

X_pca_train = pca.transform(x_train_sc)
X_pca_test = pca.transform(x_test_sc)

# Creating tenserflow ANN object
model = Sequential([
    Dense(64, input_shape=(8,), activation='relu'),  # Input layer with 10 principal components
    Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    Dense(3, activation='softmax')  # Output layer with 3 neurons (one for each class)
])
model.compile(optimizer=Adam(learning_rate=.01), loss='sparse_categorical_crossentropy')

# Training the model
history = model.fit(X_pca_train, y_train, epochs=10, validation_data=(X_pca_test, y_test), verbose=0)

# Testing the model
training_error = np.sqrt(mean_squared_error(y_train, model.predict(X_pca_train)))
testing_error = np.sqrt(mean_squared_error(y_test, model.predict(X_pca_test)))
r_2_score = r2_score(y_test, model.predict(X_pca_test))

print("Training RMSE: " + str(training_error))
print("Testing RMSE: " + str(testing_error))
print("Testing R^2 Score: " + str(r_2_score))

# Plotting error across epochs
plt.plot(history.history['loss'], label='Training Data MSE')
plt.plot(history.history['val_loss'], label='Test Data MSE')
plt.title('Mean Squared Error across Epochs')
plt.ylabel('MSE (Mean Squared Error)')
plt.xlabel('Epochs')
plt.legend()
plt.show()