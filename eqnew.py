# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import folium

# Load dataset
# Make sure to replace 'your_dataset.csv' with the actual file path or URL of your dataset
earthquake_data = pd.read_csv('database.csv')

# Data Exploration
# You can add your data exploration code here

# Global Visualization using Folium
def plot_earthquakes_on_map(data):
    m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=2)
    for index, row in data.iterrows():
        folium.CircleMarker([row['Latitude'], row['Longitude']], radius=row['Magnitude']*2, color='red').add_to(m)
    return m

# Uncomment the line below to visualize earthquakes on a map
# plot_earthquakes_on_map(earthquake_data).save('earthquake_map.html')

# Data Preprocessing
# Feature Selection
selected_features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
data_subset = earthquake_data[selected_features]

# Normalize numerical data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_subset)

# Split data into features and target variable
X = normalized_data[:, :-1]
y = normalized_data[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Model Training
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
