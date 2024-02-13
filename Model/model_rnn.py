import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten

# Load the data
df = pd.read_csv('/content/historic.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])
df['main_promotion_encoded'] = label_encoder.fit_transform(df['main_promotion'])
df['color_encoded'] = label_encoder.fit_transform(df['color'])

# Split features and target
X = df.drop(['success_indicator', 'item_no', 'category', 'main_promotion', 'color'], axis=1)
y = df['success_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to sequences for RNN input
X_train_seq = np.array(X_train_scaled).reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_seq = np.array(X_test_scaled).reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Encode target variable y_train
label_encoder_target = LabelEncoder()
y_train_encoded = label_encoder_target.fit_transform(y_train)

# Define the RNN model
model = Sequential([
    SimpleRNN(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with encoded y_train
model.fit(X_train_seq, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Encode target variable y_test
y_test_encoded = label_encoder_target.transform(y_test)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_seq, y_test_encoded)
print(f"Test Accuracy: {accuracy}")

# Make predictions
y_pred = (model.predict(X_test_seq) > 0.5).astype("int32")

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test_encoded, y_pred)
print("Classification Report:\n", report)
