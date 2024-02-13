import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class ANNClassifier:
    def __init__(self):
        self.pipeline = None
        self.label_encoder_category = None
        self.label_encoder_main_promotion = None
        self.label_encoder_color = None
        self.scaler = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        x = self.data.drop(['success_indicator', 'item_no'], axis=1)
        y = self.data['success_indicator']

        # Encoding categorical variables
        self.label_encoder_category = LabelEncoder()
        x['category_encoded'] = self.label_encoder_category.fit_transform(x['category'])
        x.drop('category', axis=1, inplace=True)

        self.label_encoder_main_promotion = LabelEncoder()
        x['main_promotion_encoded'] = self.label_encoder_main_promotion.fit_transform(x['main_promotion'])
        x.drop('main_promotion', axis=1, inplace=True)

        self.label_encoder_color = LabelEncoder()
        x['color_encoded'] = self.label_encoder_color.fit_transform(x['color'])
        x.drop('color', axis=1, inplace=True)

        # Binning stars ratings
        x['stars'] = np.where(x['stars'] <= 3, 0, 1)

        # Encoding target variable ('FLOP' as 0, 'TOP' as 1)
        label_encoder_target = LabelEncoder()
        y_encoded = label_encoder_target.fit_transform(y)
        y_encoded = np.where(y_encoded == label_encoder_target.classes_.tolist().index('flop'), 0, y_encoded)
        y_encoded = np.where(y_encoded == label_encoder_target.classes_.tolist().index('top'), 1, y_encoded)

        # Splitting the data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=77)

        # Scaling features
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)

    def create_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit_model(self):
        keras_model = KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=32, verbose=0)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', keras_model)
        ])
        self.pipeline.fit(self.x_train, self.y_train)

    def predict(self):
        return self.pipeline.predict(self.x_test)

    def evaluate_model(self):
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Accuracy: {:.2f}".format(accuracy))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1 Score: {:.2f}".format(f1))

    def load_test_data(self, file_path):
        self.test_data = pd.read_csv(file_path)

    def preprocess_test_data(self):
        test_data_processed = self.test_data.drop(['item_no'], axis=1)
        test_data_processed['category_encoded'] = self.label_encoder_category.transform(test_data_processed['category'])
        test_data_processed.drop('category', axis=1, inplace=True)
        test_data_processed['main_promotion_encoded'] = self.label_encoder_main_promotion.transform(test_data_processed['main_promotion'])
        test_data_processed.drop('main_promotion', axis=1, inplace=True)
        test_data_processed['color_encoded'] = self.label_encoder_color.transform(test_data_processed['color'])
        test_data_processed.drop('color', axis=1, inplace=True)
        test_data_processed['stars'] = np.where(test_data_processed['stars'] <= 3, 0, 1)
        test_data_processed = self.scaler.transform(test_data_processed)
        return test_data_processed

    def predict_for_test_data(self):
        test_data_processed = self.preprocess_test_data()
        return self.pipeline.predict(test_data_processed)

# Initialize the classifier
pipeline = ANNClassifier()

# Load and preprocess the training data
pipeline.load_data('/content/historic.csv')
pipeline.preprocess_data()

# Create and train the model
pipeline.create_model()
pipeline.fit_model()

# Evaluate the model
pipeline.evaluate_model()

# Load and preprocess the test data
pipeline.load_test_data('/content/prediction_input.csv')
predicted_classes = pipeline.predict_for_test_data()
