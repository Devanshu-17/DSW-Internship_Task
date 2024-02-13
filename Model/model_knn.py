import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class KNNPipeline:
    def __init__(self):
        self.knn_clf = None
        self.scaler = None
        self.label_encoder_category = None
        self.label_encoder_main_promotion = None
        self.label_encoder_color = None
        
    def load_data(self, file_path):
        """Load the data from the specified file path."""
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        """Preprocess the data."""
        x = self.data.drop(['success_indicator', 'item_no'], axis=1)
        y = self.data['success_indicator']
        
        # Encode categorical variables
        self.label_encoder_category = LabelEncoder()
        x['category_encoded'] = self.label_encoder_category.fit_transform(x['category'])
        x.drop('category', axis=1, inplace=True)
        
        self.label_encoder_main_promotion = LabelEncoder()
        x['main_promotion_encoded'] = self.label_encoder_main_promotion.fit_transform(x['main_promotion'])
        x.drop('main_promotion', axis=1, inplace=True)
        
        self.label_encoder_color = LabelEncoder()
        x['color_encoded'] = self.label_encoder_color.fit_transform(x['color'])
        x.drop('color', axis=1, inplace=True)
        
        # Bin stars rating into two categories
        x['stars'] = np.where(x['stars'] <= 3, 0, 1)
        
        # Encoding target variable ('flop' as 0, 'top' as 1)
        label_encoder_target = LabelEncoder()
        y_encoded = label_encoder_target.fit_transform(y)
        y_encoded = np.where(y_encoded == label_encoder_target.classes_.tolist().index('flop'), 0, y_encoded)
        y_encoded = np.where(y_encoded == label_encoder_target.classes_.tolist().index('top'), 1, y_encoded)

        # Split the data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

        # Scale features
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)

    def train_model(self):
        """Train the KNN model."""
        self.knn_clf = KNeighborsClassifier(n_neighbors=5)
        self.knn_clf.fit(self.x_train, self.y_train)

    def test_model(self):
        """Test the trained model."""
        y_pred = self.knn_clf.predict(self.scaler.transform(self.x_test))
        
        # Evaluate the model performance
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print("Accuracy with KNN Classifier:", accuracy)
        print("Precision with KNN Classifier:", precision)
        print("Recall with KNN Classifier:", recall)
        print("F1 Score with KNN Classifier:", f1)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

    def load_test_data(self, file_path):
        """Load the unlabelled test data."""
        self.test_data = pd.read_csv(file_path)

    def preprocess_test_data(self):
        """Preprocess the test data."""
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
        """Predict the success indicator for the test data."""
        test_data_processed = self.preprocess_test_data()
        return self.knn_clf.predict(test_data_processed)

# Initialize the KNN pipeline
pipeline = KNNPipeline()

# Load and preprocess the data
pipeline.load_data('/content/historic.csv')
pipeline.preprocess_data()

# Train the KNN model
pipeline.train_model()

# Test the model
pipeline.test_model()

# Load and preprocess the test data
pipeline.load_test_data('/content/prediction_input.csv')
predicted_classes = pipeline.predict_for_test_data()
