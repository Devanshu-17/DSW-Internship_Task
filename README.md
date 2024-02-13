# DSW-Internship_Task

### Project Report: Fashion E-commerce Product Success Prediction

**Introduction:**

The aim of this project is to develop a machine learning application for a fashion e-commerce company to predict the success of potential products for the upcoming year. The company has compiled historical data on past products categorized as either "top" (successful) or "flop" (not successful). With this data, the goal is to build predictive models that can classify new product candidates as either "top" or "flop" based on their attributes.

**Data Description:**

The dataset consists of two main files:

1. `historic.csv`: Contains information on products from the past two years.
   - Attributes include `item_no`, `category`, `main_promotion`, `color`, `stars`, and `success_indicator`.
   - `item_no`: Internal identifier for a past product.
   - `category`: Category of the product.
   - `main_promotion`: Main promotion used to promote the product.
   - `color`: The main color of the product.
   - `stars`: Stars of reviews from a comparable product of a competitor (0= very negative reviews to 5= very positive reviews).
   - `success_indicator`: Indicates whether a product was successful (top) or not (flop) in the past.

2. `prediction_input.csv`: Contains information on potential products for the upcoming year.
   - Similar attributes as the historic data, except for the absence of the `success_indicator`.

**Data Preprocessing:**

- Categorical variables such as `category`, `main_promotion`, and `color` were encoded using LabelEncoder.
- The `stars` feature was binned into two categories: 0 for stars <= 3 (flop) and 1 for stars > 3 (top).
- The target variable `success_indicator` was encoded as 0 for 'flop' and 1 for 'top'.
- The dataset was split into training and testing sets with a 80-20 ratio.

**Modeling:**

Several machine learning models were trained and evaluated for this task:

1. Artificial Neural Network (ANN)
2. Random Forest Classifier
3. Logistic Regression
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Recurrent Neural Network (RNN)

**Model Evaluation:**

The models were evaluated using metrics such as accuracy, precision, recall, and F1-score. 

- Random Forest and KNN classifiers outperformed the other models in terms of accuracy, precision, recall, and F1-score.
- ANN, Logistic Regression, and SVM classifiers had relatively lower performance compared to Random Forest and KNN.
- RNN achieved competitive performance but slightly lower than Random Forest and KNN.

**Results:**

Here are the results obtained from each model:

- **ANN**:
  - Accuracy: 0.65
  - Precision: 0.65
  - Recall: 1.00
  - F1 Score: 0.79

- **Random Forest Classifier**:
  - Accuracy: 0.843125
  - Precision: 0.8504504504504504
  - Recall: 0.9173955296404276
  - F1 Score: 0.8826554464703131

- **Logistic Regression**:
  - Accuracy: 0.784375
  - Precision: 0.8220338983050848
  - Recall: 0.8483965014577259
  - F1 Score: 0.8350071736011477

- **SVM**:
  - Accuracy: 0.784375
  - Precision: 0.8220338983050848
  - Recall: 0.8483965014577259
  - F1 Score: 0.8350071736011477

- **KNN**:
  - Accuracy: 0.84125
  - Precision: 0.8578024007386889
  - Recall: 0.902818270165209
  - F1 Score: 0.8797348484848485

- **RNN**:
  - Test Accuracy: 0.8168749809265137
  - Precision: 0.82
  - Recall: 0.92
  - F1 Score: 0.87

**Conclusion:**

Based on the evaluation results, the Random Forest Classifier achieved the highest accuracy, precision, recall, and F1 score among the traditional machine learning models. However, the Recurrent Neural Network (RNN) also provided competitive results, especially in terms of accuracy and F1 score.

Considering the nature of the problem and the complexity of the dataset, further optimization and fine-tuning of the RNN model could potentially lead to even better performance. Additionally, exploring other deep learning architectures or ensemble techniques may further enhance predictive accuracy.

In conclusion, the developed machine learning models can effectively predict the success of fashion e-commerce products based on their attributes, providing valuable insights for decision-making in product selection and marketing strategies.
