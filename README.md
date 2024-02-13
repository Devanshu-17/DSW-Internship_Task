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

