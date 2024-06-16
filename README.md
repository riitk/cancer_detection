# cancer_detection
Cancer Detection Using ML

# Breast Cancer Detection using Machine Learning

## Overview
This project aims to detect breast cancer using various features from the breast cancer dataset. The dataset consists of 569 instances with 33 columns, including diagnostic information and various measurements of cell nuclei.

## Dataset
The dataset used for this project contains the following columns:

- `id`: Unique identifier
- `diagnosis`: The diagnosis of breast tissues (M = malignant, B = benign)
- `radius_mean`: Mean of distances from center to points on the perimeter
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Mean size of the core tumor
- `area_mean`: Mean area of the tumor
- `smoothness_mean`: Mean of local variation in radius lengths
- `compactness_mean`: Mean of perimeter^2 / area - 1.0
- `concavity_mean`: Mean of severity of concave portions of the contour
- `concave points_mean`: Mean for number of concave portions of the contour
- `symmetry_mean`: Mean of symmetry
- `fractal_dimension_mean`: Mean for "coastline approximation" - 1
- `radius_se`: Standard error of distances from center to points on the perimeter
- `texture_se`: Standard error of gray-scale values
- `perimeter_se`: Standard error of size of the core tumor
- `area_se`: Standard error of area of the tumor
- `smoothness_se`: Standard error of local variation in radius lengths
- `compactness_se`: Standard error of perimeter^2 / area - 1.0
- `concavity_se`: Standard error of severity of concave portions of the contour
- `concave points_se`: Standard error for number of concave portions of the contour
- `symmetry_se`: Standard error of symmetry
- `fractal_dimension_se`: Standard error for "coastline approximation" - 1
- `radius_worst`: "Worst" or largest mean value for distances from center to points on the perimeter
- `texture_worst`: "Worst" or largest mean value of gray-scale values
- `perimeter_worst`: "Worst" or largest mean value of size of the core tumor
- `area_worst`: "Worst" or largest mean value of area of the tumor
- `smoothness_worst`: "Worst" or largest mean value of local variation in radius lengths
- `compactness_worst`: "Worst" or largest mean value of perimeter^2 / area - 1.0
- `concavity_worst`: "Worst" or largest mean value of severity of concave portions of the contour
- `concave points_worst`: "Worst" or largest mean value for number of concave portions of the contour
- `symmetry_worst`: "Worst" or largest mean value of symmetry
- `fractal_dimension_worst`: "Worst" or largest mean value for "coastline approximation" - 1
- `Unnamed: 32`: Column with all null values, which was dropped

## Data Preprocessing
1. **Basic Analysis**:
   - Checked the shape, descriptions, and basic information using `df.info()` and `df.describe()`.
   - Verified null values using `df.isnull().sum()`.

2. **Column Dropping**:
   - Dropped the `Unnamed: 32` column as it contained all null values.
    ```python
        df.drop("Unnamed: 32", axis =1, inplace=True)
     ```
   - Dropped the `id` column as it was not relevant for prediction.
    ```python
        df.drop("id", axis =1, inplace=True)
     ```

3. **Diagnosis Value Counts**:
   - Verified the count of each diagnosis category:
    ```python
        df["diagnosis"].value_counts()
     ```
     ```plaintext
     B    357
     M    212
     Name: count, dtype: int64
     ```

4. **Encoding Diagnosis**:
   - Converted the `diagnosis` column to numerical values using the `map` function:
     ```python
     df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
     ```

5. **Correlation Analysis**:
   - Plotted a heatmap to visualize correlations between all columns.
   - Checked the correlation of the `diagnosis` column with other features.
   ```python
    df.corr()["diagnosis"].sort_values()
    ```

## Model Training
1. **Train-Test Split**:
   - Used `train_test_split` from `sklearn` with stratification on the `diagnosis` column to ensure balanced distribution in the training and testing sets:
     ```python
     from sklearn.model_selection import train_test_split
     X = df.drop('diagnosis', axis=1)
     y = df['diagnosis']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
     ```

2. **Model Selection**:
   - Trained a `LogisticRegression` model due to the small dataset size:
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression(max_iter=10000)
     model.fit(X_train, y_train)
     ```

## Model Evaluation
Evaluated the model using `accuracy_score`, `classification_report`, and `confusion_matrix`:
- **Accuracy Score**: 0.94
- **Confusion Matrix**:
  ```plaintext
  [[106  1]
   [ 9 55]]


## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/riitk/cancer_detection.git
   cd cancer_detection