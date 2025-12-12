# Heart Disease Prediction Using Machine Learning
A complete classification project including EDA, preprocessing, model development, and model evaluation.

## Project Overview
This project predicts the presence of heart disease using clinical attributes from a publicly available dataset.  
It follows an end-to-end machine learning workflow that includes:

- Exploratory Data Analysis (EDA)  
- Data preprocessing and scaling  
- Training multiple classification algorithms  
- Evaluating model performance  
- Comparing results across models  
- Identifying the best-performing classifier  

The goal is to build a reliable and interpretable predictive system while understanding algorithm strengths and limitations.

## Dataset Credit
Source: Kaggle – Heart Disease Dataset  
Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### Features
The dataset contains 13 clinical measurements:

- age  
- sex  
- cp (chest pain type)  
- trestbps (resting blood pressure)  
- chol (cholesterol level)  
- fbs (fasting blood sugar)  
- restecg (electrocardiographic results)  
- thalach (maximum heart rate achieved)  
- exang (exercise-induced angina)  
- oldpeak (ST depression)  
- slope  
- ca  
- thal  

### Target Variable
- `target = 1` → heart disease present  
- `target = 0` → no heart disease  

Dataset contains no missing values.

## Exploratory Data Analysis (EDA)
The following analyses were performed:

### Numeric Feature Distributions
Histograms for:
- age  
- trestbps  
- chol  
- thalach  
- oldpeak  

### Categorical Feature Counts
Countplots for:
- sex  
- cp  
- fbs  
- restecg  
- exang  
- slope  
- ca  
- thal  
- target  

### Correlation Heatmap
A heatmap was used to examine correlations between all numerical features.

### Pairplot
Visual inspection of class separation across selected features.

### Boxplots (Target vs Key Risk Factors)
- Age vs Target  
- Cholesterol vs Target  
- Oldpeak vs Target  

These visualizations reveal important clinical relationships, such as higher ST depression and older age being linked to increased heart disease risk.

## Preprocessing
- Features (X) and labels (y) were separated  
- 80/20 train-test split  
- StandardScaler applied to models requiring scaling (Logistic Regression, KNN, SVC, Naive Bayes)  
- Tree-based models (Decision Tree, Random Forest) trained on unscaled data  

## Models Trained
Six machine learning algorithms were implemented:

1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Support Vector Classifier (RBF kernel)  
4. Decision Tree Classifier  
5. Random Forest Classifier  
6. Gaussian Naive Bayes  

Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|-----------|
| Logistic Regression | 0.795 | 0.756 | 0.874 | 0.811 |
| KNN | 0.834 | 0.800 | 0.893 | 0.844 |
| SVC (RBF) | 0.888 | 0.851 | 0.942 | 0.894 |
| Decision Tree | 0.985 | 1.000 | 0.971 | 0.985 |
| Random Forest | 0.985 | 1.000 | 0.971 | 0.985 |
| Naive Bayes | 0.800 | 0.754 | 0.893 | 0.818 |

## Best Performing Models
- **Decision Tree** and **Random Forest** achieved the highest accuracy (98.5%).  
  However, both models show signs of overfitting because they were trained with unlimited depth.  
- **Support Vector Classifier (RBF)** demonstrated the strongest generalization performance, achieving:
  - 88.8% accuracy  
  - 94.2% recall  
  - 89.4% F1 score  

SVC provides the best balance of stability, sensitivity, and predictive power.

## Future Improvements
Potential enhancements include:

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- ROC curve and AUC scoring  
- Cross-validation for more robust metrics  
- Feature importance visualization for Random Forest  
- Deploying the model using Streamlit or Flask  
- Testing additional models such as XGBoost or LightGBM  

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

---

## Author
**Noor Hazem Ibrahim Seif**  
Computer Science Student – AUS  
LinkedIn: https://www.linkedin.com/in/noor-seif-39babb328/

