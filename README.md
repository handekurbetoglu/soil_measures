# Predicting Optimal Crop Selection Based on Soil Metrics Using Machine Learning
## Overview
This project leverages machine learning and SHAP (SHapley Additive exPlanations) to help farmers select the best crops based on soil metrics. The dataset includes 22 different crops with soil features such as nitrogen (N), phosphorus (P), potassium (K), and pH value.

## Table of Contents
### 1. Introduction
### 2. Dataset
### 3. Exploratory Data Analysis (EDA)
### 4. Model Selection
### 5. SHAP Summary Plot
### 6. SHAP Dependence Plot
### 7. How to Run the Analysis
### 8. Interpreting the Results
### 9. Conclusion


## 1. Introduction
Selecting the optimal crop to plant based on soil conditions is a critical decision for farmers aiming to maximize yield and profitability. This project employs advanced machine learning techniques to predict the best crop for given soil conditions, based on a dataset of 22 crops and key soil metrics. By utilizing SHAP (SHapley Additive exPlanations), we can visualize and interpret the contributions of each soil feature to the crop prediction, providing valuable insights for farmers.

## 2. Dataset
The dataset, soil_measures.csv, contains the following columns:

"N": Nitrogen content ratio in the soil

"P": Phosphorus content ratio in the soil

"K": Potassium content ratio in the soil

"ph": pH value of the soil

"crop": Categorical values representing various crops (target variable)


## 3. Exploratory Data Analysis (EDA)
EDA helps us understand the structure and characteristics of the dataset. It involves visualizing the distributions of features and identifying any correlations between them.


Steps for EDA:

Load the Dataset:

    
    crops = pd.read_csv("soil_measures.csv")

    
    crops.describe()

Visualize Distributions:



    for column in crops.columns[:-1]:
        plt.figure()
        plt.hist(crops[column])
        plt.xlabel(column)
        plt.xticks(rotation=90)
        plt.ylabel('Frequency')
        plt.show()

Correlation Heatmap:


    crops_corr = crops[["N", "P", "K", "ph"]].corr()

    sns.heatmap(crops_corr, annot=True)

    plt.show()


## 4. Model Selection
Choosing the right model is crucial for accurate predictions. This project uses an ensemble approach with a VotingClassifier combining:

RandomForestClassifier

GradientBoostingClassifier

XGBClassifier (XGBoost)

DecisionTreeClassifier


Steps for Model Selection:

Split the Data:

    X_train, X_test, y_train, y_test = train_test_split(
        crops[final_features],
        crops["crop"],
        test_size=0.2,
        random_state=10
    )

Define and Train the Models:

    rf = RandomForestClassifier(random_state=10)

    gr = GradientBoostingClassifier(random_state=10)

    XGB = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=10)

    dt = DecisionTreeClassifier(random_state=10)


Ensemble VotingClassifier:

    vc.fit(X_train, y_train)

    vc_pred = vc.predict(X_test)

    model_performance_vc = accuracy_score(y_test, vc_pred)

    print('Voting Classifier\'s model performance: '  + str(model_performance_vc))

VotingClassifier performed best in these models.

## 5. SHAP Summary Plot
A SHAP summary plot provides an overview of the importance and impact of each feature on the model's predictions.


### How to Read a SHAP Summary Plot
#### 1. Feature Importance:

Y-axis: Lists the features in order of importance. Features at the top are the most important.

Interpretation: The higher a feature is on the y-axis, the greater its overall impact on the model's predictions.

#### 2. Impact on Model Output:

X-axis: Shows the SHAP values. A higher absolute SHAP value indicates a greater impact on the model's prediction.

Interpretation: Positive SHAP values push the model output towards a specific class, while negative SHAP values push it away.

#### 3. Distribution of Impact:

Dots: Each dot represents a SHAP value for a specific feature and instance.

Color: The color of the dots represents the feature value (red for high values, blue for low values).

Interpretation:

Horizontal Spread: The wider the spread of points, the more variable the feature's impact.

Color Gradient: Indicates the relationship between the feature value and its impact. For example, if red dots (high values) are mostly on the right, high feature values increase the prediction.


## 6. SHAP Dependence Plot
A SHAP dependence plot shows the relationship between a feature and the model's output. It can also highlight interactions with another feature.

### How to Read a SHAP Dependence Plot
#### 1. Feature Value:

X-axis: Represents the value of the feature being plotted.

Interpretation: Shows how different values of the feature affect the model's prediction.

#### 2. Impact on Model Output:

Y-axis: Shows the SHAP value for the feature, indicating its impact on the model's prediction.

Interpretation: Higher SHAP values mean a stronger positive impact on predicting the specific class.

#### 3. Color Gradient:

Color: Represents the values of another feature (interaction feature).

Interpretation: Helps identify interactions between features. For example, if high values of the interaction feature consistently change the impact of the main feature, there might be an interaction effect.


## 7. How to Run the Analysis
Prerequisites

Python 3.6 or higher

Required libraries: pandas, scikit-learn, matplotlib, seaborn, xgboost, shap

### Steps
1. Load the Dataset:

    crops = pd.read_csv("soil_measures.csv")

2. Encode the Target Variable:
 
    label_encoder = LabelEncoder()

    crops["crop"] = label_encoder.fit_transform(crops["crop"])

3. Split the Data:


 
    X_train, X_test, y_train, y_test = train_test_split(
        crops[["N", "P", "K", "ph"]],
        crops["crop"],
        test_size=0.2,
        random_state=10
    )

4. Train the Model:
 
    vc = VotingClassifier(estimators=[('rf', rf), ('gr', gr), ('XGB', XGB), ('dt', dt)], voting='hard')
    vc.fit(X_train, y_train)

5. Generate SHAP Values:
 
    explainer = shap.TreeExplainer(vc.named_estimators_['XGB'])
    shap_values = explainer.shap_values(X_test)

6. Plot SHAP Values for Each Class:
 
    for i in range(len(label_encoder.classes_)):

        class_name = label_encoder.classes_[i]
        class_shap_values = shap_values[i]
        shap.summary_plot(class_shap_values, X_test, feature_names=final_features, show=False)
        plt.title(f"Summary plot for class: {class_name}")
        plt.show()

        for j, feature in enumerate(final_features):
            shap.dependence_plot(j, class_shap_values, X_test, feature_names=final_features, show=False)
            plt.title(f"Dependence plot for {feature} - Class: {class_name}")
            plt.show()



## 8. Interpreting the Results
Example Interpretation for "Coconut"

P (Phosphorus): High values of P decrease the likelihood of predicting "coconut".

K (Potassium): The impact is varied, indicating no clear pattern.

N (Nitrogen): Low values of N decrease the likelihood of predicting "coconut", while high values have mixed effects.

ph (Soil pH): Low values of ph decrease the likelihood of predicting "coconut".

Conclusion for "Coconut"

Ideal soil conditions for coconut:

Low phosphorus content.

Higher nitrogen levels (with some caution).

Higher pH (less acidic).

## 9. Conclusion
Please keep in mind that the current model shows an accuracy of 0.82. For further analysis and improvements, additional data and feature engineering are essential. Incorporating more soil metrics, weather data, and historical crop yield information can provide a more comprehensive understanding of the factors affecting crop growth. Additionally, experimenting with different machine learning models and hyperparameter tuning may further enhance the model's predictive performance. Continuous validation with real-world data and collaboration with agricultural experts will ensure the model's practical applicability and accuracy.

For detailed instructions and guidance on how to run the analysis, please refer to the `README` file.

- **Dataset**: Please download `soil_measures.csv` for the data.
  
- **Exploratory Data Analysis (EDA) and Model Selection**: Refer to the `soil_measures_EDA_model_selection` file for comprehensive steps on performing EDA and selecting the appropriate models.
  
- **Model Application and Evaluation**: Consult the `soil_measures_model_results` file for applying the model, evaluating its performance, and visualizing SHAP results.