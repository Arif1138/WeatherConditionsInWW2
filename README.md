# Weather Maximum Temperature Prediction 🌡️

This repository contains an end-to-end machine learning project designed to predict the daily maximum temperature (`MaxTemp`) based on historical weather records and geographical station data. 

The project demonstrates a complete data science workflow, including data cleaning, relational merging, feature engineering, data leakage prevention, standard scaling, and the transition from a baseline linear model to an advanced ensemble algorithm.

## Features & Workflow

* **Data Wrangling:** Merges two large datasets (`Summary of Weather.csv` and `Weather Station Locations.csv`) using inner joins on station IDs.
* **Robust Preprocessing:** 
  * Replaces trace precipitation/snowfall values (`'T'`) with numeric zeroes.
  * Dynamically drops columns with excessive missing values using thresholding.
  * Filters for purely numerical data to prevent type errors during model training.
* **Leakage Prevention:** Removes redundant target-correlated features (like `MeanTemp`, `MIN`, `MAX`) to ensure the model learns generalized patterns rather than memorizing exact mathematical derivatives.
* **Feature Scaling:** Implements `StandardScaler` to normalize features (e.g., elevation vs. precipitation) to a uniform range, applying `fit_transform` to training data and `transform` to testing data to prevent data leakage.
* **Model Training & Selection:** 
  * Established a baseline using `LinearRegression` (R² Score: ~0.44).
  * Upgraded to `RandomForestRegressor` to capture non-linear, complex weather patterns, achieving a final R² Score of **~0.95**.
* **Production Ready:** Exports the trained model and the feature scaler via `joblib` for immediate deployment and inference.

## Prerequisites

Ensure you have Python 3.x installed along with the following libraries:
* `pandas`
* `scikit-learn`
* `joblib`

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Arif1138/weather-temp-prediction.git](https://github.com/Arif1138/weather-temp-prediction.git)
   cd weather-temp-prediction
