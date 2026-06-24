# EV Adoption Insights & Prediction Hub

## Project Goal

This Streamlit application explores electric vehicle adoption behavior and predicts whether a consumer profile has a High, Medium, or Low likelihood of adopting an EV.

The app is intended to help users inspect the dataset, understand the feature meanings, submit a consumer profile, and interpret the model's predicted adoption likelihood.

## Dataset

The app loads `global_ev_adoption_behavior_2026_cleaned.csv` when available, otherwise it falls back to `global_ev_adoption_behavior_2026.csv`. If neither file is available, the app creates a mock dataset so the interface can still run.

Target column:

- `ev_adoption_likelihood`: adoption likelihood class with values such as High, Medium, and Low.

Numerical features:

- `age`
- `annual_income`
- `daily_commute_km`
- `weekly_travel_distance_km`
- `vehicle_age_years`
- `fuel_expense_per_month`
- `charging_station_accessibility`
- `nearest_charging_station_km`
- `home_charging_available`
- `electricity_cost_per_kwh`
- `environmental_awareness_score`
- `government_incentive_awareness`
- `technology_affinity_score`
- `range_anxiety_score`
- `battery_replacement_concern`
- `ev_knowledge_score`
- `previous_ev_experience`
- `monthly_energy_consumption_kwh`
- `monthly_charging_cost`

Categorical features:

- `education_level`
- `city_type`
- `current_vehicle_type`

Engineered features used by the Streamlit prediction form:

- `fuel_to_electricity_ratio`: `fuel_expense_per_month / (monthly_charging_cost + 1)`
- `charging_convenience_index`: `charging_station_accessibility / (nearest_charging_station_km + 1)`

## Preprocessing

The saved final pipeline in the notebook uses:

- `KNNImputer` and `RobustScaler` for numerical features.
- `SimpleImputer(strategy="most_frequent")` and `OrdinalEncoder` for `education_level`.
- `OneHotEncoder(drop="first", handle_unknown="ignore")` for nominal categorical features such as `city_type` and `current_vehicle_type`.
- `SMOTE` to handle class imbalance during model training.

## Model

The saved model file is `ev_adoption_model.pkl`. According to the notebook, the final pipeline is:

- Preprocessing
- SMOTE
- LogisticRegression with `C=1` and `penalty="l2"`

The Streamlit app loads this file with `joblib`. If the file is missing or cannot be loaded, the app uses a simple fallback simulation rule so the UI can still demonstrate the prediction workflow.

## Evaluation Metrics

The notebook compares several classifiers using 5-fold cross-validation and `f1_macro`.

Reported test macro-F1 scores from the model comparison:

- RandomForest: 77.16%
- DecisionTree: 72.94%
- LogisticRegression: 83.31%
- SVC: 81.05%
- KNeighbors: 71.80%
- NaiveBayes: 74.24%
- XGB: 82.52%
- CatBoost: 82.81%
- LightGBM: 82.16%

The Logistic Regression grid search reports a best cross-validation macro-F1 score of about 0.8337.

## Prediction Interpretation

The prediction classes mean:

- High: the profile appears more likely to adopt an EV.
- Medium: the profile has mixed indicators and may need stronger incentives, knowledge, or infrastructure.
- Low: the profile has barriers such as high range anxiety, charging inconvenience, low EV knowledge, or weak EV-related motivation.

The probability or confidence shown in the app should be treated as a model confidence estimate, not a guaranteed real-world probability.

## Limitations And Assumptions

- The model learns from the available dataset and may not generalize perfectly to all countries, cities, or market conditions.
- Survey-like scores such as awareness, range anxiety, and technology affinity are subjective.
- The app does not include live fuel prices, EV prices, electricity tariffs, local charger availability, or current government policy changes.
- SMOTE helps balance training data but creates synthetic examples, so results should be interpreted carefully.
- The fallback simulation mode is only for demonstration and is not the trained model.

## Chatbot Scope

The chatbot should answer questions about this project, the dataset, preprocessing, model, metrics, prediction outputs, and deployment configuration. If a user asks about unrelated topics, the chatbot should politely explain that the question is outside the project scope.
