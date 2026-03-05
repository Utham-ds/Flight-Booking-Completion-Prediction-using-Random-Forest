# Flight Booking Completion Prediction

A supervised learning project that predicts whether customers **complete a flight booking** (`booking_complete`) using behavioural, itinerary and preference data. Implemented in Python with scikit‑learn to practise end‑to‑end ML pipelines, model evaluation and feature‑importance analysis.

## Project Structure

```text
flight-booking-completion-prediction/
├─ Flight Prediction using Random Forest.ipynb      # Main script: preprocessing, model training, evaluation
└─ customer_booking.csv  # Customer booking dataset (path configured in the script)
```

## Objectives

- Formulate booking completion as a binary classification problem.  
- Engineer meaningful features from raw booking and flight information.  
- Build a reproducible scikit-learn pipeline (preprocessing + Random Forest).  
- Evaluate performance with hold-out testing and cross-validation, and interpret results via feature importance.

## Approach

### Data and features

- Use fields such as `num_passengers`, `sales_channel`, `trip_type`, `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `route`, `booking_origin`, `flight_duration`, and customer preference flags (extra baggage, preferred seat, in-flight meals).  
- Add engineered features:  
  - `is_weekend` (Saturday/Sunday flights)  
  - `stay_per_day_of_lead` (length of stay relative to booking lead time)  
  - `route_origin` and `route_dest` extracted from the route code  

### Preprocessing and model

- Pass numeric features through unchanged and one-hot encode categorical features using `ColumnTransformer` and `OneHotEncoder(handle_unknown="ignore")`.  
- Train a `RandomForestClassifier` (200 trees with tuned depth/leaf settings) wrapped with the preprocessor in a single `Pipeline` for clean training and inference.

### Evaluation and interpretation

- Perform a train/test split with stratification on `booking_complete`.  
- Report accuracy, precision, recall, F1 score, ROC-AUC, and a full classification report.  
- Visualise the confusion matrix with seaborn.  
- Run 5-fold cross-validation with ROC-AUC to assess stability across splits.  
- Extract feature importances from the fitted Random Forest, plot the top 15, and export the full table to `feature_importance_random_forest.csv`.

## How to Run

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
Ensure customer_booking.csv is available and update the data_path variable at the top of Flight Prediction using Random Forest.ipynb if needed.

2. Execute the script:
   ```
   python Flight Prediction using Random Forest.ipynb
   ```
The script will train the model, print evaluation metrics, display the confusion‑matrix and feature‑importance plots, and write feature_importance_random_forest.csv to disk.

## Author
Utham Kumar Mohanlal
