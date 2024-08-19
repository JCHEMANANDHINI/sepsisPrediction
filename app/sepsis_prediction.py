import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import os
from tensorflow import keras
from sklearn.base import BaseEstimator, ClassifierMixin

# Placeholder for create_nn function
def create_nn(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define a custom KerasClassifier to match the loaded model
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs

    def fit(self, X, y, **kwargs):
        if self.build_fn is None:
            self.model_ = create_nn(X.shape[1])
        else:
            self.model_ = self.build_fn(**self.kwargs)
        self.model_.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return (self.model_.predict(X, **kwargs) > 0.5).astype('int32')

    def predict_proba(self, X, **kwargs):
        proba = self.model_.predict(X, **kwargs)
        return np.column_stack([1 - proba, proba])

# Define the feature names for the 10 known features
known_features = ['Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']

# Create a list of all 5994 feature names (as per the model information)
all_features = known_features + [f'feature_{i}' for i in range(len(known_features), 5994)]

def load_model(pickle_file_path):
    with open(pickle_file_path, 'rb') as model_file:
        return pickle.load(model_file)

def preprocess_user_input(input_dict):
    # Create a DataFrame with all 5994 features, filling unknown features with 0
    full_input = pd.DataFrame([[0] * len(all_features)], columns=all_features)
    for key, value in input_dict.items():
        full_input[key] = value
    return full_input

def predict_sepsis(input_dict, model):
    input_data_processed = preprocess_user_input(input_dict)
    prediction = model.predict(input_data_processed)
    probabilities = model.predict_proba(input_data_processed)
    return "Positive" if prediction[0] == 1 else "Negative", probabilities[0]

def get_user_input():
    input_data = {}
    print("Please enter the following information:")
    for feature in known_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_data[feature] = value
                break
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a number.")
    return input_data

def main():
    try:
        # Get the current script's directory
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the full path to the model file
        #model_path = os.path.join(script_dir, 'models\BHavs_ensemble_model.pkl')
        model_path = r"C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\model\BHavs_ensemble_model.pkl"
        # Load the model
        model = load_model(model_path)

        print(f"Model type: {type(model)}")
        if isinstance(model, VotingClassifier):
            print("This is an ensemble model with the following estimators:")
            for i, (name, estimator) in enumerate(model.named_estimators_.items()):
                print(f"  Estimator {i+1}: {type(estimator)}")
        print(f"Number of features expected by the model: {model.n_features_in_}")

        # Get user input
        user_input = get_user_input()

        # Make prediction
        result, probabilities = predict_sepsis(user_input, model)
        print(f"Sepsis prediction: {result}")
        print(f"Prediction probabilities: Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}")

    except FileNotFoundError:
        print("Model file not found. Please ensure 'BHavs_ensemble_model.pkl' is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure the model file is correct and all required libraries are installed.")

if __name__ == "__main__":
    main()