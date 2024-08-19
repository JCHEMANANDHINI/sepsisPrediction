import pandas as pd
import numpy as np
from django.http import HttpResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import shap
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


def load_dataset(filepath):
    data = pd.read_csv(filepath)
    data_sampled = data.sample(frac=0.1, random_state=42)
    return data_sampled

def separate_features_target(sepsis_data):
    features = sepsis_data.drop(columns=['SepsisLabel', 'Unnamed: 0', 'Patient_ID'])
    target = sepsis_data['SepsisLabel']
    return features, target

def drop_nan_target(sepsis_data):
    return sepsis_data.dropna(subset=['SepsisLabel'])

def preprocess_features(features):
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)
    return features_scaled, imputer, scaler

def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)

def resample_data(features_train, target_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(features_train, target_train)

def tune_model(features_train, target_train):
    param_dist = {
        'n_estimators': randint(50, 150),
        'max_depth': randint(3, 7),
        'learning_rate': uniform(0.01, 0.19),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42, error_score='raise')
    random_search.fit(features_train, target_train)
    best_model = random_search.best_estimator_
    best_model.set_params(scale_pos_weight=(len(target_train) - sum(target_train)) / sum(target_train))
    best_model.fit(features_train, target_train)
    return best_model, random_search.best_params_

def evaluate_model(model, features_test, target_test):
    target_pred = model.predict(features_test)
    target_pred_prob = model.predict_proba(features_test)[:, 1]
    accuracy = accuracy_score(target_test, target_pred)
    precision = precision_score(target_test, target_pred)
    recall = recall_score(target_test, target_pred)
    f1 = f1_score(target_test, target_pred)
    roc_auc = roc_auc_score(target_test, target_pred_prob)
    return accuracy, precision, recall, f1, roc_auc

def shap_summary_plot(model, features_test, plot_filename='shap_summary_plot.png'):
    shap_explainer = shap.Explainer(model)
    shap_values = shap_explainer(features_test)
    shap.summary_plot(shap_values, features_test, show=False)  # Set show to False
    plt.savefig(plot_filename)  # Save the plot as a file
    plt.close()  # Close the plot to free up memory


def preprocess_user_input(user_input, imputer, scaler, feature_columns):
    user_input_df = pd.DataFrame([user_input], columns=feature_columns)
    user_input_imputed = imputer.transform(user_input_df)
    user_input_scaled = scaler.transform(user_input_imputed)
    return user_input_scaled

def predict_sepsis(user_input, model, imputer, scaler, feature_columns):
    processed_input = preprocess_user_input(user_input, imputer, scaler, feature_columns)
    prediction = model.predict(processed_input)
    prediction_prob = model.predict_proba(processed_input)[:, 1]
    return prediction[0], prediction_prob[0]

def sepsis_prediction_workflow(user_input_top_features, user_input_all_features=None):
    global best_xgb_model, feature_imputer, feature_scaler, feature_columns
    
    initial_prediction, initial_prediction_prob = predict_sepsis(user_input_top_features, best_xgb_model, feature_imputer, feature_scaler, feature_columns)
    # print(f'Initial Prediction: {initial_prediction} (1 indicates sepsis, 0 indicates no sepsis)')
    # print(f'Initial Prediction Probability: {initial_prediction_prob:.4f}')
    
    if initial_prediction == 1 and user_input_all_features is not None:
        detailed_prediction, detailed_prediction_prob = predict_sepsis(user_input_all_features, best_xgb_model, feature_imputer, feature_scaler, feature_columns)
        # print(f'Detailed Prediction: {detailed_prediction} (1 indicates sepsis, 0 indicates no sepsis)')
        # print(f'Detailed Prediction Probability: {detailed_prediction_prob:.4f}')
        return detailed_prediction, detailed_prediction_prob
    
    return initial_prediction, initial_prediction_prob

def example_input_usage():
    user_input_example = {
        'Age': 19, 'BUN': 30, 'BaseExcess': 0.5, 'AST': 24, 'Alkalinephos': 0.21,
        'Calcium': 7.4, 'Chloride': 40, 'Creatinine': 98, 'Bilirubin_total': 30, 'Platelets': 15,
        # Add other feature values here...
    }
    t=sepsis_prediction_workflow(user_input_example)
    return t

def run_sepsis_prediction(filepath):
    global feature_imputer, feature_scaler, feature_columns, best_xgb_model
    sepsis_data = load_dataset(filepath)
    sepsis_data = drop_nan_target(sepsis_data)
    features, target = separate_features_target(sepsis_data)
    features_scaled, feature_imputer, feature_scaler = preprocess_features(features)
    feature_columns = features.columns.tolist()  # Ensure feature_columns is a list
    features_train, features_test, target_train, target_test = split_data(features_scaled, target)
    features_train_resampled, target_train_resampled = resample_data(features_train, target_train)
    best_xgb_model, best_model_params = tune_model(features_train_resampled, target_train_resampled)
    accuracy, precision, recall, f1, roc_auc = evaluate_model(best_xgb_model, features_test, target_test)
    # print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}')
    shap_summary_plot(best_xgb_model, features_test)
    y=example_input_usage()
    return y


def main(request):
    pre=run_sepsis_prediction(r'C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\Dataset\Dataset.csv')
    return HttpResponse(pre[0])

# if __name__ == '__main__':
#     main()
    # run_sepsis_prediction(r'C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\Dataset\Dataset.csv')
    # print(pre)