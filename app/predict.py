
# # if os.path.exists(file_path):
# #     with open(file_path, 'rb') as file:
# #         model = pickle.load(file)

# from django.shortcuts import render

# import pickle
# import os
# import numpy as np


# import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# import numpy as np
# file_path = r"C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\Final_ensemble_models.pkl"
# # file_path = r"C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\best_model.pkl"


# # Define a simple model
# # def create_model():
# #     model = Sequential()
# #     model.add(Dense(12, input_dim=8, activation='relu'))
# #     model.add(Dense(8, activation='relu'))
# #     model.add(Dense(1, activation='sigmoid'))
# #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #     return model

# # # Create and train the model
# # model = create_model()
# # X_train = np.random.rand(100, 8)
# # y_train = np.random.randint(2, size=100)
# # model.fit(X_train, y_train, epochs=10, verbose=0)

# # # Save the model
# # model.save('model.h5')

# # # Load the model
# # model = tf.keras.models.load_model('model.h5')

# # # Example prediction
# # X_test = np.random.rand(10, 8)
# # predictions = model.predict(X_test)
# # print(predictions)
# import pickle
# import dill
# import tensorflow as tf
# import pandas as pd
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense

# # Custom unpickler class
# # class CustomUnpickler(pickle.Unpickler):
# #     def find_class(self, module, name):
# #         if name == 'Sequential':
# #             return Sequential
# #         elif name == 'Dense':
# #             return Dense
# #         return super().find_class(module, name)

# # # Load the model from pickle file
# # with open(file_path, 'rb') as file:
# #     model = CustomUnpickler(file).load()

# # # Example prediction
# # import numpy as np
# # X_test = np.random.rand(10, 8)
# # predictions = model.predict(X_test)
# # print(predictions)


# def load_model():
#     try:
#         # with open(file_path, 'rb') as file:
#         #     model = pickle.load(file)
#         with open('best_model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except FileNotFoundError:
#         return None
#     except pickle.UnpicklingError:
#         return None
#     except Exception as e:
#         print(f"An error occurred while loading the model: {e}")
#         return None
# model = load_model()
# print(model)
# numerical_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
# categorical_cols = ['Gender', 'Age']

# import dill

# def load_model_and_preprocessor(pickle_file_path):
#     try:
#         print(f"Attempting to load model and preprocessor from: {pickle_file_path}")
#         with open(pickle_file_path, 'rb') as model_file:
#             loaded_objects = dill.load(model_file)
#         if not isinstance(loaded_objects, tuple) or len(loaded_objects) != 2:
#             print("The loaded objects are not as expected.")
#             return None, None
#         model, preprocessor = loaded_objects
#         if model is None:
#             print("Model is None.")
#         if preprocessor is None:
#             print("Preprocessor is None.")
#         if model and preprocessor:
#             print("Model and preprocessor loaded successfully.")
#         return model, preprocessor
#     except FileNotFoundError:
#         print(f"The file at path {pickle_file_path} was not found.")
#         return None, None
#     except Exception as e:
#         print(f"An error occurred while loading the model and preprocessor: {e}")
#         return None, None

# # import dill

# # pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/app/xgb_model.pkl'

# # try:
# #     with open(pickle_file_path, 'rb') as model_file:
# #         model, preprocessor = dill.load(model_file)
# #     print("Model and Preprocessor loaded successfully.")
# #     print(f"Model: {type(model)}")
# #     print(f"Preprocessor: {type(preprocessor)}")
# # except FileNotFoundError:
# #     print(f"The file at path {pickle_file_path} was not found.")
# # except Exception as e:
# #     print(f"An error occurred while loading the model and preprocessor: {e}")


# # def preprocess_user_input(input_list, preprocessor):
# #             if preprocessor is None:
# #                 raise ValueError("Preprocessor is not loaded.")
    
# #             input_df = pd.DataFrame([input_list])  # Assuming input_list is a single row of input data
# #             input_data_processed = preprocessor.transform(input_df)
# #             return input_data_processed

# # def predict_sepsis(input_list, model, preprocessor):
# #             input_data_processed = preprocess_user_input(input_list, preprocessor)
# #             prediction = model.predict(input_data_processed)
# #             return prediction
# from django.shortcuts import render
# import dill
# import numpy as np
# import pandas as pd
# # from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# import dill

# def load_model_and_preprocessor(pickle_file_path):
#     try:
#         print(f"Attempting to load model and preprocessor from: {pickle_file_path}")
#         with open(pickle_file_path, 'rb') as model_file:
#             loaded_objects = dill.load(model_file)
#         if not isinstance(loaded_objects, tuple) or len(loaded_objects) != 2:
#             print("The loaded objects are not as expected.")
#             return None, None
#         model, preprocessor = loaded_objects
#         if model is None:
#             print("Model is None.")
#         if preprocessor is None:
#             print("Preprocessor is None.")
#         if model and preprocessor:
#             print("Model and preprocessor loaded successfully.")
#         return model, preprocessor
#     except FileNotFoundError:
#         print(f"The file at path {pickle_file_path} was not found.")
#         return None, None
#     except Exception as e:
#         print(f"An error occurred while loading the model and preprocessor: {e}")
#         return None, None
# from django.shortcuts import render
# import dill
# import pandas as pd

# def preprocess_user_input(input_list, preprocessor):
#     if preprocessor is None:
#         raise ValueError("Preprocessor is not loaded.")
    
#     input_df = pd.DataFrame([input_list])  # Assuming input_list is a single row of input data
#     input_data_processed = preprocessor.transform(input_df)
#     return input_data_processed

# def predict_sepsis(input_list, model, preprocessor):
#     if model is None or preprocessor is None:
#         raise ValueError("Model or preprocessor is not loaded.")
    
#     input_data_processed = preprocess_user_input(input_list, preprocessor)
#     prediction = model.predict(input_data_processed)
#     return prediction

# def predict_view(request):
#     # Load the model and preprocessor
#     pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/app/best_model.pkl'
#     model, preprocessor = load_model_and_preprocessor(pickle_file_path)

#     if model is None or preprocessor is None:
#         return render(request, 'predict.html', {'error': 'Model or preprocessor could not be loaded.'})

#     # Example user input (replace with actual input handling)
#     user_input_list = [0.5, 1.2, 3.4, 0.2, 1.5, 2.3, 3.1, 0.9]

#     try:
#         prediction = predict_sepsis(user_input_list, model, preprocessor)
#         return render(request, 'predict.html', {'predictions': prediction})
#     except Exception as e:
#         return render(request, 'predict.html', {'error': f'An error occurred during prediction: {e}'})

# def predict(request):


#     if request.method == 'POST':
#         # Print all POST data for debugging
#         print("POST data:", request.POST)

#         # Retrieve POST data
#         val1 = request.POST.get('HR')
#         val7 = request.POST.get('PO')
#         val2 = request.POST.get('tem')
#         val3 = request.POST.get('SBP')
#         val4 = request.POST.get('MAP')
#         val5 = request.POST.get('DBP')
#         val6 = request.POST.get('Res')
#         # val7 = request.POST.get('Etco2')
#         # val7 = request.POST.get('PO')
#         val8 = request.POST.get('Age')
#         # if request.POST.get('Gender')=="Male":
#         #     val10 = 0
#         # else:
#         #     val10 = 1
#         val9 = request.POST.get('Gender')
#         val10=request.POST.get("O2Sat")
#         data = [[val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]]
#         print(data)

#         # Check for None values and provide default values or handle errors
#         # if any(v is None for v in [val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]):
#         #     return HttpResponse("Error: Missing required fields.")

#         # Convert values to floats
#         # try:
#         #     val1 = float(val1)
#         #     val2 = float(val2)
#         #     val3 = float(val3)
#         #     val4 = float(val4)
#         #     val5 = float(val5)
#         #     val6 = float(val6)
#         #     val7 = float(val7)
#         #     val8 = float(val8)
#         #     val9 = float(val9)
#         #     val10 = float(val10)
#         # except ValueError as e:
#         #     return HttpResponse(f"ValueError: {e}")

#         # Prepare data for prediction
#         # data = [[val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]]
#         # print(data)

        
# # Function to load the model and preprocessor and make predictions
#         # def load_model_and_preprocessor(pickle_file_path):
#         #     with open(pickle_file_path, 'rb') as model_file:
#         #         data = pickle.load(model_file)
#         #     return data['model'], data['preprocessor']


        
        

#         # Function to preprocess user input
#     #     def preprocess_user_input(input_list, preprocessor):
#     # # Ensure the input list is in the correct order of features
#     #         feature_order = numerical_cols + categorical_cols
#     # # Convert the input list to a DataFrame
#     #         input_df = pd.DataFrame([input_list], columns=feature_order)
#     # # Apply the preprocessor
#     #         input_data_processed = preprocessor.transform(input_df)
#     #         return input_data_processed
        

#         # def predict_sepsis(input_list, model, preprocessor):
#         #     input_data_processed = preprocess_user_input(input_list, preprocessor)
#         #     prediction = model.predict(input_data_processed)
#         #     return prediction
#         # # Load the model and preprocessor
#         # def predict_view(request):
#     # Load the model and preprocessor
    
    

#     # # Your prediction logic here
#     # prediction = model.predict([0.5, 1.2, 3.4, 0.2, 1.5, 2.3, 3.1, 0.9])
#     # return HttpResponse({'prediction': prediction})
#     # pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/Final_ensemble_model.pkl'
#     # pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/Final_ensemble_model.pkl'
#     # model, preprocessor = load_model_and_preprocessor(pickle_file_path)

#     # if model is None or preprocessor is None:
#     #     return render(request, 'app/prediction.html', {'error': 'Model or preprocessor could not be loaded.'})

#     # # Example user input (replace with actual input handling)
#     # user_input_list = [0.5, 1.2, 3.4, 0.2, 1.5, 2.3, 3.1, 0.9]

#     # try:
#     #     prediction = predict_sepsis(user_input_list, model, preprocessor)
#     #     return render(request, 'app/prediction.html', {'predictions': prediction})
#     # except Exception as e:
#     #     return render(request, 'app/prediction.html', {'error': f'An error occurred during prediction: {e}'})
#     # try:
#     #     with open(pickle_file_path, 'rb') as model_file:
#     #         model, preprocessor = pickle.load(model_file)
#     #         print("Model and Preprocessor loaded successfully.")
#     #         print(f"Model: {type(model)}")
#     #         print(f"Preprocessor: {type(preprocessor)}")
#     # except FileNotFoundError:
#     #     print(f"The file at path {pickle_file_path} was not found.")
#     # except Exception as e:
#     #     print(f"An error occurred while loading the model and preprocessor: {e}")
#     # pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/app/best_model.pkl'

#     # try:
#     #     with open(pickle_file_path, 'rb') as model_file:
#     #         model, preprocessor = dill.load(model_file)
#     #     print("Model and Preprocessor loaded successfully.")
#     #     print(f"Model: {type(model)}")
#     #     print(f"Preprocessor: {type(preprocessor)}")
#     # except FileNotFoundError:
#     #     print(f"The file at path {pickle_file_path} was not found.")
#     # except Exception as e:
#     #     print(f"An error occurred while loading the model and preprocessor: {e}")






#     # pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/xgb_model.pkl'
#     # model, preprocessor = load_model_and_preprocessor(pickle_file_path)

#     # if model is None and preprocessor is None:
#     #     return render(request, 'predict.html', {'error': 'Model or preprocessor could not be loaded.'})

#     # Example user input (replace with actual input handling)
    
#     # user_input_list = [0.5, 1.2, 3.4, 0.2, 1.5, 2.3, 3.1, 0.9]

#     # try:
#     #     prediction = predict_sepsis(user_input_list, model, preprocessor)
#     #     return render(request, 'app/prediction.html', {'predictions': prediction})
#     # except Exception as e:
#     #     return render(request, 'app/prediction.html', {'error': f'An error occurred during prediction: {e}'})
#         # print(predict_view())

#         # model, preprocessor = load_model_and_preprocessor(file_path)

# # # Example user input as a list
#         # user_input_list = [80, 98, 36.6, 120, 85, 70, 20, 0, 45]  # Example values in the order of the features

# # # Make the prediction
#         # prediction = predict_sepsis(user_input_list, model, preprocessor)
#         # print(prediction)
#         file_path = r"C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\Final_ensemble_models.pkl"
        
# # Example user input as a list
#         user_input_list = [150, 100, 43.9, 150, 95, 90, 30, 0, 75]  # Example values in the order of the features
#         # model, preprocessor = load_model_and_preprocessor(file_path)
# # Make the prediction
#         # prediction = predict_sepsis(user_input_list, model, preprocessor)
#         # print("Sepsis Prediction:", "Positive" if prediction[0] == 1 else "Negative")
#         # data = [[100, 100, 43.9, 150, 95, 90, 20, 0, 75,100, 100, 43.9, 150, 95, 90, 20, 0, 75,100, 100, 43.9, 150, 95, 90, 20, 0, 75,100, 100, 43.9, 150, 95, 90, 20, 0, 75,100, 100, 43.9, 150]]
#         data=[[150, 100, 43.9, 150, 95, 90, 30, 0, 75]]
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         #     preprocessor = pickle.load(file)
#         # print(model,preprocessor)
#         # prediction = predict_sepsis(user_input_list, model, preprocessor)
        
#         if model is not None:
#             try:
#                 prediction = model.predict(data)
#                 return HttpResponse(f'Prediction: {prediction[0]}')
#             except Exception as e:
#                 return HttpResponse(f"Prediction error: {e}")
#         else:
#             return HttpResponse("Model loading error.")
#     else:
#         return render(request, 'app/prediction.html')





# # with open('model_and_key_components.pkl', 'rb') as file:
# #     loaded_components = pickle.load(file)
# # model = loaded_components['model']

 
# # @csrf_exempt
# # def prediction(request):
# #     return render(request, 'app/predict.html')
# # # @app.route('/predict', methods=['GET', 'POST'])
# # @csrf_exempt
# # def predict(request):
# #     if request.method=='POST':
# #         val1 = request.POST.get['prg_ctr']
# #         val2 = request.form['pl_glucose_conc.']
# #         val3 = request.form['bp']
# #         val4 = request.form['skin_thick']
# #         val5 = request.form['insulin']
# #         val6 = request.form['bmi']
# #         val7 = request.form['diabetes']
# #         val8 = request.form['age']
# #     arr = np.array([val1, val2, val3, val4,val5, val6, val7, val8])
# #     pred = model.predict([arr])
# #     if pred==[1]:
# #         output="Sepsis is Present"
# #         prompt = "why did i got sespsis?"
# #     else:
# #         output="Sepsis is Absent"
# #         prompt = "What are the precautions to be taken for preventing sepsis?"
# #     # response = llm_model.generate_content(prompt)

# #     return render('app/predict.html', data=output)#+" "+response.text)


# #         # <!-- <a href="{% url 'main' %}" class="button">Prediction</a> -->
# def predict_view(request):
#     # Load the model and preprocessor
#     pickle_file_path = 'C:/Users/jchem/OneDrive/Documents/cts/sepsis/app/best_model.pkl'
#     model, preprocessor = load_model_and_preprocessor(pickle_file_path)

#     if model is None or preprocessor is None:
#         return render(request, 'predict.html', {'error': 'Model or preprocessor could not be loaded.'})

#     # Example user input (replace with actual input handling)
#     user_input_list = [0.5, 1.2, 3.4, 0.2, 1.5, 2.3, 3.1, 0.9]

#     try:
#         prediction = predict_sepsis(user_input_list, model, preprocessor)
#         return render(request, 'predict.html', {'predictions': prediction})
#     except Exception as e:
#         return render(request, 'predict.html', {'error': f'An error occurred during prediction: {e}'})



import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Define the feature names
# feature_names = [
#     'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender'
# ]

# # Separate numerical and categorical features
# numerical_features = [
#     'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age'
# ]
# categorical_features = ['Gender']

# def load_model_and_preprocessor(pickle_file_path):
#     with open(pickle_file_path, 'rb') as model_file:
#         data = pickle.load(model_file)
#     return data['model'], data['preprocessor']

# def preprocess_user_input(input_dict, preprocessor):
#     input_df = pd.DataFrame([input_dict])
#     input_data_processed = preprocessor.transform(input_df)
#     return input_data_processed

# def predict_sepsis(input_dict, model, preprocessor):
#     input_data_processed = preprocess_user_input(input_dict, preprocessor)
#     prediction = model.predict(input_data_processed)
#     return "Positive" if prediction[0] == 1 else "Negative"

# # def get_user_input():
# #     input_data = {}
    
# #     print("Please enter the following information:")
    
# #     for feature in numerical_features:
# #         while True:
# #             try:
# #                 value = float(input(f"{feature}: "))
# #                 input_data[feature] = value
# #                 break
# #             except ValueError:
# #                 print("Invalid input. Please enter a number.")
    
# #     # For Gender, we'll assume it's binary (0 or 1) for simplicity
# #     while True:
# #         gender = input("Gender (0 for female, 1 for male): ")
# #         if gender in ['0', '1']:
# #             input_data['Gender'] = int(gender)
# #             break
# #         else:
# #             print("Invalid input. Please enter 0 or 1.")
    
# #     return input_data

# def load_dataset(file_path):
#     df = pd.read_csv(file_path)
#     return df
# def evaluate_model(model, preprocessor, df):
#     X = df[feature_names]
#     y = df['SepsisLabel']
    
#     X_processed = preprocessor.transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     print(f"Model Accuracy: {accuracy:.2f}")

# # Load the model and preprocessor



# def predict(request):

#     if request.method == 'POST':
#         # Print all POST data for debugging
#         print("POST data:", request.POST)

#         # Retrieve POST data
#         val1 = request.POST.get('HR')
#         val7 = request.POST.get('PO')
#         val2 = request.POST.get('tem')
#         val3 = request.POST.get('SBP')
#         val4 = request.POST.get('MAP')
#         val5 = request.POST.get('DBP')
#         val6 = request.POST.get('Res')
#         # val7 = request.POST.get('Etco2')
#         # val7 = request.POST.get('PO')
#         val8 = request.POST.get('Age')
#         # if request.POST.get('Gender')=="Male":
#         #     val10 = 0
#         # else:
#         #     val10 = 1
#         val9 = request.POST.get('Gender')
#         val10=request.POST.get("O2Sat")
#         data = [[val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]]
#         print(data)

#         # Check for None values and provide default values or handle errors
#         # if any(v is None for v in [val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]):
#         #     return HttpResponse("Error: Missing required fields.")

#         # Convert values to floats
#         # try:
#         #     val1 = float(val1)
#         #     val2 = float(val2)
#         #     val3 = float(val3)
#         #     val4 = float(val4)
#         #     val5 = float(val5)
#         #     val6 = float(val6)
#         #     val7 = float(val7)
#         #     val8 = float(val8)
#         #     val9 = float(val9)
#         #     val10 = float(val10)
#         # except ValueError as e:
#         #     return HttpResponse(f"ValueError: {e}")

#         # Prepare data for prediction
#         # data = [[val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]]
#         # print(data)

        
# # Function to load the model and preprocessor and make predictions
#         # def load_model_and_preprocessor(pickle_file_path):
#         #     with open(pickle_file_path, 'rb') as model_file:
#         #         data = pickle.load(model_file)
#         #     return data['model'], data['preprocessor']

#     model, preprocessor = load_model_and_preprocessor('app/Final_ensemble_model.pkl')

# # Load the dataset
#     # dataset = load_dataset('data\Dataset.csv')

# # Evaluate the model
#     # evaluate_model(model, preprocessor, dataset)

# # Get user input
#     user_input ={'Hour':2, 'HR':78, 'O2Sat':0, 'Temp':0, 'SBP':42.5, 'MAP':0, 'DBP':100, 'Resp':98, 'Age':25,'Gender':0}
        
# # Make prediction
#     result = predict_sepsis(user_input, model, preprocessor)
#     print(f"Sepsis prediction: {result}")
#     return HttpResponse(f'Prediction: {result}')
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import os
from tensorflow import keras
from sklearn.base import BaseEstimator, ClassifierMixin
from django.http import HttpResponse

# Placeholder for create_nn function
# def create_nn(input_shape):
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Define a custom KerasClassifier to match the loaded model
# class KerasClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, build_fn=None, **kwargs):
#         self.build_fn = build_fn
#         self.kwargs = kwargs

#     def fit(self, X, y, **kwargs):
#         if self.build_fn is None:
#             self.model_ = create_nn(X.shape[1])
#         else:
#             self.model_ = self.build_fn(**self.kwargs)
#         self.model_.fit(X, y, **kwargs)
#         return self

#     def predict(self, X, **kwargs):
#         return (self.model_.predict(X, **kwargs) > 0.5).astype('int32')

#     def predict_proba(self, X, **kwargs):
#         proba = self.model_.predict(X, **kwargs)
#         return np.column_stack([1 - proba, proba])

# # Define the feature names for the 10 known features
# known_features = ['Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']

# # Create a list of all 5994 feature names (as per the model information)
# all_features = known_features + [f'feature_{i}' for i in range(len(known_features), 5994)]

# def load_model(pickle_file_path):
#     with open(pickle_file_path, 'rb') as model_file:
#         return pickle.load(model_file)

# def preprocess_user_input(input_dict):
#     # Create a DataFrame with all 5994 features, filling unknown features with 0
#     full_input = pd.DataFrame([[0] * len(all_features)], columns=all_features)
#     for key, value in input_dict.items():
#         full_input[key] = value
#     return full_input

# def predict_sepsis(input_dict, model):
#     input_data_processed = preprocess_user_input(input_dict)
#     prediction = model.predict(input_data_processed)
#     probabilities = model.predict_proba(input_data_processed)
#     return "Positive" if prediction[0] == 1 else "Negative", probabilities[0]

# def get_user_input():
#     input_data = {}
#     print("Please enter the following information:")
#     for feature in known_features:
#         while True:
#             try:
#                 value = float(input(f"{feature}: "))
#                 input_data[feature] = value
#                 break
#             except ValueError:
#                 print(f"Invalid input for {feature}. Please enter a number.")
#     return input_data

# def main():
# # def predict(request):
#     print("HI")
#     if True:
#     # if request.method == 'POST':
#     #     # Print all POST data for debugging
#     #     print("POST data:", request.POST)

#     #     # Retrieve POST data
#     #     val1 = request.POST.get('HR')
#     #     val7 = request.POST.get('PO')
#     #     val2 = request.POST.get('tem')
#     #     val3 = request.POST.get('SBP')
#     #     val4 = request.POST.get('MAP')
#     #     val5 = request.POST.get('DBP')
#     #     val6 = request.POST.get('Res')
#     #     # val7 = request.POST.get('Etco2')
#     #     # val7 = request.POST.get('PO')
#     #     val8 = request.POST.get('Age')
#     #     # if request.POST.get('Gender')=="Male":
#     #     #     val10 = 0
#     #     # else:
#     #     #     val10 = 1
#     #     val9 = request.POST.get('Gender')
#     #     val10=request.POST.get("O2Sat")
#     #     data = [[val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]]
#     #     print(data)
#         # try:
#         try:
#         # Get the current script's directory
#         #script_dir = os.path.dirname(os.path.abspath(__file__))
        
#         # Construct the full path to the model file
#         #model_path = os.path.join(script_dir, 'models\BHavs_ensemble_model.pkl')
#             # model_path = r"app\model_and_key_components.pkl"
#         # Load the model
#             model = load_model("app\model_and_key_components.pkl")

#             print(f"Model type: {type(model)}")
#             if isinstance(model, VotingClassifier):
#                 print("This is an ensemble model with the following estimators:")
#                 for i, (name, estimator) in enumerate(model.named_estimators_.items()):
#                     print(f"  Estimator {i+1}: {type(estimator)}")
#             print(f"Number of features expected by the model: {model.n_features_in_}")
            
#         # Get user input
#             user_input = {'Hour':2, 'HR':78, 'O2Sat':0, 'Temp':0, 'SBP':42.5, 'MAP':0, 'DBP':100, 'Resp':98, 'Age':25,'Gender':0}

#         # Make prediction
#             result, probabilities = predict_sepsis(user_input, model)
#             print(f"Sepsis prediction: {result}")
#             print(f"Prediction probabilities: Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}")
#             # return HttpResponse(f'Prediction: probabilities[0]')
#         except FileNotFoundError:
#             print("Model file not found. Please ensure 'BHavs_ensemble_model.pkl' is in the same directory as this script.")
#         except Exception as e:
#             print(f"An error occurred: {str(e)}")
#             print("Please make sure the model file is correct and all required libraries are installed.")
#             # return HttpResponse(f'Prediction:')
# import xgboost as xgb
# import pickle
# from sklearn.preprocessing import StandardScaler  # or any other scaler you are using

# # Assuming you have already trained the model and stored it in a variable called `model`
# model = xgb.XGBClassifier()  # Example of defining a model
# # Training code here...
# scaler = StandardScaler()  # or the specific scaler you are using


# xgboost_model_filename = 'app/xgboost_model.pkl'
# with open(xgboost_model_filename, 'wb') as file:
#     pickle.dump(model, file)
#     # pickle.load(file)
# print("XGBoost model saved successfully.")

# # Load the saved XGBoost model from the file
# with open(xgboost_model_filename, 'rb') as file:
#     loaded_model = pickle.load(file)

# # Define the hardcoded user input values for prediction (ensure it matches encoded gender columns)
# user_input_values = [75, 36.8, 16, 80, 120, 93, 98, 35, 1, 0]  # Example values (last two for Gender_0.6, Gender_1.0)
# columns = ['HR', 'Temp', 'Resp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age', 'Gender_0.6', 'Gender_1.0']

# # Create a DataFrame from the user input
# user_input_df = pd.DataFrame([user_input_values], columns=columns)

# # Ensure the same feature scaling and encoding is applied as during training
# user_input_scaled = scaler.fit_transform(user_input_df.drop(columns=['Gender_0.6', 'Gender_1.0']))
# user_input_scaled = pd.DataFrame(user_input_scaled, columns=user_input_df.columns.drop(['Gender_0.6', 'Gender_1.0']))

# # Add the Gender columns back to the scaled data
# user_input_scaled = pd.concat([user_input_scaled, user_input_df[['Gender_0.6', 'Gender_1.0']]], axis=1)

# # Make predictions using the loaded model
# predicted_output = loaded_model.predict(user_input_scaled)
# predicted_probability = loaded_model.predict_proba(user_input_scaled)[:, 1]

# # Display the prediction results
# print("Predicted Class (0=No Sepsis, 1=Sepsis):", predicted_output[0])
# print("Predicted Probability of Sepsis:", predicted_probability[0])


# import pickle
# import pandas as pd
from django.shortcuts import render
# # Load the saved StandardScaler
# scaler_filename = 'app\scaler_final.pkl'
# with open(scaler_filename, 'rb') as file:
#     scaler = pickle.load(file)

# # Load the saved XGBoost model
# xgboost_model_filename = r'app\best_xgb_model_lasttt.pkl'
# with open(xgboost_model_filename, 'rb') as file:
#     loaded_model = pickle.load(file)

# # Define the hardcoded user input values for prediction (ensure it matches encoded gender columns)
# user_input_values = [75, 36.8, 16, 80, 120, 93, 98, 35, 1, 0]  # Example values (last two for Gender_0.6, Gender_1.0)
# columns = ['HR', 'Temp', 'Resp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age', 'Gender_0.6', 'Gender_1.0']

# # Create a DataFrame from the user input
# user_input_df = pd.DataFrame([user_input_values], columns=columns)

# # Ensure the same feature scaling and encoding is applied as during training
# user_input_scaled = scaler.transform(user_input_df.drop(columns=['Gender_0.6', 'Gender_1.0']))
# user_input_scaled = pd.DataFrame(user_input_scaled, columns=user_input_df.columns.drop(['Gender_0.6', 'Gender_1.0']))

# # Add the Gender columns back to the scaled data
# user_input_scaled = pd.concat([user_input_scaled, user_input_df[['Gender_0.6', 'Gender_1.0']]], axis=1)

# # Make predictions using the loaded model
# predicted_output = loaded_model.predict(user_input_scaled)
# predicted_probability = loaded_model.predict_proba(user_input_scaled)[:, 1]

# # Display the prediction results
# print("Predicted Class (0=No Sepsis, 1=Sepsis):", predicted_output[0])
# print("Predicted Probability of Sepsis:", predicted_probability[0])

# HTML generation code to display the prediction results

def generate_html(predictions, output, data, name, gen,prob,pred):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
    <link rel="icon" href="https://i.ibb.co/7GQw6SW/Screenshot-2024-07-25-203806.png" type="image/icon type">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">
    
        
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            .download-btn{{
                margin-left:1000px;
            }}
            h1 {{
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f4f4f4;
            }}
            .header {{
                display: flex;
                align-items: center;
            }}
            .header img {{
                width: 200px;
                margin-right: 1500px;
            }}
            /*.header h2 {{
                margin: 0;
            }}*/
            .user-details {{
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            #mytable {{
            width: 50%;
                border: none;
                border-collapse: collapse;
            }}
            #mytable th, #mytable td {{
                border: none; /* No borders for mytable */
            }}
        </style>
    </head>
    <body>
        <button class="download-btn" onclick="downloadPDF()">Download as PDF</button>
        <div class="header">
            <img src="https://i.ibb.co/Fnf5xdM/Screenshot-2024-07-25-203453.png" alt="Sepsis Foresee Logo">
            //<h2>Sepsis Foresee</h2>
        </div>
        
        <table id="mytable">
        <tr>
        <td><b>Name:</b></td>
        <td>{name}</td>
        </tr>
        <tr>
        <td><b>Age:</b></td>
        <td>{data[7]}</td>
        </tr>
        <tr>
        <td><b>Gender:</b></td>
        <td>{gen}</td>
        </tr>
                <tr>
                   <th><b>RESULT : </b></th>
                   <th><B>{pred}</b></th><br>
                </tr>
                <tr>
                <th><b>CONFIDENCE : </b></th>
                <th>{prob}</th><br>
                </tr>
                
        </table>

   
        <h1>Prediction Results</h1>
        <tbody>
                <tr>
                <th><b>RESULT : </b></th>

        """
    for i in range(len(output)):
                    html_content += f"""
                    <p>{output[i]}</p>
                </tr>
    """
    html_content+=f"""
        <table id="detail">
            <thead>
                <tr>
                    <th>Vital Sign</th>
                    <th>Value</th>
                    <th>Level</th>
                </tr>
            </thead>
            
                """
    
    

    for vital_sign, level in predictions.items():
        html_content += f"""
                <tr>
                    <td>{vital_sign}</td>
                    <td>{level[0]}</td>
                    <td>{level[1]}</td>
                </tr>
        """
    html_content += """
    </tbody>
        </table>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/1.0.272/jspdf.debug.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.4.1/html2canvas.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.23/jspdf.plugin.autotable.min.js"></script>
        <script>
    function downloadPDF() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Add logo
        const img = new Image();
        img.src = 'https://i.ibb.co/Fnf5xdM/Screenshot-2024-07-25-203453.png';
        doc.addImage(img, 'PNG', 140, 10, 50, 20);
        
        // Add title
        doc.setFontSize(8);
        doc.text('PREDICT . PROTECT . PREVENT', 145, 35);

        doc.setFontSize(30);
        doc.text('REPORT SUMMARY', 10, 30);

        // Add user details
        //doc.setFontSize(12);
        //doc.text('Name: {name}', 10, 50);
        //doc.text('Age: {data[7]}', 10, 60);
        
        doc.setFontSize(15);
        doc.text('Details :', 15, 60);
        
        doc.autoTable({
            html: '#mytable',
            startY: 70,
        });

        doc.setFontSize(15);
        doc.text('Abnormalities :', 15, 120);
        // Generate table content
        doc.autoTable({
            html: '#detail',
            theme: 'grid',
            startY: 130,
            headStyles: { fillColor: [0, 0, 0] },
            alternateRowStyles: { fillColor: [255, 255, 255] }
        });

        // Save the generated PDF
        doc.save('prediction_results.pdf');
    }
</script>

    </body>
    </html>
    """
    
    # Define the path to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    
    # Create the templates directory if it doesn't exist
    os.makedirs(templates_dir, exist_ok=True)
    
    # Define the path to the HTML file within the templates directory
    file_path = os.path.join(templates_dir, 'prediction_results.html')

    # Write the HTML content to the file
    with open(file_path, 'w') as file:
        file.write(html_content)

    return file_path
#             </tbody>
#         </table>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
#         <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/1.0.272/jspdf.debug.js"></script>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.4.1/html2canvas.js"></script>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.23/jspdf.plugin.autotable.min.js"></script>
#         <script>
#     function downloadPDF() {
#         const { jsPDF } = window.jspdf;
#         const doc = new jsPDF();
        
#         // Add logo
#         const img = new Image();
#         img.src = 'https://i.ibb.co/Fnf5xdM/Screenshot-2024-07-25-203453.png';
#         doc.addImage(img, 'PNG', 10, 10, 30, 30);
        
#         // Add title
#         doc.setFontSize(18);
#         doc.text('Sepsis Foresee', 50, 25);
        
        
        

       

#         // Generate table content
#         doc.setFontSize(12);
#         doc.autoTable({
#         html: 'table',
#             theme: 'grid',
#             startY: 80,
#             headStyles: { fillColor: [0, 0, 0] },
#             alternateRowStyles: { fillColor: [255, 255, 255] }
#         });

#         // Save the generated PDF
#         doc.save('prediction_results.pdf');
#     }
# </script>

#     </body>
#     </html>
#     """
    
#     # Define the path to the templates directory
#     templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    
#     # Create the templates directory if it doesn't exist
#     os.makedirs(templates_dir, exist_ok=True)
    
#     # Define the path to the HTML file within the templates directory
#     file_path = os.path.join(templates_dir, 'prediction_results.html')

#     # Write the HTML content to the file
#     with open(file_path, 'w') as file:
#         file.write(html_content)

#     return file_path
#     html_content += """
#             </tbody>
#         </table>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
#         <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/1.0.272/jspdf.debug.js"></script>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.4.1/html2canvas.js"></script>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.23/jspdf.plugin.autotable.min.js"></script>
#     <script>
        
    
#     function downloadPDF() {
#         const { jsPDF } = window.jspdf;
#         const doc = new jsPDF();

#         // Generate table content
#         doc.autoTable({
#             html: 'table',
#             theme: 'grid',
#             startY: 20,
#             headStyles: { fillColor: [0, 0, 0] }, // Light grey background for the header
#             alternateRowStyles: { fillColor: [255, 255, 255] } // White background for the body
#         });

#         // Save the generated PDF
#         doc.save('prediction_results.pdf');
#     }
# </script>

    
#     </body>
#     </html>
#     """
#       # Define the path to the templates directory
#     templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    
#     # Create the templates directory if it doesn't exist
#     os.makedirs(templates_dir, exist_ok=True)
    
#     # Define the path to the HTML file within the templates directory
#     file_path = os.path.join(templates_dir, 'prediction_results.html')

#     # Write the HTML content to the file
#     with open(file_path, 'w') as file:
#         file.write(html_content)

#     return file_path

    # return html_content


# # Example prediction results
# predictions = {
#     "HR": "Normal",
#     "Temp": "High (Fever)",
#     "SBP": "High",
#     "MAP": "Normal",
#     "DBP": "Normal",
#     "O2Sat": "Normal"
# }

# Generate HTML content



def predict_levels(HR=None, resp=None, tem=None, SBP=None, MAP=None, DBP=None, O2Sat=None):
    predictions = {}

    # Heart Rate (HR) Levels
    if HR is not None:
        if HR < 30:
            predictions['Heart Rate'] = [HR,'Severely Low']
        elif 30 <= HR < 60:
            predictions["Heart Rate"] = [HR,'Low']
        elif 60 <= HR <= 100:
            predictions['Heart Rate'] = [HR,'Normal']
        elif 100 < HR <= 200:
            predictions['Heart Rate'] = [HR,'High']
        else:
            predictions['Heart Rate'] = [HR,'Severely High']

    if resp is not None:
        if resp < 10:
            predictions['Respiration Rate']=[resp,'Severely Low']
        elif 10 <= resp < 16:
            predictions['Respiration Rate']=[resp,'Low']
        elif 16 <= resp <= 20:
            predictions['Respiration Rate']=[resp,'Normal']
        elif 20 < resp <= 30:
            predictions['Respiration Rate']=[resp,'High']
        else:
            predictions['Respiration Rate']=[resp,'Severely High']
    # Temperature (tem) Levels
    if tem is not None:
        if tem < 95:
            predictions['Temperature (celsius)'] = [tem,'Severely Low (Hypothermia)']
        elif 95 <= tem <= 99.5:
            predictions['Temperature (celsius)'] = [tem,'Normal']
        elif 99.5 < tem <= 105:
            predictions['Tempature (celsius)'] = [tem,'High (Fever)']
        else:
            predictions['Temperature (celsius)'] = [tem,'Severely High (Hyperthermia)']

    # Diastolic Blood Pressure (DBP) Levels
    if DBP is not None:
        if DBP < 40:
            predictions['Diastolic Blood Pressure'] = [DBP,'Severely Low']
        elif 40 <= DBP < 60:
            predictions['Diastolic Blood Pressure'] = [DBP,'Low']
        elif 60 <= DBP <= 80:
            predictions['Diastolic Blood Pressure'] = [DBP,'Normal']
        elif 80 < DBP <= 120:
            predictions['Diastolic Blood Pressure'] = [DBP,'High']
        else:
            predictions['Diastolic Blood Pressure'] = [DBP,'Severely High']
    # Systolic Blood Pressure (SBP) Levels
    if SBP is not None:
        if SBP < 70:
            predictions['Systolic Blood Pressure'] = [SBP,'Severely Low']
        elif 70 <= SBP < 90:
            predictions['Systolic Blood Pressure'] = [SBP,'Low']
        elif 90 <= SBP <= 120:
            predictions['Systolic Blood Pressure'] = [SBP,'Normal']
        elif 120 < SBP <= 200:
            predictions['Systolic Blood Pressure'] = [SBP,'High']
        else:
            predictions['Systolic Blood Pressure'] = [SBP,'Severely High']

    # Mean Arterial Pressure (MAP) Levels
    if MAP is not None:
        if MAP < 50:
            predictions['Mean Arterial Pressure'] = [MAP,'Severely Low']
        elif 50 <= MAP < 70:
            predictions['Mean Arterial Pressure'] = [MAP,'Low']
        elif 70 <= MAP <= 100:
            predictions['Mean Arterial Pressure'] = [MAP,'Normal']
        elif 100 < MAP <= 120:
            predictions['Mean Arterial Pressure'] = [MAP,'High']
        else:
            predictions['Mean Arterial Pressure'] = [MAP,'Severely High']


    # Oxygen Saturation (O2Sat) Levels
    if O2Sat is not None:
        if O2Sat < 70:
            predictions['Oxygen Saturation'] = [O2Sat,'Severely Low (Hypoxemia)']
        elif 70 <= O2Sat < 90:
            predictions['Oxygen Saturation'] = [O2Sat,'Low']
        elif 90 <= O2Sat <= 100:
            predictions['Oxygen Saturation'] = [O2Sat,'Normal']
        else:
            predictions['Oxygen Saturation'] = [O2Sat,'Abnormal High']

    return predictions
import os
import pytesseract # type: ignore
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber # type: ignore
import pickle
import pandas as pd
import re
from django.http import HttpResponse
from django.shortcuts import render

def extract_data_from_pdf(pdf_path):
    # Define possible variations for each field
    # keywords = {
    #     "Age": r"\b(Age)\b",
    #     "Heart Rate": r"\b(Heart Rate|HR)\b",
    #     "Respiration Rate": r"\b(Respiration Rate|RR|Resp Rate)\b",
    #     "Temperature": r"\b(Temperature|Temp|Temperature (celsius))\b",
    #     "Diastolic Blood Pressure": r"\b(Diastolic Blood Pressure|DBP)\b",
    #     "Systolic Blood Pressure": r"\b(Systolic Blood Pressure|SBP)\b",
    #     "Mean Arterial Pressure": r"\b(Mean Arterial Pressure|MAP)\b",
    #     "Oxygen Saturation": r"\b(Oxygen Saturation|O2Sat|Oxygen saturation (O2Sat))\b",
    # }

    # # Initialize a dictionary to store the extracted data
    # extracted_data = {key: None for key in keywords}

    # # Open the PDF file
    # doc = fitz.open(pdf_path)
    # with pdfplumber.open(pdf_path) as pdf:
    #     for page in pdf.pages:
    #         text = page.extract_text()

    #         # Search for each keyword and extract the associated value
    #         lines = text.split('\n')
    #         for i, line in enumerate(lines):
    #             for key, pattern in keywords.items():
    #                 if re.search(pattern, line, re.IGNORECASE):
    #                     match = re.search(r'(\d+(\.\d+)?)', line)
    #                     if match:
    #                         extracted_data[key] = match.group(0)
    #                     else:
    #                         for j in range(1, 3):
    #                             if i + j < len(lines):
    #                                 next_line_match = re.search(r'(\d+(\.\d+)?)', lines[i + j])
    #                                 if next_line_match:
    #                                     extracted_data[key] = next_line_match.group(0)
    #                                     break

    # # Convert the dictionary to a list in the desired order
    # data_list = [
    #     extracted_data["Heart Rate"],
    #     extracted_data["Respiration Rate"],
    #     extracted_data["Temperature"],
    #     extracted_data["Diastolic Blood Pressure"],
    #     extracted_data["Systolic Blood Pressure"],
    #     extracted_data["Mean Arterial Pressure"],
    #     extracted_data["Oxygen Saturation"],
    #     extracted_data["Age"],
    # ]

    # return data_list
    keywords = {
        # "Name": r"\b(Name)\b",
        "Age": r"\b(Age)\b",
        # "Gender": r"\b(Gender)\b",
        "Heart Rate": r"\b(Heart Rate|HR)\b",
        "Respiration Rate": r"\b(Respiration Rate|RR|Resp Rate)\b",
        "Temperature": r"\b(Temperature|Temp|Temperature (celsius))\b",
        "Diastolic Blood Pressure": r"\b(Diastolic Blood Pressure|DBP)\b",
        "Systolic Blood Pressure": r"\b(Systolic Blood Pressure|SBP)\b",
        "Mean Arterial Pressure": r"\b(Mean Arterial Pressure|MAP)\b",
        "Oxygen Saturation": r"\b(Oxygen saturation|Oxygen Saturation (O2Sat)|O2Sat|Oxygen Saturation|Oxygen saturation)\b",
    }

    # Initialize a dictionary to store the extracted data
    extracted_data = {key: None for key in keywords}
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    # data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
    # for page in doc.pages:
    #     text = page.extract_text()
            name = re.search(r'Name:\s*(.*)', text).group(1).strip() if re.search(r'Name:\s*(.*)', text) else None
            gender = re.search(r'Gender:\s*(\w+)', text).group(1).strip() if re.search(r'Gender:\s*(\w+)', text) else None
        print(name,gender)
    # Loop through each page (assuming single-page PDFs for simplicity)
    for page in doc:
        text = page.get_text("text")
        lines = text.split('\n')

        # Search for each keyword and extract the associated value
        for i, line in enumerate(lines):
            for key, pattern in keywords.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Look for the value in the current line or the next few lines
                    match = re.search(r'(\d+(\.\d+)?)', line)  # Extracting a number (integer or float)
                    if match:
                        extracted_data[key] = match.group(0)
                    else:
                        # Check the next line(s) if the value might be separated
                        for j in range(1, 3):
                            if i + j < len(lines):
                                next_line_match = re.search(r'(\d+(\.\d+)?)', lines[i + j])
                                if next_line_match:
                                    extracted_data[key] = next_line_match.group(0)
                                    break
    # ['HR', 'Resp','Temp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age']
    # Convert the dictionary to a list in the desired order
    data_list = [
        name,
        # 'hema',
        # extracted_data["Name"],
        extracted_data["Heart Rate"],
        extracted_data["Respiration Rate"],
        extracted_data["Temperature"],
        extracted_data["Diastolic Blood Pressure"],
        extracted_data["Systolic Blood Pressure"],
        extracted_data["Mean Arterial Pressure"],
        extracted_data["Oxygen Saturation"],
        extracted_data["Age"],
        # extracted_data["Gender"]
        gender
        # 'Female'

        
    ]
    
    return data_list


# def extract_data_from_image(image_path):
#     # Define possible variations for each field
#     keywords = {
#         "Age": r"\b(Age)\b",
#         "Heart Rate": r"\b(Heart Rate|HR)\b",
#         "Respiration Rate": r"\b(Respiration Rate|RR|Resp Rate)\b",
#         "Temperature": r"\b(Temperature|Temp|Temperature (celsius))\b",
#         "Diastolic Blood Pressure": r"\b(Diastolic Blood Pressure|DBP)\b",
#         "Systolic Blood Pressure": r"\b(Systolic Blood Pressure|SBP)\b",
#         "Mean Arterial Pressure": r"\b(Mean Arterial Pressure|MAP)\b",
#         "Oxygen Saturation": r"\b(Oxygen Saturation|O2Sat|Oxygen saturation (O2Sat))\b",
#     }

#     # Initialize a dictionary to store the extracted data
#     extracted_data = {key: None for key in keywords}

#     # Open the image file
#     img = Image.open(image_path)
#     text = pytesseract.image_to_string(img)

#     # Search for each keyword and extract the associated value
#     lines = text.split('\n')
#     for i, line in enumerate(lines):
#         for key, pattern in keywords.items():
#             if re.search(pattern, line, re.IGNORECASE):
#                 match = re.search(r'(\d+(\.\d+)?)', line)
#                 if match:
#                     extracted_data[key] = match.group(0)
#                 else:
#                     for j in range(1, 3):
#                         if i + j < len(lines):
#                             next_line_match = re.search(r'(\d+(\.\d+)?)', lines[i + j])
#                             if next_line_match:
#                                 extracted_data[key] = next_line_match.group(0)
#                                 break

#     # Convert the dictionary to a list in the desired order
#     data_list = [
#         extracted_data["Heart Rate"],
#         extracted_data["Respiration Rate"],
#         extracted_data["Temperature"],
#         extracted_data["Diastolic Blood Pressure"],
#         extracted_data["Systolic Blood Pressure"],
#         extracted_data["Mean Arterial Pressure"],
#         extracted_data["Oxygen Saturation"],
#         extracted_data["Age"],
#     ]

#     return data_list
import easyocr # type: ignore
import re
from PIL import Image

def extract_data_from_image(image_path):
    # Fields and their expected keywords
    # keywords = {
    #     "Name": "Name",
    #     "Gender": "Gender",
    #     "Age": "Age",
    #     "Heart Rate": "Heart Rate",
    #     "Respiration Rate": "Respiration Rate",
    #     "Temperature": "Temperature",
    #     "Diastolic Blood Pressure": "Diastolic Blood Pressure",
    #     "Systolic Blood Pressure": "Systolic Blood Pressure",
    #     "Mean Arterial Pressure": "Mean Arterial Pressure",
    #     "Oxygen Saturation": "(Oxygen Saturation (O2Sat)|O2Sat|Oxygen saturation)",
    # }
    keywords = {
        "Name": r"\b(Name)\b",
        "Age": r"\b(Age)\b",
        "Gender": r"\b(Gender)\b",
        "Heart Rate": r"\b(Heart Rate|HR)\b",
        "Respiration Rate": r"\b(Respiratory|Respiration Rate|RR|RESPIRATORY RATE)\b",
        "Temperature": r"\b(Temperature|Temp|Temperature (celsius))\b",
        "Diastolic Blood Pressure": r"\b(Diastolic Blood Pressure|DBP)\b",
        "Systolic Blood Pressure": r"\b(Systolic Blood Pressure|SBP)\b",
        "Mean Arterial Pressure": r"\b(Mean Arterial Pressure|MAP)\b",
        "Oxygen Saturation": r"\b(Oxygen saturation (O2Sat)|Oxygen Saturation (O2Sat)|O2Sat|Oxygen Saturation|Oxygen saturation)\b",
    }
    # keywords = {
    #     "Name": r"\bName\b",
    #     "Gender": r"\bGender\b",
    #     "Age": r"\bAge\b",
    #     "Heart Rate": r"Heart\b",
    #     "Respiration Rate": r"Respiration\b",
    #     "Temperature": r"Temperature\b",
    #     "Diastolic Blood Pressure": r"Diastolic Blood \b",
    #     "Systolic Blood Pressure": r"Systolic Blood \b",
    #     "Mean Arterial Pressure": r"Mean Arterial \b",
    #     "Oxygen Saturation": r"Oxygen \b",
    # }

    # Initialize a dictionary to store the extracted data
    extracted_data = {key: None for key in keywords}

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform text detection and recognition
    results = reader.readtext(image_path)

    for i, (bbox, text, prob) in enumerate(results):
        text = text.strip()

        for key, keyword in keywords.items():
            if re.search(keyword, text, re.IGNORECASE):
                # Extract the next text as the value for Name, Gender, Age, etc.
                if key in ["Name", "Gender", "Age"]:
                    if i + 1 < len(results):
                        extracted_data[key] = results[i + 1][1].strip()
                else:
                    # Extract the numeric value in the same or next line for vital signs
                    match = re.search(r'(\d+(\.\d+)?)', text)
                    if match:
                        extracted_data[key] = match.group(0)
                    else:
                        # Check next line if current line doesn't have the value
                        if i + 1 < len(results):
                            next_text = results[i + 1][1].strip()
                            match_next = re.search(r'(\d+(\.\d+)?)', next_text)
                            if match_next:
                                extracted_data[key] = match_next.group(0)

    # Convert the dictionary to a list in the desired order
    data_list = [
        extracted_data["Name"],
        extracted_data["Heart Rate"],
        extracted_data["Respiration Rate"],
        extracted_data["Temperature"],
        extracted_data["Diastolic Blood Pressure"],
        extracted_data["Systolic Blood Pressure"],
        extracted_data["Mean Arterial Pressure"],
        extracted_data["Oxygen Saturation"],
        extracted_data["Age"],
        extracted_data["Gender"],
    ]
    
    print(data_list)
    return data_list

# def extract_data_from_image(image_path):
#     # Define possible variations for each field
#     # keywords = {
#     #     "Name": r"\b(Name)\b",
#     #     "Gender": r"\b(Gender)\b",
#     #     "Age": r"\b(Age)\b",
#     #     "Heart Rate": r"\b(Heart Rate|HR)\b",
#     #     "Respiration Rate": r"\b(Respiration Rate|RR|Resp Rate)\b",
#     #     "Temperature": r"\b(Temperature|Temp|Temperature \(celsius\))\b",
#     #     "Diastolic Blood Pressure": r"\b(Diastolic Blood Pressure|DBP)\b",
#     #     "Systolic Blood Pressure": r"\b(Systolic Blood Pressure|SBP)\b",
#     #     "Mean Arterial Pressure": r"\b(Mean Arterial Pressure|MAP)\b",
#     #     "Oxygen Saturation": r"\b(Oxygen Saturation|O2Sat|Oxygen saturation \(O2Sat\))\b",
#     # }
    # keywords = {
    #     "Name": r"\bName\b",
    #     "Gender": r"\bGender\b",
    #     "Age": r"\bAge\b",
    #     "Heart Rate": r"Heart\b",
    #     "Respiration Rate": r"Respiration\b",
    #     "Temperature": r"Temperature\b",
    #     "Diastolic Blood Pressure": r"Diastolic Blood \b",
    #     "Systolic Blood Pressure": r"Systolic Blood \b",
    #     "Mean Arterial Pressure": r"Mean Arterial \b",
    #     "Oxygen Saturation": r"Oxygen \b",
    # }

#     # Initialize a dictionary to store the extracted data
#     extracted_data = {key: None for key in keywords}

#     # Load the image
#     img = Image.open(image_path)

#     # Initialize EasyOCR reader
#     reader = easyocr.Reader(['en'])

#     # Perform text detection and recognition
#     results = reader.readtext(image_path)

#     # Concatenate all detected text into a single string
#     text = " ".join([res[1] for res in results])

#     # Search for each keyword and extract the associated value
#     lines = text.split(' ')
#     for i, line in enumerate(lines):
#         for key, pattern in keywords.items():
#             if re.search(pattern, line, re.IGNORECASE):
#                 if key in ["Name", "Gender"]:
#                     # Name and Gender may not be followed by a numeric value
#                     if i + 1 < len(lines):
#                         extracted_data[key] = lines[i + 1]
#                 else:
#                     match = re.search(r'(\d+(\.\d+)?)', line)
#                     if match:
#                         extracted_data[key] = match.group(0)
#                     else:
#                         for j in range(1, 3):
#                             if i + j < len(lines):
#                                 next_line_match = re.search(r'(\d+(\.\d+)?)', lines[i + j])
#                                 if next_line_match:
#                                     extracted_data[key] = next_line_match.group(0)
#                                     break

#     # Convert the dictionary to a list in the desired order
#     data_list = [
#         extracted_data["Name"],
#         extracted_data["Heart Rate"],
#         extracted_data["Respiration Rate"],
#         extracted_data["Temperature"],
#         extracted_data["Diastolic Blood Pressure"],
#         extracted_data["Systolic Blood Pressure"],
#         extracted_data["Mean Arterial Pressure"],
#         extracted_data["Oxygen Saturation"],
#         extracted_data["Age"],
#         extracted_data["Gender"],
#     ]
    
#     print(data_list)
#     return data_list

# Example usage:
# image_path = '/mnt/data/Screenshot 2024-08-15 225549.png'
# extract_data_from_image(image_path)


# Example usage:
# image_path = 'path_to_your_image.png'
# extracted_data = extract_data_from_image(image_path)
# print(extracted_data)


# Example usage
# image_path = '/mnt/data/Screenshot 2024-08-15 225549.png'
# data = extract_data_from_image(image_path)



def pdf_predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            return HttpResponse("No file uploaded.")

        # Define a new directory for storing temporary files
        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Save the file temporarily
        temp_file_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(temp_file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Determine the file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension in ['.pdf']:
            # Extract data from the PDF
            extracted_data = extract_data_from_pdf(temp_file_path)
            name = extracted_data[0]
            gender = extracted_data[-1]
            data = extracted_data[1:-1]
            print(data)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            # Extract data from the image
            extracted_data = extract_data_from_image(temp_file_path)
            name = extracted_data[0]
            gender = extracted_data[-1]
            data = extracted_data[1:-1]
            # data = extracted_data
        else:
            os.remove(temp_file_path)
            return HttpResponse("Unsupported file type.")

        # Delete the temporary file
        os.remove(temp_file_path)

        # Separate the extracted data
        # data = extracted_data
        print(data)

        # Predict using the data extracted from the file
        result = predict_levels(
            HR=float(data[0]),
            resp=float(data[1]),
            tem=float(data[2]),
            SBP=float(data[4]),
            MAP=float(data[5]),
            DBP=float(data[3]),
            O2Sat=float(data[6])
        )

        # Use the same prediction function for risk assessment
        with open('app/xgboost_optimized_model_bayesian.pkl', 'rb') as model_file:
            optimized_model = pickle.load(model_file)

        with open('app/scaler_optimized_bayesian.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        def predict_sepsis(input_features, threshold=0.1):
            feature_names = ['HR', 'Resp', 'Temp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age']

            input_df = pd.DataFrame([input_features], columns=feature_names)
            input_scaled = scaler.transform(input_df)
            prediction_proba = optimized_model.predict_proba(input_scaled)[0][1]
            if prediction_proba >= threshold:
                return 1, int(prediction_proba * 100)
            else:
                return 0, int(prediction_proba * 100)

        prediction, prediction_proba = predict_sepsis(data)

        if prediction == 1:
            output_message = ("The model predicts a high risk of sepsis for this patient.",
                              "Please ensure immediate evaluation and monitoring.",
                              "Consult with a healthcare professional to confirm and take appropriate action.")
            predict = "POSITIVE"
        else:
            output_message = ("The model predicts a low risk of sepsis for this patient.",
                              "Continue with routine monitoring and follow-up as advised.",
                              "Please note that this prediction does not replace professional medical assessment.")
            predict = "NEGATIVE"

        # Call the generate_html function with the correct parameters
        generate_html(predictions=result, output=output_message, data=data, name=name, gen=gender, prob=prediction_proba, pred=predict)

        return render(request, 'prediction_results.html')

    return HttpResponse("Please upload a file.")


# import os
# import pdfplumber
# import re
# import pickle
# import pandas as pd
# from django.http import HttpResponse
# from django.shortcuts import render

# import fitz  # PyMuPDF
# import fitz  # PyMuPDF
# import re

# def extract_data_from_pdf(pdf_path):
#     # Define possible variations for each field
#     keywords = {
#         # "Name": r"\b(Name)\b",
#         "Age": r"\b(Age)\b",
#         # "Gender": r"\b(Gender)\b",
#         "Heart Rate": r"\b(Heart Rate|HR)\b",
#         "Respiration Rate": r"\b(Respiration Rate|RR|Resp Rate)\b",
#         "Temperature": r"\b(Temperature|Temp|Temperature (celsius))\b",
#         "Diastolic Blood Pressure": r"\b(Diastolic Blood Pressure|DBP)\b",
#         "Systolic Blood Pressure": r"\b(Systolic Blood Pressure|SBP)\b",
#         "Mean Arterial Pressure": r"\b(Mean Arterial Pressure|MAP)\b",
#         "Oxygen Saturation": r"\b(Oxygen Saturation|O2Sat|Oxygen saturation (O2Sat))\b",
#     }

#     # Initialize a dictionary to store the extracted data
#     extracted_data = {key: None for key in keywords}
    
#     # Open the PDF file
#     doc = fitz.open(pdf_path)
#     # data = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#     # for page in doc.pages:
#     #     text = page.extract_text()
#             name = re.search(r'Name:\s*(.*)', text).group(1).strip() if re.search(r'Name:\s*(.*)', text) else None
#             gender = re.search(r'Gender:\s*(\w+)', text).group(1).strip() if re.search(r'Gender:\s*(\w+)', text) else None
#         print(name,gender)
#     # Loop through each page (assuming single-page PDFs for simplicity)
#     for page in doc:
#         text = page.get_text("text")
#         lines = text.split('\n')

#         # Search for each keyword and extract the associated value
#         for i, line in enumerate(lines):
#             for key, pattern in keywords.items():
#                 if re.search(pattern, line, re.IGNORECASE):
#                     # Look for the value in the current line or the next few lines
#                     match = re.search(r'(\d+(\.\d+)?)', line)  # Extracting a number (integer or float)
#                     if match:
#                         extracted_data[key] = match.group(0)
#                     else:
#                         # Check the next line(s) if the value might be separated
#                         for j in range(1, 3):
#                             if i + j < len(lines):
#                                 next_line_match = re.search(r'(\d+(\.\d+)?)', lines[i + j])
#                                 if next_line_match:
#                                     extracted_data[key] = next_line_match.group(0)
#                                     break
#     # ['HR', 'Resp','Temp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age']
#     # Convert the dictionary to a list in the desired order
#     data_list = [
#         name,
#         # 'hema',
#         # extracted_data["Name"],
#         extracted_data["Heart Rate"],
#         extracted_data["Respiration Rate"],
#         extracted_data["Temperature"],
#         extracted_data["Diastolic Blood Pressure"],
#         extracted_data["Systolic Blood Pressure"],
#         extracted_data["Mean Arterial Pressure"],
#         extracted_data["Oxygen Saturation"],
#         extracted_data["Age"],
#         # extracted_data["Gender"]
#         gender
#         # 'Female'

        
#     ]
    
#     return data_list


# # def extract_data_from_pdf(pdf_path):
# #     data = []
# #     with pdfplumber.open(pdf_path) as pdf:
# #         for page in pdf.pages:
# #             text = page.extract_text()

# #             # Extract fields and append to the list
# #             name = re.search(r'Name:\s*(.*)', text).group(1).strip() if re.search(r'Name:\s*(.*)', text) else None
# #             age = re.search(r'Age:\s*(\d+)', text).group(1).strip() if re.search(r'Age:\s*(\d+)', text) else None
# #             gender = re.search(r'Gender:\s*(\w+)', text).group(1).strip() if re.search(r'Gender:\s*(\w+)', text) else None
# #             heart_rate = re.search(r'Heart Rate\s*(\d+)', text).group(1).strip() if re.search(r'Heart Rate\s*(\d+)', text) else None
# #             respiration_rate = re.search(r'Respiration Rate\s*(\d+)', text).group(1).strip() if re.search(r'Respiration Rate:\s*(\d+)', text) else None
# #             temperature = re.search(r'Tempature (celsius)\s*(\d+\.\d+|\d+)', text).group(1).strip() if re.search(r'Temperature:\s*(\d+\.\d+|\d+)', text) else None
# #             diastolic_bp = re.search(r'Diastolic Blood Pressure\s*(\d+)', text).group(1).strip() if re.search(r'Diastolic Blood Pressure:\s*(\d+)', text) else None
# #             systolic_bp = re.search(r'Systolic Blood Pressure\s*(\d+)', text).group(1).strip() if re.search(r'Systolic Blood Pressure:\s*(\d+)', text) else None
# #             map_value = re.search(r'Mean Arterial Pressure\s*(\d+)', text).group(1).strip() if re.search(r'Mean Arterial Pressure:\s*(\d+)', text) else None
# #             oxygen_saturation = re.search(r' Oxygen saturation (O2Sat)\s*(\d+)', text).group(1).strip() if re.search(r'Oxygen Saturation:\s*(\d+)', text) else None

# #             data.extend([heart_rate, respiration_rate, temperature, diastolic_bp, systolic_bp, map_value, oxygen_saturation, age])
# #             data.append(gender)
# #             data.insert(0, name)
# #     print(data)
# #     return data

# def pdf_predict(request):
#     if request.method == 'POST':
#         pdf_file = request.FILES['pdf_file']

#         # Define a new directory for storing temporary files
#         tmp_dir = os.path.join(os.getcwd(), 'tmp')
#         if not os.path.exists(tmp_dir):
#             os.makedirs(tmp_dir)

#         # Save the file temporarily
#         temp_pdf_path = os.path.join(tmp_dir, pdf_file.name)
#         with open(temp_pdf_path, 'wb+') as destination:
#             for chunk in pdf_file.chunks():
#                 destination.write(chunk)

#         # Extract data from the PDF
#         extracted_data = extract_data_from_pdf(temp_pdf_path)
#         print(extracted_data)

#         # Delete the temporary file
#         os.remove(temp_pdf_path)

#         # Separate the extracted data
#         name = extracted_data[0]
#         gender = extracted_data[-1]
#         data = extracted_data[1:-1]
#         print(data)
#         # Gender conversion (assuming Male = 0, Female = 1)
#         val9 = 0 if gender == "Male" else 1
# # [heart_rate, respiration_rate, temperature, diastolic_bp, systolic_bp, map_value, oxygen_saturation, age]
#         # Predict using the data extracted from the PDF
#         result = predict_levels(
#             HR=float(data[0]),
#             resp=float(data[1]),
#             tem=float(data[2]),
#             SBP=float(data[4]),
#             MAP=float(data[5]),
#             DBP=float(data[3]),
#             O2Sat=float(data[6])
#         )

#         # Use the same prediction function for risk assessment
#         with open(r'app\xgboost_optimized_model_bayesian.pkl', 'rb') as model_file:
#             optimized_model = pickle.load(model_file)

#         with open('app\scaler_optimized_bayesian.pkl', 'rb') as scaler_file:
#             scaler = pickle.load(scaler_file)

#         def predict_sepsis(input_features, threshold=0.1):
#             feature_names = ['HR', 'Resp', 'Temp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age']

#             input_df = pd.DataFrame([input_features], columns=feature_names)
#             input_scaled = scaler.transform(input_df)
#             prediction_proba = optimized_model.predict_proba(input_scaled)[0][1]
#             if prediction_proba >= threshold:
#                 return 1, int(prediction_proba * 100)
#             else:
#                 return 0, int(prediction_proba * 100)

#         prediction, prediction_proba = predict_sepsis(data)

#         if prediction == 1:
#             output_message = ("The model predicts a high risk of sepsis for this patient.",
#                               "Please ensure immediate evaluation and monitoring.",
#                               "Consult with a healthcare professional to confirm and take appropriate action.")
#             predict = "POSITIVE"
#         else:
#             output_message = ("The model predicts a low risk of sepsis for this patient.",
#                               "Continue with routine monitoring and follow-up as advised.",
#                               "Please note that this prediction does not replace professional medical assessment.")
#             predict = "NEGATIVE"

#         # Call the generate_html function with the correct parameters
#         generate_html(predictions=result, output=output_message, data=data, name=name, gen=gender, prob=prediction_proba, pred=predict)

#         return render(request, 'prediction_results.html')

#     return HttpResponse("Please upload a PDF file.")

# user_input_values = data
def predict(request):
    if request.method == 'POST':
    #     # Print all POST data for debugging
        print("POST data:", request.POST)
        name=request.POST.get('Name')
        # Retrieve POST data
        val1 = request.POST.get('HR')
        # val7 = request.POST.get('PO')
        val2 = request.POST.get('tem')
        val3 = request.POST.get('SBP')
        val4 = request.POST.get('MAP')
        val5 = request.POST.get('DBP')
        val6 = request.POST.get('Res')
        # val7 = request.POST.get('Etco2')
        # val7 = request.POST.get('PO')
        val8 = request.POST.get('Age')
        gender = request.POST.get('Gender')
        # val19=request.POST.get("Male")
        # val29=request.POST.get("Female")
        # print(val19,val29)
        if gender=="Male":
            # val19=1
            val9=0
        elif  gender=="Female":
            # val19=0
            val9=1
        # print(val19,val29)    
        # val19=1
        # val29=0
        # val9 = request.POST.get('Gender')
        val10=request.POST.get("O2Sat")
        data = [val1, val6, val2, val5, val3, val4, val10, val8]
        # data={ 'HR':78, 'O2Sat':0, 'Temp':0, 'SBP':42.5, 'MAP':0, 'DBP':100, 'Resp':98, 'Age':25,'Gender':0}
        # data=[108, 17,37.17, 51, 86, 63, 100, 42.52]
        print(data)

        result = predict_levels(
        HR=float(val1),
        resp=float(val6),
        tem=float(val2),
        SBP=float(val3),
        MAP=float(val4),
        DBP=float(val5),
        O2Sat=float(val10)
        )
        
        # from sklearn.preprocessing import StandardScaler, OneHotEncoder
        # from sklearn.decomposition import PCA
        # import numpy as np

#         def preprocess_and_predict(raw_user_data):
#     # Load pre-trained models and transformers
#             with open('app/xgb_model.pkl', 'rb') as f:
#                 loaded_model = pickle.load(f)
    
#             with open('app/scaler.pkl', 'rb') as f:
#                 loaded_scaler = pickle.load(f)

#             with open('app/pca.pkl', 'rb') as f:
#                 loaded_pca = pickle.load(f)
    
#     # Step 1: Apply the same preprocessing steps as during training
#     # Example: If one-hot encoding was applied, do the same here
#     # Assuming raw_user_data is appropriately structured
    
#     # Step 2: Scale the data
#             user_data_scaled = loaded_scaler.transform(raw_user_data)

#             # Step 3: Apply PCA to get to the 3016 features
#             user_data_pca = loaded_pca.transform(user_data_scaled)
    
#     # Step 4: Make predictions
#             prediction = loaded_model['model'].predict(user_data_pca)
#             prediction_proba = loaded_model['model'].predict_proba(user_data_pca)[:, 1]
    
#             return prediction_proba,prediction

# # Example raw input data
#         example_raw_user_data = [[80, 20, 37.5, 85, 120, 90, 98, 45, 1]]

# # Run prediction
#         prediction_proba = preprocess_and_predict(example_raw_user_data)
#         print("Prediction Probability:", prediction_proba)

        # import pickle
        # import numpy as np
        # import pandas as pd
        with open(r'app\xgboost_optimized_model_bayesian.pkl', 'rb') as model_file:
            optimized_model = pickle.load(model_file)

        with open('app\scaler_optimized_bayesian.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        def predict_sepsis(input_features, threshold=0.1):
            feature_names = ['HR', 'Resp','Temp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age']


            input_df = pd.DataFrame([input_features], columns=feature_names)
            input_scaled = scaler.transform(input_df)
            prediction_proba = optimized_model.predict_proba(input_scaled)[0][1]
            print(prediction_proba)
            if prediction_proba >= threshold:
                print("sepsis")
                return 1,(int(prediction_proba*100))
            else:
                print("not")
                return 0,(int(prediction_proba*100))
        # positive_sepsis_input_no_gender = [98.9, 17, 38, 68, 121, 86, 96, 45]
        prediction,prediction_proba = predict_sepsis(data)
        print(prediction)


        # xgboost_model_filename = r'app\best_xgb_model_lasttt.pkl'
        # with open(xgboost_model_filename, 'rb') as file:
        #     loaded_model = pickle.load(file)
        # model=r'app\xgb_model.pkl'
        # with open(model, 'rb') as model_file:
        #     xgb_model = pickle.load(model_file)
        # import pickle

        # model_file_path = r'app\xgb_model.pkl'
        # with open(model_file_path, 'rb') as model_file:
        #     xgb_model = pickle.load(model_file)
        #     model_content = xgb_model['model']
        # print(type(model_content))
        # # for key, value in model_content.items():
        # #     print(f"Key: {key}, Type of value: {type(value)}")

        # with open('app\scaler.pkl', 'rb') as scaler_file:
        #     scaler = pickle.load(scaler_file)
        # # def predict_sepsis(input_features):
        # input_array = np.array(data).reshape(1, -1)
        # input_scaled = scaler.transform(data)
        # prediction = model_content.predict(data)
        # predicted_probability=model_content.predict_proba(input_scaled)[:, 1]
            # if prediction[0] == 1:
            #     return "The model predicts that the user has sepsis."
            # else:
            #     return "The model predicts that the user does not have sepsis."

# user_input = [105, 45.5, 10, 60, 80, 94, 98, 62, 0]  
# result = predict_sepsis(user_input)

        # for i in result:
        #     return render(request, 'app/prediction_result.html', {'result': i})
        # scaler_filename = 'app\scaler_final.pkl'
        # with open(scaler_filename, 'rb') as file:
        #     scaler = pickle.load(file)

# Load the saved XGBoost model
        # xgboost_model_filename = r'app\best_xgb_model_lasttt.pkl'
        # with open(xgboost_model_filename, 'rb') as file:
        #     loaded_model = pickle.load(file)

# Define the hardcoded user input values for prediction (ensure it matches encoded gender columns)
        # user_input_values = [75, 36.8, 16, 80, 120, 93, 98, 35, 1, 0]  # Example values (last two for Gender_0.6, Gender_1.0)
        # user_input_values = data
        # columns = ['HR', 'Temp', 'Resp', 'DBP', 'SBP', 'MAP', 'O2Sat', 'Age', 'Gender_0.6', 'Gender_1.0']



# Create a DataFrame from the user input
        # user_input_df = pd.DataFrame([user_input_values], columns=columns)

# Ensure the same feature scaling and encoding is applied as during training
        # user_input_scaled = scaler.transform(user_input_df.drop(columns=[ 'Gender_0.6', 'Gender_1.0']))
        # user_input_scaled = pd.DataFrame(user_input_scaled, columns=user_input_df.columns.drop(['Gender_0.6', 'Gender_1.0']))

# Add the Gender columns back to the scaled data
        # user_input_scaled = pd.concat([user_input_scaled, user_input_df[['Gender_0.6', 'Gender_1.0']]], axis=1)

# Make predictions using the loaded model
        # predicted_output = loaded_model.predict(user_input_scaled)
        # predicted_probability = loaded_model.predict_proba(user_input_scaled)[:, 1]

# Display the prediction results
        # print("Predicted Class (0=No Sepsis, 1=Sepsis):", prediction[0])
        # print("Predicted Probability of Sepsis:", predicted_probability[0])
        # return HttpResponse(f'Prediction:{result}')
        if prediction == 1:
            output_message = ("The model predicts a high risk of sepsis for this patient.",
                              "Please ensure immediate evaluation and monitoring.",
                              "Consult with a healthcare professional to confirm and take appropriate action.")
            predict="POSITIVE"

        else:
            output_message = ("The model predicts a low risk of sepsis for this patient.",
                              "Continue with routine monitoring and follow-up as advised.",
                              "Please note that this prediction does not replace professional medical assessment.")
            predict="NEGATIVE"
        # # for i in output_message:
        generate_html(result,output_message,data,name,gender,prediction_proba,predict)


        # Render the generated HTML template
        return render(request, 'prediction_results.html')

    return render(request, 'prediction.html')
        # html_content = generate_html(result,output_message)

# Write the HTML content to a file
        # with open('app/prediction_result.html', 'w') as file:
        #     file.write(html_content)

        # # "app/prediction_results.html"
        # # return HttpResponse(f'Prediction:{output_message}')
        # return render(request, 'prediction_result.html')
        # return render(request, 'app\prediction_result.html', {'result': output_message})

    # If not a POST request, render an empty form or appropriate response
    # return render(request, 'prediction.html')
    # return render(request, 'index.html', {'result': result})
# def prediction(request):
#     if request.method == 'POST':
#         return HttpResponse(f'Prediction:')
# main()
# predict(request=None)
#     file_path = r'C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\BHavs_ensemble_model.pkl'
        # try:
        # # Get the current script's directory
        #     # script_dir = os.path.dirname(os.path.abspath(_file_))
        
        # # Construct the full path to the model file
        #     # model_path = os.path.join(script_dir, 'app\BHavs_ensemble_model.pkl')
        #     model_path = r'app/BHavs_ensemble_model.pkl'
            
        # # Load the model
        #     model = load_model(model_path)

            # print(f"Model type: {type(model)}")
            # if isinstance(model, VotingClassifier):
            #     print("This is an ensemble model with the following estimators:")
            #     for i, (name, estimator) in enumerate(model.named_estimators_.items()):
            #         print(f"  Estimator {i+1}: {type(estimator)}")
            # print(f"Number of features expected by the model: {model.n_features_in_}")

        # Get user input
    #         user_input = {'Hour':2, 'HR':78, 'O2Sat':0, 'Temp':0, 'SBP':42.5, 'MAP':0, 'DBP':100, 'Resp':98, 'Age':25,'Gender':0}

    #     # Make prediction
    #         result, probabilities = predict_sepsis(user_input, model)
    #         print(f"Sepsis prediction: {result}")
    #         print(f"Prediction probabilities: Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}")
    #         return HttpResponse(f'Prediction: {result}')
    #     except FileNotFoundError:
    #         print("Model file not found. Please ensure 'BHavs_ensemble_model.pkl' is in the same directory as this script.")

    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         print("Please make sure the model file is correct and all required libraries are installed.")
    #         return HttpResponse(status=405) 
    # else:
    #     return HttpResponse(status=405) 
# if _name_ == "_main_":
#     main()
# # numerical_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
# categorical_cols = ['Gender', 'Age']


# def check_pickle_contents(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             data = pickle.load(file)
#             print("Pickle file contents:", data)
#             if 'model' in data and 'preprocessor' in data:
#                 print("Model and preprocessor keys are present.")
#             else:
#                 print(f"Keys 'model' and 'preprocessor' not found. Available keys: {list(data.keys())}")
#     except Exception as e:
#         print(f"Error while loading the pickle file: {e}")

# # Run the check

# def load_model_and_preprocessor(file_path):
#     try:
#         with open(file_path, 'rb') as model_file:
#             try:
#                 data = pickle.load(model_file)
#                 if 'model' in data and 'preprocessor' in data:
#                     model, preprocessor = data['model'], data['preprocessor']
#                     print("Model and preprocessor loaded successfully")
#                     return model, preprocessor
#                 else:
#                     print(f"Keys 'model' and 'preprocessor' not found in the loaded data. Available keys: {list(data.keys())}")
#                     return None, None
#             except pickle.UnpicklingError as e:
#                 print(f"Unpickling error: {e}")
#                 return None, None
#             except EOFError as e:
#                 print(f"EOF error: {e}")
#                 return None, None
#             except Exception as e:
#                 print(f"General error while unpickling: {e}")
#                 return None, None
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#         return None, None
#     except IOError as e:
#         print(f"I/O error({e.errno}): {e.strerror}")
#         return None, None
#     except Exception as e:
#         print(f"Error opening file: {e}")
#         return None, None

# def preprocess_user_input(input_list, preprocessor):
#     feature_order = numerical_cols + categorical_cols
#     input_df = pd.DataFrame([input_list], columns=feature_order)
#     input_data_processed = preprocessor.transform(input_df)
#     return input_data_processed

# def predict_sepsis(input_list, model, preprocessor):
#     input_data_processed = preprocess_user_input(input_list, preprocessor)
#     prediction = model.predict(input_data_processed)
#     return prediction



# Path to the uploaded pickle file
# def predict(request):

#     if request.method == 'POST':
#         # Print all POST data for debugging
#         print("POST data:", request.POST)

#         # Retrieve POST data
#         val1 = request.POST.get('HR')
#         val7 = request.POST.get('PO')
#         val2 = request.POST.get('tem')
#         val3 = request.POST.get('SBP')
#         val4 = request.POST.get('MAP')
#         val5 = request.POST.get('DBP')
#         val6 = request.POST.get('Res')
#         # val7 = request.POST.get('Etco2')
#         # val7 = request.POST.get('PO')
#         val8 = request.POST.get('Age')
#         # if request.POST.get('Gender')=="Male":
#         #     val10 = 0
#         # else:
#         #     val10 = 1
#         val9 = request.POST.get('Gender')
#         val10=request.POST.get("O2Sat")
#         data = [[val1, val7, val2, val3, val4, val5, val6, val8, val9, val10]]
#         print(data)
#     else:
#         return HttpResponse(status=405) 
# #     file_path = r'C:\Users\jchem\OneDrive\Documents\cts\sepsis\app\BHavs_ensemble_model.pkl'
# #     check_pickle_contents(file_path)
# # # Load the model and preprocessor
# #     # file_path = '/mnt/data/Final_ensemble_model.pkl'
# #     model, preprocessor = load_model_and_preprocessor(file_path)

# #     if model is not None and preprocessor is not None:
# #         user_input_list = [80, 98, 36.6, 120, 85, 70, 20, 0, 45]  # Example values
# #         prediction = predict_sepsis(user_input_list, model, preprocessor)
# #         print("Sepsis Prediction:", "Positive" if prediction[0] == 1 else "Negative")
# #     else:
# #         print("Failed to load the model and preprocessor")

# #     return HttpResponse(f'Prediction: ')