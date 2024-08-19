SEPSIS_PREDICTION
Goal of the analysis is the early prediction of sepsis using vital signs data.The prediction of sepsis is potentially life-saving. We will be predicting whether a person if affected with sepsis or not with the reasons stated using laboratory values, vital signs and patient information, in the website.
 Summary Report: Prediction of Sepsis
 Introduction: This report outlines a Sepsis Prediction application that utilizes vital signs data and an XGBoost machine learning model to predict the risk of sepsis in patients. Sepsis is a severe medical condition characterized by a systemic inflammatory response to infection, and early detection is critical for timely intervention.
 Data Source: The application uses patient data from a CSV file, which includes vital signs and demographic information.
 
Model Training:
1.	 The XGBoost machine learning model is used for prediction.
2.	The dataset is split into training and testing sets (80% training, 20% testing) to train and evaluate the model.
3.	 The model is trained with the following parameters:
•	      Libliner :
•	      Evaluation Metric: Logarithmic loss
•	      Learning Rate (Eta): 0.18
•	      Maximum Depth: 9
•	      Subsample: 0.84
•	      Column Sample by Tree: 0.98
•	      N_estimators:262
•	      Gamma: 0.48
•	      Reg_Lanbda:0.90 
•	      Reg_alpha:0.01
Input Data: Users can input patient information via a Streamlit interface. The following vital signs and demographic information are collected:
1.	Age
2.	Gender
3.	Temperature(Temp)
4.	Heart rate(HR)
5.	Mean Artial Pressure()
6.	Respiratory rate(RR)
7.	O2 Saturation level
8.	Systolic Blood pressure
9.	Diastolic Blood Pressure
 
Sepsis Risk Assessment:
 
1.	1Users input patient data and submit it.
2.	2.The model predicts the probability of sepsis based on the input data.
3.	3.The application displays the risk assessment result:
•	     If the probability is less than 0.4, it indicates a minor chance of sepsis.
•	     If the probability is greater than or equal to 0.4, it indicates a major chance of sepsis.
 
Sepsis Risk Factors:
 Chat bot:
This chatbot, built using Python and Jupyter Notebook, interacts with users by generating responses based on user input. It employs the "llama-3.1-8b-instant" language model, accessed via the Groq API, to simulate a conversational AI assistant. The chatbot integrates the model using the Groq API, which requires an API key for authentication. Groq, a company specializing in machine learning infrastructure, provides the necessary services for the chatbot to generate coherent and human-like text responses based on the conversation history.
Conclusion: 
This Sepsis Prediction application provides a user-friendly interface for predicting the risk of sepsis in patients based on their vital signs and demographic information. It can aid healthcare professionals in identifying individuals at risk of sepsis and taking appropriate measures for early intervention.
