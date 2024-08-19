import os
import numpy as np
import pickle
import sklearn
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from google.oauth2 import id_token
from google.auth.transport import requests
from django.db import IntegrityError
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
# with open('charithamodel.pkl', 'rb') as file:
#     loaded_components = pickle.load(file)
# model = loaded_components['model']
# Create your views here.
@login_required(login_url='login')
def base(request):
    return render(request,'app/base.html')
@csrf_exempt 
def sign_up(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')
        
        try:
            if pass1!=pass2 or uname=='' or email=='':
            # return HttpResponse("Your password and confrom password are not Same!!")
                return render(request,'app/sign_up.html', {'message':"Your password and confirm password are not Same!!"})
            else:
                user = User.objects.create_user(username=uname, email=email, password=pass1)
                user.save()
                # my_user=User.objects.create_user(uname,email,pass1)
                # my_user.save()
                return redirect('home')
         
            #     user = User.objects.create_user(username=username, password=password)
            # user.save()
            # Redirect to success page
        except IntegrityError:
            # Handle the error, e.g., by rendering the signup page with an error message
            return render(request, 'app/signup.html', {'message': 'Username already exists'})

    return render(request,'app/sign_up.html')
@csrf_exempt
def login(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            # login(request,user)
            return redirect('base')
        else:
            # return HttpResponse ("Username or Password is incorrect!!!")
            return render(request,'app/sign_up.html', {'message':"Check your user name and password!!"})
    return render (request,'app/login.html')
@csrf_exempt
def logout(request):
    # logout(request)
    return redirect('home')

# def home(request):
#     try:
#         return render(request, "home.html")
#     except Exception as e:
#         import traceback
#         error_message = traceback.format_exc()
#         return HttpResponse(f"Error: {error_message}")

# def logout_view(request):
#     logout(request)
#     return redirect("/")
# @csrf_exempt
# def home(request):
#     return render(request, 'app/home.html')

def contact(request):
    return render(request, 'app/contact.html')

def about(request):
    return render(request, 'app/about.html')

def services(request):
    return render(request, 'app/services.html')

def home(request):
    return render(request, 'app/home.html')#, {'section':'base'})

def reports(request):
    return render(request, 'app/reports.html')

def symptoms(request):
    return render(request, 'app/symptoms.html')
# @csrf_exempt
# def sign_up(request):
#     return render(request, 'app/sign_up.html')

# @csrf_exempt
# def login(request):
#     return render(request, 'app/login.html')

@csrf_exempt
def auth_receiver(request):
    """
    Google calls this URL after the user has signed in with their Google account.
    """
    print('Inside')
    print(request.POST)
    # token = request.POST['credential']
    token = request.POST.get('credential')
    if token:
        token = request.POST['credential']
    try:
        user_data = id_token.verify_oauth2_token(
            token, requests.Request(), os.environ['GOOGLE_OAUTH_CLIENT_ID']
        )
    except ValueError:
        return HttpResponse(status=403)

    # In a real app, I'd also save any new user here to the database.
    # You could also authenticate the user here using the details from Google (https://docs.djangoproject.com/en/4.2/topics/auth/default/#how-to-log-a-user-in)
    request.session['user_data'] = user_data

    return redirect('login')


# def logout(request):
#     # del request.session['user_data']
#     return redirect('home')

# def index():
#     return render('index.html')
# @app.route('/predict', methods=['GET', 'POST'])


@csrf_exempt
def prediction(request):
    return render(request, 'app/prediction.html')


@csrf_exempt
def predict(request):
    with open('charithamodel.pkl', 'rb') as file:
        loaded_components = pickle.load(file)
    model = loaded_components['model']
    if request.method=='POST':
        val1 = request.POST.get['prg_ctr']
        val2 = request.form['pl_glucose_conc.']
        val3 = request.form['bp']
        val4 = request.form['skin_thick']
        val5 = request.form['insulin']
        val6 = request.form['bmi']
        val7 = request.form['diabetes']
        val8 = request.form['age']
    arr = np.array([val1, val2, val3, val4,val5, val6, val7, val8])
    pred = model.predict([arr])
    prompt
    if pred==[1]:
        output="Sepsis is Present"
        prompt = "why did i got sespsis?"
    else:
        output="Sepsis is Absent"
        prompt = "What are the precautions to be taken for preventing sepsis?"
    # response = llm_model.generate_content(prompt)

    return render('prediction.html', data=output)#+" "+response.text)
    # Trigger LLM model by passing this prompt to automatically generate relavant response.

# @csrf_exempt
# def prediction(request):
#     return render(request, 'app/prediction.html')

# @csrf_exempt
# def predict(request):
#     with open('charithamodel.pkl', 'rb') as file:
#         loaded_components = pickle.load(file)
#     model = loaded_components['model']
    
#     if request.method == 'POST':
#         val1 = request.POST.get('prg_ctr')
#         val2 = request.POST.get('pl_glucose_conc.')
#         val3 = request.POST.get('bp')
#         val4 = request.POST.get('skin_thick')
#         val5 = request.POST.get('insulin')
#         val6 = request.POST.get('bmi')
#         val7 = request.POST.get('diabetes')
#         val8 = request.POST.get('age')
        
#         arr = np.array([val1, val2, val3, val4, val5, val6, val7, val8])
#         pred = model.predict([arr])
        
#         if pred == [1]:
#             output = "Sepsis is Present"
#             prompt = "Why did I get sepsis?"
#         else:
#             output = "Sepsis is Absent"
#             prompt = "What are the precautions to be taken for preventing sepsis?"

#         # Assuming you have an LLM model to generate content based on the prompt
#         # response = llm_model.generate_content(prompt)

#         return render(request, 'app/prediction_result.html', {'data': output})  # + response.text)

#     return render(request, 'app/predict.html')

# def predict(request):
#     # Path to the model file
#     model_path = os.path.join(os.path.dirname(__file__), 'model_and_key_components.pkl')
    
#     try:
#         with open(model_path, 'rb') as file:
#             loaded_components = pickle.load(file)
#     except FileNotFoundError:
#         return render(request, 'app/prediction.html', {'error': 'Model file not found.'})
#     except Exception as e:
#         return render(request, 'app/prediction.html', {'error': f'Error loading model: {e}'})
    
#     model = loaded_components.get('model')
#     if not model:
#         return render(request, 'app/prediction.html', {'error': 'Model not found in the file.'})
    
#     if request.method == 'POST':
#         try:
#             val1 = float(request.POST.get('prg_ctr', 0))
#             val2 = float(request.POST.get('pl_glucose_conc.', 0))
#             val3 = float(request.POST.get('bp', 0))
#             val4 = float(request.POST.get('skin_thick', 0))
#             val5 = float(request.POST.get('insulin', 0))
#             val6 = float(request.POST.get('bmi', 0))
#             val7 = float(request.POST.get('diabetes', 0))
#             val8 = float(request.POST.get('age', 0))

#             arr = np.array([val1, val2, val3, val4, val5, val6, val7, val8])
#             pred = model.predict([arr])
#         except Exception as e:
#             return render(request, 'app/prediction.html', {'error': f'Error during prediction: {e}'})

#         if pred == [1]:
#             output = "Sepsis is Present"
#             prompt = "Why did I get sepsis?"
#         else:
#             output = "Sepsis is Absent"
#             prompt = "What are the precautions to be taken for preventing sepsis?"

#         # Assuming you have an LLM model to generate content based on the prompt
#         # response = llm_model.generate_content(prompt)

#         return render(request, 'app/prediction_result.html', {'data': output})  # + response.text)

#     return render(request, 'app/prediction.html')