import os
import json
from groq import Groq
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = "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"

client = Groq()

def chat_with_llm(user_prompt, chat_history):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *chat_history,
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        assistant_response = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response
    except Exception as e:
        return f"An error occurred: {str(e)}"
chat_history = []
@csrf_exempt
def chatbot(query):
    # while True:
        # user_prompt = input("You: ")
        # query=request.POST.get('message')
        user_prompt = query
        # if user_prompt.lower() in ['exit', 'quit']:
        #     break
        chat_history.append({"role": "user", "content": user_prompt})
        response = chat_with_llm(user_prompt, chat_history)
        print(f"LLAMA: {response}")
        return response
        # return render(request, 'app/base.html', {'result': response})
# chat("what is sepsis")
# @csrf_exempt
# def chat(request):
#     if request.method == "POST":
#         user_input = request.POST.get('message')
#         # Process the user's input (You can integrate an AI model here)
#         bot_response = chatbot(user_input)
#         return render(request, 'app/base.html', {'result': bot_response})
#     return render(request, 'chat.html')



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_input = data.get('message')
        
        # Process the user's input
        bot_response = chatbot(user_input)
        
        # Return the bot's response as JSON
        return JsonResponse({'response': bot_response})
    

# import pickle

# # Load from pickle
# with open('components.pkl', 'rb') as f:
#     loaded_components = pickle.load(f)

# # Access the components
# llm_loaded = loaded_components["llm"]
# vector_store_loaded = loaded_components["vector_store"]
# qa_chain_loaded = loaded_components["qa_chain"]

# # You can now use these components as needed
# query = "What is sepsis?"
# print(qa_chain_loaded.run(query))