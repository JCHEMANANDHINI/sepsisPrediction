# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home),
#     path('logout', views.logout_view, name='logout'),
#     # path('auth-recseiver', views.auth_receiver, name='auth_receiver'),
# ]
# from django.urls import path
# from .views import home,logout_view

# urlpatterns = [
#     path('', home, name='home'),
#     path('logout', logout_view, name='logout'),
# ]

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.sign_in, name='sign_in'),
#     path('sign-out', views.sign_out, name='sign_out'),
#     path('auth-receiver', views.auth_receiver, name='auth_receiver'),
# ]

from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
# from . import Sepsis
from . import predict
from . import chat
urlpatterns = [
    path('', views.home, name='home'),
    # path('login/', auth_views.LoginView.as_view(), name='login'),
    # path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('signup/', views.sign_up, name='sign_up'),
    path('base/', views.base, name='base'),
    # path('sign-out', views.sign_out, name='sign_out'),
    path('auth-receiver', views.auth_receiver, name='auth_receiver'),
    path('contact/', views.contact, name='contact'),
    path('services/', views.services, name='services'),
    path('about/', views.about, name='about'),
    path('prediction/', views.prediction, name='prediction'),
    path('reports/', views.reports, name='reports'),
    path('symptoms/', views.symptoms, name='symptoms'),

    # path('predict/', predict.predict, name='prediction'),
    path('predict/', predict.predict, name='predict'),
    # path('predict/', predict.predict, name='predict'),
    path('chat/',chat.chat,name='chat'),
    path('pdf_predict/', predict.pdf_predict, name='pdf_predict'),
]

