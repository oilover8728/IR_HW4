from django.urls import path, include

from . import views

urlpatterns = [
    path('default', views.default, name='default'),
    path('default_no_tfidf', views.default_no_tfidf, name='default_no_tfidf'),
]