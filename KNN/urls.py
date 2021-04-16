from django.urls import path
from . import views


urlpatterns = [
    path('knn/',views.knn,name='knn'),
    path('ada/',views.ada,name='ada'),
    path('rforest/',views.rforest,name='rforest'),
    path('conef/',views.conef,name='conef'),
    path('',views.home,name='home')
]
