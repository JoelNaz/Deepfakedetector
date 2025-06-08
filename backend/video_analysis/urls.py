from django.urls import path
from . import views

urlpatterns = [
    path('analyze-video/', views.analyze_video, name='analyze_video'),
    path('history/', views.analysis_history, name='analysis_history'),
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),

]