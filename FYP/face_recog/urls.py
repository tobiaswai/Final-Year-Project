from django.urls import path
from .views import capture_face, train_faces, recognize_faces

urlpatterns = [
    path('capture-face/', capture_face, name='capture_face'),
    path('train-faces/', train_faces, name='train_faces'),
    path('recognize-faces/', recognize_faces, name='recognize_faces'),
]