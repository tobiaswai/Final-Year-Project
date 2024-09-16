from django import forms
from recognition.models import FaceRecognition

class FaceRecognitionform(forms.ModelForm):

    class Meta:
        model = FaceRecognition
        fields = ['image']