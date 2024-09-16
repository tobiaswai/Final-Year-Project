from django.shortcuts import render
from django.http import HttpResponse
from recognition.forms import FaceRecognitionform
from recognition.machinelearning import pipeline_model
from django.conf import settings
from recognition.models import FaceRecognition
import os

def index(request):
    form = FaceRecognitionform()

    if request.method == "POST":
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)

            primary_key = save.pk
            imgobj = FaceRecognition.objects.get(pk=primary_key)
            fileroot = str(imgobj.imgae)
            filepath = os.path.join(settings.MDEIA_ROOT,fileroot)
            results = pipeline_model(filepath)
            print(results)

            return render(request,'index.html',{'form':form,'upload':True,'results':results})

    return render(request,'index.html', {'form':form,'upload':False})