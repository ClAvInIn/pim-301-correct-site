import json
import base64
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ml.findObject.YOLO_Test import objectFinder
from ml.findScene.VGG16_classifier_location import predict
# from ml.genText.z import genereteDescription

def index(request):
    return render(request,'index.html')

def photo(request):
    return render(request, 'foto.html')

def video(request):
    return render(request, 'video.html')

@csrf_exempt
def descriptionImage(request):
    if request.method == 'POST':
        image = json.load(request)
        fileData = image['value']
        filePath = 'temp/'+image['name']

        with open(filePath, "wb") as fh: fh.write(base64.b64decode(fileData))
        scene = predict(filePath)
        objects = objectFinder(filePath)[0]
        # description = genereteDescription({"scene": scene, "objects":objects})
        description = -1
        os.remove(filePath)

        return JsonResponse({'status': 200, 'scene': scene, 'objects': objects, 'description' : description})

    else:
        return JsonResponse({'status': 404})