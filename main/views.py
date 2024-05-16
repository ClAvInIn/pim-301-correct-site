import json
import base64
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ml.findObject.YOLO_Test import objectFinder
from ml.findScene.VGG16_classifier_location import predict
from ml.genText.z import genereteDescription

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

        if(not image): return JsonResponse({"error": "Empty data" },status = 406)
        if (not isBase64(image['value'])): return JsonResponse({ "error": "Wrong data"},status = 406)
        if(not(all(elem in image for elem in ["value", "name"]))): return JsonResponse({"error": "Incomplete data" },status = 406)
        
        fileData = image['value']
        filePath = 'temp/'+image['name']

        with open(filePath, "wb") as fh: fh.write(base64.b64decode(fileData))

        scene = predict(filePath)
        objects = objectFinder(filePath)[0]
        description = genereteDescription({"scene": scene, "objects":objects})
        os.remove(filePath)

        return JsonResponse({'scene': scene, 'objects': objects, 'description' : description}, status = 200 )

    else:
        return JsonResponse({"error": "Incorrect request method" },status = 405)
    

def isBase64(sb):
    try:
        if isinstance(sb, str):
            # If there's any unicode here, an exception will be thrown and the function will return false
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False