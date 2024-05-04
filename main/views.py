from django.shortcuts import render
def index(request):
    return render(request,'index.html')

def photo(request):
    return render(request, 'foto.html')

def video(request):
    return render(request, 'video.html')
