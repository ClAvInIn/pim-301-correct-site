import os
from ultralytics import YOLO


model = YOLO('yolov8n.yaml')

# Определяем существует ли тренерованная модель
if os.path.isdir(r'runs/detect/train/weights'):
  model = YOLO('runs/detect/train/weights/best.pt')
else:
  model = YOLO('yolov8n.pt')

  results = model.train(data='coco128.yaml', epochs=3)
  results = model.val()
  success = model.export(format="torchscript")

# Получение списка именн классов в наборе данных
classesName = model.names

# Функция для получение сколько и каких объектов на изображении по URL
def objectFinder (url:str):
  results = model.predict(url, conf = 0.3, iou = 0.9)

  ret = []

  # Если передано несколько значений
  for res in results:
    boxes = res.boxes  # Объект Boxes для вывода ограничивающего прямоугольника

    # Создание объекта в котором перечисляются сколько каких объектов
    objects = {}
    for className, count in Counter(boxes.cls.tolist()).items(): objects[classesName[className]] = count

    ret.append(objects)

  return ret 
