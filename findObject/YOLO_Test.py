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

# Инспекция медиа
results = model.predict("street.jpg", conf=0.3, iou = 0.9)


# Список результатов процесса
for result in results:
    boxes = result.boxes  # Объект Boxes для вывода ограничивающего прямоугольника
    result.show()  # Вывод размеченного изображения

    # Вывод результатов
    for cls, conf in zip(boxes.cls, boxes.conf):
      print(classesName[cls.item()], conf.item())