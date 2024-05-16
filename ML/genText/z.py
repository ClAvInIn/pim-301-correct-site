from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")

translate_objects = {"person" : "человек", "bicycle" : "велосипед", "car" : "машина", "motorcycle" : "мотоцикл", "airplane" : "самолет", "bus" : "автобус", "train" : "поезд", "truck" : "грузовик", "boat" : "лодка", "traffic light" : "светофор", "fire hydrant" : "пожарный гидрант", "stop sign" : "знак стоп", "parking meter" : "паркомат", "bench" : "лавка", "bird" : "птица", "cat" : "кот", "dog" : "собака", "horse" : "лошадь", "sheep" : "овца", "cow" : "корова", "elephant" : "слон", "bear" : "медведь", "zebra" : "зебра", "giraffe" : "жирафа", "backpack" : "рюкзак", "umbrella" : "зонтик", "handbag" : "сумочка", "tie" : "галстук", "suitcase" : "чемодан", "frisbee" : "фрисби", "skis" : "лыжи", "snowboard" : "сноуборд", "sports ball" : "спортивный мяч", "kite" : "летающий змей", "baseball bat" : "бейсбольная бита", "baseball glove" : "бейсбольная перчатка", "skateboard" : "скейтборд", "surfboard" : "доска для серфинга", "tennis racket" : "теннисная ракетка", "bottle" : "бутылка", "wine glass" : "бокал для вина", "cup" : "чашка", "fork" : "вилка", "knife" : "нож", "spoon" : "ложка", "bowl" : "чаша", "banana" : "банан", "apple" : "яблоко", "sandwich" : "бутерброд", "orange" : "апельсин", "broccoli" : "брокколи", "carrot" : "морковь", "hot dog" : "хот-дог", "pizza" : "пицца", "donut" : "пончик", "cake" : "торт", "chair" : "стул", "couch" : "диван", "potted plant" : "растение в горшке", "bed" : "кровать", "dining table" : "обеденный стол", "toilet" : "туалет", "tv" : "ТВ", "laptop" : "ноутбук", "mouse" : "мышь", "remote" : "удаленный", "keyboard" : "клавиатура", "cell phone" : "сотовый телефон", "microwave" : "микроволновая печь", "oven" : "печь", "toaster" : "тостер", "sink" : "раковина", "refrigerator" : "холодильник", "book" : "книга", "clock" : "часы", "vase" : "ваза", "scissors" : "ножницы", "teddy bear" : "плюшевый мишка", "hair drier" : "фен", "toothbrush" : "зубная щетка"}

def genereteDescription(promt):
  objects = ""
  scene = promt['scene'].lower()
  prompt_text = f'Что происходит на {scene} '

  if len(promt['objects']) > 0 :
    for key, value in promt['objects'].items():
      objects = objects + str(value) + " " + key.replace(key, translate_objects[key]) + ", "
  
    objects = objects[:-2]
    prompt_text += f', где расположена {objects}'

  prompt_text += '?'

  
  input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

  output = model.generate( input_ids,
      max_length=40,
      num_return_sequences=1,
      temperature=1.0,
      top_k=50,
      top_p=0.9,
      repetition_penalty=1.2,
      do_sample=True
  )
    
  # Преобразование выходных данных в текст и вывод результатов
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

  # Фильтрация диалоговых ответов
  return "\n".join(re.sub(r'(- [^\n]*:\s*)', '', generated_text).split("\n")[1:])


print(genereteDescription({
    "scene": "Улица",
    "objects": {
        "car": 21,
        "person": 12,
        "truck": 1
    }
}))