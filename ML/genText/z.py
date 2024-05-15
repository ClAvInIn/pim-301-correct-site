from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка токенизатора и модели RuGPT-3.5
tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/ruGPT-3.5-13B")
model = GPT2LMHeadModel.from_pretrained("ai-forever/ruGPT-3.5-13B")

translate_objects = {"person" : "человек", "bicycle" : "велосипед", "car" : "машина", "motorcycle" : "мотоцикл", "airplane" : "самолет", "bus" : "автобус", "train" : "поезд", "truck" : "грузовик", "boat" : "лодка", "traffic light" : "светофор", "fire hydrant" : "пожарный гидрант", "stop sign" : "знак стоп", "parking meter" : "паркомат", "bench" : "лавка", "bird" : "птица", "cat" : "кот", "dog" : "собака", "horse" : "лошадь", "sheep" : "овца", "cow" : "корова", "elephant" : "слон", "bear" : "медведь", "zebra" : "зебра", "giraffe" : "жирафа", "backpack" : "рюкзак", "umbrella" : "зонтик", "handbag" : "сумочка", "tie" : "галстук", "suitcase" : "чемодан", "frisbee" : "фрисби", "skis" : "лыжи", "snowboard" : "сноуборд", "sports ball" : "спортивный мяч", "kite" : "летающий змей", "baseball bat" : "бейсбольная бита", "baseball glove" : "бейсбольная перчатка", "skateboard" : "скейтборд", "surfboard" : "доска для серфинга", "tennis racket" : "теннисная ракетка", "bottle" : "бутылка", "wine glass" : "бокал для вина", "cup" : "чашка", "fork" : "вилка", "knife" : "нож", "spoon" : "ложка", "bowl" : "чаша", "banana" : "банан", "apple" : "яблоко", "sandwich" : "бутерброд", "orange" : "апельсин", "broccoli" : "брокколи", "carrot" : "морковь", "hot dog" : "хот-дог", "pizza" : "пицца", "donut" : "пончик", "cake" : "торт", "chair" : "стул", "couch" : "диван", "potted plant" : "растение в горшке", "bed" : "кровать", "dining table" : "обеденный стол", "toilet" : "туалет", "tv" : "ТВ", "laptop" : "ноутбук", "mouse" : "мышь", "remote" : "удаленный", "keyboard" : "клавиатура", "cell phone" : "сотовый телефон", "microwave" : "микроволновая печь", "oven" : "печь", "toaster" : "тостер", "sink" : "раковина", "refrigerator" : "холодильник", "book" : "книга", "clock" : "часы", "vase" : "ваза", "scissors" : "ножницы", "teddy bear" : "плюшевый мишка", "hair drier" : "фен", "toothbrush" : "зубная щетка"}

def genereteDescription(promt):
  print(0)
  input_text = ""
  scene = promt['scene'].lower()
  for key, value in promt['objects'].items():
    input_text = input_text + str(value) + " " + key.replace(key, translate_objects[key]) + " "
  
  print(input_text)

  input_ids = tokenizer.encode(input_text, return_tensors="pt")

  output = model.generate(input_ids, 
                        max_length=100, 
                        num_return_sequences=3, 
                        pad_token_id=tokenizer.eos_token_id,
                        early_stopping=True,
                        num_beams=3)
  
  return tokenizer.decode(output[0], skip_special_tokens=True )

  # for i, sample_output in enumerate(output):
  #   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

print(genereteDescription({
    "scene": "Улица",
    "objects": {
        "car": 21,
        "person": 12,
        "truck": 1
    }
}))