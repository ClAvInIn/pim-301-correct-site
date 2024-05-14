from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка токенизатора и модели RuGPT-3.5
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")

# Ваш запрос
user_query = "Море красное"

# Токенизация запроса
input_ids = tokenizer.encode(user_query, return_tensors="pt")

# Генерация ответа
output = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Декодирование ответа
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Вывод сгенерированного текста
print(generated_text)
