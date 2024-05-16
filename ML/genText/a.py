from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")

# Получение темы от пользователя
main_topic = input("Введите основную тему: ")

# Получение дополнительных слов от пользователя
additional_words = input("Введите дополнительные слова или тему, которую вы хотите видеть: ")

# Объединение основной темы и дополнительных слов
prompt_text = main_topic + ", " + additional_words

# Преобразование текста в токены
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

# Генерация текста с улучшенными параметрами
output = model.generate(
    input_ids,
    max_length=100,
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
filtered_text = re.sub(r'(- [^\n]*:\s*)', '', generated_text)

print("Сгенерированное описание:", filtered_text)
