from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

start_text = input("Type start of sentence: ")
answer = generator(
    start_text, max_length=50, num_return_sequences=3)

print(answer)
