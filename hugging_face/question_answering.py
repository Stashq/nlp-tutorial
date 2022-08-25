from transformers import pipeline
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer


model_name = "deepset/roberta-base-squad2"

QA_input = {
    'question': 'Who Daniel does\'nt like?',
    'context': 'Daniel wondering how is it possible to be' +
    'such stupid as lady Elizabeth.'
}

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
res = nlp(QA_input)
print(res)

# # b) Load model & tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokens = tokenizer.encode(
#     QA_input['question'], QA_input['context'], padding=True,
#     return_tensors='pt')

# output = model(tokens)

# start_logits = output['start_logits']
# end_logits = output['end_logits']


# print(start_logits.shape, end_logits.shape, tokens.shape)
# print(start_logits.argmax(), end_logits.argmax())
# tokenizer.decode(tokens[0, start_logits.argmax():end_logits.argmax()+1])
