# from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

checkpoint = "dslim/bert-base-NER"
# tokenizer = AutoTokenizer.from_pretrained()
# model = AutoModelForTokenClassification.from_pretrained(
#     "dslim/bert-base-NER")

nlp = pipeline("ner", model=checkpoint, tokenizer=checkpoint)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)
