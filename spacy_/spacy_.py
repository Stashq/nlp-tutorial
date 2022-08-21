# import spacy
# from spacy.lang.en.examples import sentences
import spacy_streamlit
# import en_core_web_sm


# nlp = spacy.load("en_core_web_sm")
# doc = nlp(sentences[0])
# print(doc.is_tagged)


models = ["en_core_web_sm"]
default_text = "Sundar Pichai is the CEO of Google."
spacy_streamlit.visualize(models, default_text)
