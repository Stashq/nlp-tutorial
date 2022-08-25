from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

sequence_to_classify = "In recent years government debt appeared to matter " +\
    "less and less even as countries borrowed more and more."
candidate_labels = [
    'politic', 'cooking', 'travelling', 'calendar', 'economy', 'sociology']
res = classifier(sequence_to_classify, candidate_labels)

print("Single class:\n", res, '\n')

res = classifier(sequence_to_classify, candidate_labels, multi_class=True)

print("Multi class:\n", res)
