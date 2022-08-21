from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
res = classifier(sequence_to_classify, candidate_labels)

print("Single class:", res)

candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
res = classifier(sequence_to_classify, candidate_labels, multi_class=True)

print("Multi class:", res)
