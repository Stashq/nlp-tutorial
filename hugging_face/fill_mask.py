from transformers import pipeline


unmasker = pipeline('fill-mask', model='bert-base-uncased')
res = unmasker("Hello I'm a [MASK] model.")

print(res)
