from sentence_transformers import SentenceTransformer
import scipy


checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(checkpoint)

sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
# print(embeddings)

res = scipy.spatial.distance.cosine(*embeddings)
print(res)
