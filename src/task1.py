import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample sentences
sentences = ["I love coding", "Programming is fun"]

# Tokenize and encode
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Print embeddings
for i, emb in enumerate(embeddings):
    print(f"Embedding for sentence {i+1}: {emb.numpy()[:5]}...")  # Show first 5 dimensions

# Cosine similarity
cos_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
print(f"Cosine similarity: {cos_sim.item()}")