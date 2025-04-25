import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample sentences
sentences = ["I love coding", "Programming is fun"]

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classification_head = nn.Linear(768, 3)  # Task A: 3 classes
        self.sentiment_head = nn.Linear(768, 3)      # Task B: 3 sentiment classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        class_logits = self.classification_head(embeddings)
        sentiment_logits = self.sentiment_head(embeddings)
        return class_logits, sentiment_logits

# Example usage
model = MultiTaskSentenceTransformer()
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
class_logits, sentiment_logits = model(inputs['input_ids'], inputs['attention_mask'])
print(f"Classification logits: {class_logits}")
print(f"Sentiment logits: {sentiment_logits}")