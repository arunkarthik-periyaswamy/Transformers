from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torch

class MultiTaskBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.class_classifier = nn.Linear(768, 2)  # 2 classes for classification
        self.sent_classifier = nn.Linear(768, 2)   # 2 classes for sentiment

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        class_logits = self.class_classifier(pooled_output)
        sent_logits = self.sent_classifier(pooled_output)
        return class_logits, sent_logits

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MultiTaskBERT()

# Sample data preparation
sentences = ["I love coding", "Programming is fun"]
labels_class = [0, 1]  # Example class labels
labels_sent = [1, 1]   # Example sentiment labels

# Tokenize and prepare data
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids_train = inputs['input_ids']
attention_mask_train = inputs['attention_mask']
labels_class_train = torch.tensor(labels_class)
labels_sent_train = torch.tensor(labels_sent)

# Hypothetical data
train_data = TensorDataset(input_ids_train, attention_mask_train, labels_class_train, labels_sent_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Training loop
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, class_labels, sent_labels = batch
        optimizer.zero_grad()
        class_logits, sent_logits = model(input_ids, attention_mask)
        loss_class = nn.CrossEntropyLoss()(class_logits, class_labels)
        loss_sent = nn.CrossEntropyLoss()(sent_logits, sent_labels)
        loss = loss_class + loss_sent
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Evaluation (placeholder)
# Compute accuracy, F1 for test set