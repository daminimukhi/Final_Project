# train_and_evaluate.py
import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download dataset
dataset_dir = kagglehub.dataset_download("souvikahmed071/social-media-and-mental-health")
csv_file = next(f for f in os.listdir(dataset_dir) if f.endswith('.csv'))
df = pd.read_csv(os.path.join(dataset_dir, csv_file))
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

text_col = '16_following_the_previous_question_how_do_you_feel_about_these_comparisons_generally_speaking'
target_col = '13_on_a_scale_of_1_to_5_how_much_are_you_bothered_by_worries'

df.dropna(subset=[text_col, target_col], inplace=True)
df[text_col] = df[text_col].apply(clean_text)

# Encode labels
unique_labels = sorted(df[target_col].unique(), key=lambda x: int(x))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
df['label'] = df[target_col].map(label_map)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

texts = df[text_col].tolist()
labels = df['label'].tolist()
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=SEED)

train_dataset = MentalHealthDataset(X_train, y_train, tokenizer)
val_dataset = MentalHealthDataset(X_val, y_val, tokenizer)

class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define Model
class BERT_LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(self.dropout(pooled))

model = BERT_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Train
for epoch in range(5):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        trues.extend(labels.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(trues, preds, target_names=[str(reverse_label_map[i]) for i in range(len(label_map))]))

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/best_model.pt")
print("\n✅ Model saved to saved_models/best_model.pt")
