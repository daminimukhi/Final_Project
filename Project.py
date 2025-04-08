# Mental Health Prediction using BERT + BiLSTM (with validation, early stopping, saving model)

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
import re

# Step 1: Download and load dataset
print("Downloading dataset...")
dataset_dir = kagglehub.dataset_download("souvikahmed071/social-media-and-mental-health")
csv_file = next(f for f in os.listdir(dataset_dir) if f.endswith('.csv'))
df = pd.read_csv(os.path.join(dataset_dir, csv_file))

# Step 2: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

# Step 3: Select and prepare columns
text_col = '16_following_the_previous_question_how_do_you_feel_about_these_comparisons_generally_speaking'
target_col = '13_on_a_scale_of_1_to_5_how_much_are_you_bothered_by_worries'
df.dropna(subset=[text_col, target_col], inplace=True)
df[target_col] = df[target_col].astype(str)

unique_labels = sorted(df[target_col].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
df['label'] = df[target_col].map(label_map)
class_names = [reverse_label_map[i] for i in range(len(label_map))]

print("\nLabel distribution:")
label_counts = Counter(df['label'])
for label, count in label_counts.items():
    print(f"Class {label} ({reverse_label_map[label]}): {count} samples")

# Step 4: Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 5: Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': self.texts[idx]
        }

# Step 6: Prepare data
texts = df[text_col].tolist()
labels = df['label'].tolist()
dataset = MentalHealthDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Step 7: Model
class BERT_LSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=len(label_map)):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(self.dropout(pooled))

# Step 8: Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_LSTM().to(device)
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Step 9: Train with validation and early stopping
best_val_acc = 0.0
patience = 3
epochs_no_improve = 0
train_losses, val_accuracies = [], []

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))
    print(f"Epoch {epoch+1} Training Loss: {train_losses[-1]:.4f}")

    # Validation
    model.eval()
    predictions, targets_val = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets_val.extend(labels.cpu().numpy())

    val_acc = accuracy_score(targets_val, predictions)
    val_accuracies.append(val_acc)
    print(f"Validation Accuracy: {val_acc:.4f}\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Best model saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered.")
            break

# Step 10: Evaluation
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
predictions, targets, texts = [], [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        texts.extend(batch['text'])

unique_labels = sorted(set(targets + predictions))
class_names_eval = [reverse_label_map[i] for i in unique_labels]

print("\nClassification Report:")
print(classification_report(targets, predictions, target_names=class_names_eval))
print("Accuracy:", accuracy_score(targets, predictions))

cm = confusion_matrix(targets, predictions, labels=unique_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_eval, yticklabels=class_names_eval)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names_eval, yticklabels=class_names_eval)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Plot Learning Curve
plt.plot(train_losses, label='Training Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()
