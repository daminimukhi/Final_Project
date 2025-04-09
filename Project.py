# Mental Health Prediction using BERT + BiLSTM (Final Complete Version)

import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
import re

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Download and load dataset
print("Downloading dataset...")
dataset_dir = kagglehub.dataset_download("souvikahmed071/social-media-and-mental-health")
csv_file = next(f for f in os.listdir(dataset_dir) if f.endswith('.csv'))
df = pd.read_csv(os.path.join(dataset_dir, csv_file))

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

# Select and prepare columns
text_col = '16_following_the_previous_question_how_do_you_feel_about_these_comparisons_generally_speaking'
target_col = '13_on_a_scale_of_1_to_5_how_much_are_you_bothered_by_worries'
df.dropna(subset=[text_col, target_col], inplace=True)
df[target_col] = df[target_col].astype(str)

unique_labels = sorted(df[target_col].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
df['label'] = df[target_col].map(label_map)
class_names = [reverse_label_map[i] for i in range(len(label_map))]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class
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

# Split data
texts = df[text_col].tolist()
labels = df['label'].tolist()
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=SEED)
train_dataset = MentalHealthDataset(X_train, y_train, tokenizer)
val_dataset = MentalHealthDataset(X_val, y_val, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
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

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_LSTM().to(device)
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Training loop
best_val_acc = 0.0
patience = 3
epochs_no_improve = 0
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(15):
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    scheduler.step()

    model.eval()
    predictions, targets_val, probs_val = [], [], []
    val_loss_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            probs_val.extend(probs.cpu().numpy())
            targets_val.extend(labels.cpu().numpy())

    val_acc = accuracy_score(targets_val, predictions)
    val_losses.append(val_loss_total / len(val_loader))
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

# Evaluation
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
predictions, targets, texts, confidences = [], [], [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1).values
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        confidences.extend(confidence.cpu().numpy())
        texts.extend(batch['text'])

# Save predictions
df_results = pd.DataFrame({
    "Text": texts,
    "True": [reverse_label_map[t] for t in targets],
    "Predicted": [reverse_label_map[p] for p in predictions],
    "Confidence": confidences
})
df_results.to_csv("model_predictions.csv", index=False)

# Metrics
unique_labels = sorted(set(targets + predictions))
class_names_eval = [reverse_label_map[i] for i in unique_labels]
report = classification_report(targets, predictions, target_names=class_names_eval, output_dict=True)
print("\nClassification Report:")
for label in class_names_eval:
    print(f"{label}: Precision={report[label]['precision']:.2f}, Recall={report[label]['recall']:.2f}, F1={report[label]['f1-score']:.2f}")
print("Overall Accuracy:", accuracy_score(targets, predictions))

# Confusion Matrix (Raw + Normalized)
cm = confusion_matrix(targets, predictions, labels=unique_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names_eval, yticklabels=class_names_eval, ax=ax1)
ax1.set_title("Confusion Matrix (Counts)")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.tick_params(axis='x', rotation=45)

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='PuBu', xticklabels=class_names_eval, yticklabels=class_names_eval, ax=ax2)
ax2.set_title("Normalized Confusion Matrix")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("confusion_matrix_side_by_side.png")
plt.show()

# Learning Curve
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(train_losses, label='Train Loss', color='tab:blue', linestyle='--', marker='o')
ax1.plot(val_losses, label='Validation Loss', color='tab:orange', linestyle='--', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(val_accuracies, label='Validation Accuracy', color='tab:green', linestyle='-', marker='^')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='upper right')

plt.title("Model Learning Curve")
plt.tight_layout()
plt.savefig("learning_curve_combined.png")
plt.show()

# Misclassified Examples
print("\n❌ Top 5 Misclassified Samples (Lowest Confidence):")
mismatches = [(texts[i], reverse_label_map[targets[i]], reverse_label_map[predictions[i]], confidences[i])
              for i in range(len(predictions)) if predictions[i] != targets[i]]
mismatches = sorted(mismatches, key=lambda x: x[3])
for i, (text, true, pred, conf) in enumerate(mismatches[:5]):
    print(f"[{i+1}] True: {true} | Predicted: {pred} | Confidence: {conf:.2f}")
    print(f"Text: {text[:120]}...")
    print("-" * 80)
