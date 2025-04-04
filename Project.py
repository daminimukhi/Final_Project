# Mental Health Prediction using BERT + BiLSTM

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import kagglehub
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Download and load dataset
print("\U0001F4C5 Downloading dataset...")
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

# Sort unique labels
unique_labels = sorted(df[target_col].unique(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
df['label'] = df[target_col].map(label_map)
class_names = [reverse_label_map[i] for i in range(len(label_map))]

# Step 4: Tokenization
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Step 6: Split dataset
texts = df[text_col].tolist()
labels = df['label'].tolist()
dataset = MentalHealthDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Step 7: Define model
class BERT_LSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=len(label_map)):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        pooled = torch.mean(lstm_out, 1)
        output = self.dropout(pooled)
        return self.fc(output)

# Step 8: Training and evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

def train_model():
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"\U0001F4DA Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f}")

def evaluate_model():
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    print("\n\U0001F4CA Classification Report:\n", classification_report(targets, predictions, target_names=class_names))
    print("✅ Accuracy:", accuracy_score(targets, predictions))
    print("✅ Precision:", precision_score(targets, predictions, average='weighted', zero_division=0))
    print("✅ Recall:", recall_score(targets, predictions, average='weighted', zero_division=0))

    cm = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# Step 9: Execute
train_model()
evaluate_model()
