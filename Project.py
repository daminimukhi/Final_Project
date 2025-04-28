# Mental Health Prediction using BERT + BiLSTM (Fixed Input Issue)

import os, random, re, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from torch.optim.lr_scheduler import OneCycleLR

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Load dataset
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

# Define columns
text_col = '16_following_the_previous_question_how_do_you_feel_about_these_comparisons_generally_speaking'
target_col = '13_on_a_scale_of_1_to_5_how_much_are_you_bothered_by_worries'
df.dropna(subset=[text_col, target_col], inplace=True)
df[text_col] = df[text_col].apply(clean_text)
df[target_col] = df[target_col].astype(str)

unique_labels = sorted(df[target_col].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}
df['label'] = df[target_col].map(label_map)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Convert input to string safely
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': text
        }

# Prepare data
texts = df[text_col].tolist()
labels = df['label'].tolist()
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=SEED)
train_dataset = MentalHealthDataset(X_train, y_train, tokenizer)
val_dataset = MentalHealthDataset(X_val, y_val, tokenizer)

# Weighted sampler
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
class BERT_LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=len(label_map)):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(self.dropout(pooled))

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=30, anneal_strategy='linear')

# Training
best_f1, patience, epochs_no_improve = 0.0, 5, 0
train_losses, val_losses, val_accuracies, val_f1_scores = [], [], [], []

for epoch in range(30):
    model.train(); total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval(); predictions, targets_val, val_loss_total = [], [], 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            val_loss_total += criterion(outputs, labels).item()
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets_val.extend(labels.cpu().numpy())

    val_acc = accuracy_score(targets_val, predictions)
    val_f1 = f1_score(targets_val, predictions, average='weighted')
    val_losses.append(val_loss_total / len(val_loader))
    val_accuracies.append(val_acc)
    val_f1_scores.append(val_f1)

    print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

    if val_f1 > best_f1: best_f1, epochs_no_improve = val_f1, 0; torch.save(model.state_dict(), "best_model.pt")
    else: epochs_no_improve += 1
    if epochs_no_improve >= patience: print("Early stopping."); break

# Evaluation
model.load_state_dict(torch.load("best_model.pt")); model.eval()
predictions, targets = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

class_names = [reverse_label_map[i] for i in range(len(label_map))]
cm = confusion_matrix(targets, predictions, labels=list(range(len(label_map))))
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

print("\nClassification Report:\n")
print(classification_report(targets, predictions, target_names=class_names))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_title("Confusion Matrix (Counts)"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax2)
ax2.set_title("Normalized Confusion Matrix"); ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

plt.tight_layout(); plt.savefig("confusion_matrix_side_by_side.png"); plt.show()
