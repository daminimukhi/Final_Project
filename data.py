import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import kagglehub
import re

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset():
    dataset_dir = kagglehub.dataset_download("souvikahmed071/social-media-and-mental-health")
    csv_file = next(f for f in os.listdir(dataset_dir) if f.endswith('.csv'))
    df = pd.read_csv(os.path.join(dataset_dir, csv_file))
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

    text_col = '16_following_the_previous_question_how_do_you_feel_about_these_comparisons_generally_speaking'
    target_col = '13_on_a_scale_of_1_to_5_how_much_are_you_bothered_by_worries'

    df.dropna(subset=[text_col, target_col], inplace=True)
    df[text_col] = df[text_col].apply(clean_text)
    df[target_col] = df[target_col].astype(str)

    unique_labels = sorted(df[target_col].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {v: k for k, v in label_map.items()}

    df['label'] = df[target_col].map(label_map)
    return df, text_col, label_map, reverse_label_map

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.texts[idx]),
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
            'text': self.texts[idx]
        }
