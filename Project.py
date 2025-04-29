# Project.py
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from PIL import Image, ImageTk, ImageFilter

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define label map
reverse_label_map = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5'
}

class BERT_LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=len(reverse_label_map)):
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERT_LSTM().to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# GUI Form
class StyledFormApp:
    def __init__(self, root):
        self.root = root
        root.title("Mental Health Self-Assessment")
        root.geometry("900x700")

        self.canvas = tk.Canvas(root, width=900, height=700)
        self.bg_img = Image.open("background.jpg").resize((900, 700)).filter(ImageFilter.GaussianBlur(radius=3))
        self.bg_img_tk = ImageTk.PhotoImage(self.bg_img)
        self.canvas.create_image(0, 0, image=self.bg_img_tk, anchor="nw")
        self.canvas.pack(fill="both", expand=True)

        self.questions = [
            ("How do you feel after comparing yourself on social media?", ["Motivated", "Jealous", "Inspired", "Insecure", "Neutral"]),
            ("How often do you seek validation from likes/comments?", ["Very Often", "Often", "Sometimes", "Rarely", "Never"]),
            ("Do you feel anxious when away from social media?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
            ("Do you compare your appearance to people online?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
            ("How often does social media affect your mood?", ["Very Often", "Often", "Sometimes", "Rarely", "Never"]),
            ("Do you feel pressure to keep up with others on social media?", ["Yes, frequently", "Yes, sometimes", "Not much", "Not at all"]),
            ("Have you ever taken a break from social media for mental health?", ["Yes", "No", "Considering it", "Tried but failed"])
        ]
        self.responses = []
        self.q_index = 0
        self.setup_question()

    def setup_question(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.bg_img_tk, anchor="nw")

        question, options = self.questions[self.q_index]
        self.canvas.create_text(450, 50, text=f"Question {self.q_index + 1}", font=("Georgia", 30, "bold"), fill="black")
        self.canvas.create_text(450, 100, text=question, font=("Georgia", 24), fill="black")

        self.selected_option = tk.StringVar()
        y_offset = 160
        for opt in options:
            rb = ttk.Radiobutton(self.canvas, text=opt, variable=self.selected_option, value=opt)
            rb.configure(style="Custom.TRadiobutton")
            self.canvas.create_window(450, y_offset, window=rb)
            y_offset += 35

        style = ttk.Style()
        style.configure("Custom.TRadiobutton", font=("Georgia", 22, 'bold'), fill="black")

        btn = tk.Button(self.root, text="Next", command=self.next_question, font=("Georgia", 22, "bold"), bg="black", fg="black")
        self.canvas.create_window(450, y_offset + 20, window=btn)

    def next_question(self):
        choice = self.selected_option.get()
        if not choice:
            messagebox.showwarning("Incomplete", "Please select an option before continuing.")
            return

        self.responses.append(choice)
        self.q_index += 1

        if self.q_index < len(self.questions):
            self.setup_question()
        else:
            self.show_result(" ".join(self.responses))

    def show_result(self, text):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.bg_img_tk, anchor="nw")

        enc = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        label = reverse_label_map[pred_class]
        interpretation = {
            '1': "🟢 Very Low Impact",
            '2': "🟢 Low Impact",
            '3': "🟡 Moderate Impact",
            '4': "🟠 High Impact",
            '5': "🔴 Severe Impact"
        }.get(label, "Unknown Impact")

        self.canvas.create_text(450, 70, text="🎯 Your Result", font=("Georgia", 20, "bold"), fill="black")
        self.canvas.create_text(450, 130, text=f"Impact Level: {label} ({interpretation})", font=("Georgia", 16), fill="black")
        self.canvas.create_text(450, 190, text=f"Confidence Score: {confidence:.2f}", font=("Georgia", 14), fill="black")

        self.canvas.create_text(450, 250, text="Precautions and Tips:", font=("Georgia", 18, "bold"), fill="black")
        suggestions = [
            "• Take regular breaks from social media",
            "• Follow positive and motivating accounts",
            "• Seek professional help if needed",
            "• Practice mindfulness and gratitude",
            "• Stay connected with friends and family"
        ]
        y = 290
        for s in suggestions:
            self.canvas.create_text(450, y, text=s, font=("Georgia", 12), fill="black")
            y += 25

if __name__ == "__main__":
    root = tk.Tk()
    app = StyledFormApp(root)
    root.mainloop()
