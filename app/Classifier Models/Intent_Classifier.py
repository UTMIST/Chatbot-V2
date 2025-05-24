#!/usr/bin/env python
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel

#############################################
# Hyperparameters and File Paths (edit as needed)
EPOCHS_PHASE1 = 5        # Number of epochs to train the two initial classifiers
EPOCHS_PHASE3 = 5         # Number of epochs to train the third classifier
BATCH_SIZE = 16           # Batch size
LEARNING_RATE = 2e-5      # Learning rate
MAX_LENGTH = 128          # Maximum token length for RoBERTa
# The label columns should match those in your CSV
LABELS = ["Provide_Preference", "Accept_Recommendation", "Reject_Recommendation",
          "Inquire_Resources", "Club_Related_Inquiry", "Short_Answer_Inquiry"]
NUM_LABELS = len(LABELS)

# File paths (change these to your local paths)
LABELLED_FILE = r"app\data\Combined_Dataset.csv"     # e.g., "./data/labelled.csv"
# LABELLED_FILE = "app/data/Combined_Dataset.csv"       # for MacOS

UNLABELLED_FILE = r"app\data\unlabelled.csv"         # e.g., "./data/unlabelled.csv"
# UNLABELLED_FILE = "app/data/unlabelled.csv"           # for MacOS

# Model save path
MODEL_SAVE_PATH = r"app\Classifier Models\intent_classification.pth"
# MODEL_SAVE_PATH = "app/Classifier Models/intent_classification.pth"       # for MacOS

# Use Automatic Mixed Precision if using CUDA
USE_AMP = True
#############################################

# Enable benchmark for faster runtime if using GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Custom Dataset for text (with or without labels)
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels  # For multi-label, each label should be a multi-hot vector
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Squeeze to remove the extra batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if self.labels is not None:
            # Convert label to a float tensor for BCEWithLogitsLoss
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return input_ids, attention_mask, label
        else:
            return input_ids, attention_mask

# Roberta-based classifier (remains mostly unchanged)
class RobertaClassifier(nn.Module):
    def __init__(self, num_labels):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the representation of the first token (<s>) as the pooled output.
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Training function for one epoch with AMP (multi-label version)
def train_epoch(model, data_loader, optimizer, device, criterion, use_amp):
    model.train()
    losses = []
    total_correct = 0
    total_elements = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for batch in data_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(input_ids, attention_mask)  # shape: [batch_size, NUM_LABELS]
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        # Apply sigmoid and threshold at 0.5 for multi-label predictions
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_elements += torch.numel(labels)
    avg_acc = total_correct / total_elements
    return np.mean(losses), avg_acc

# Evaluation function (multi-label version)
def eval_model(model, data_loader, device, criterion, use_amp):
    model.eval()
    losses = []
    total_correct = 0
    total_elements = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_elements += torch.numel(labels)
    avg_acc = total_correct / total_elements
    return np.mean(losses), avg_acc

# Function to predict labels (for pseudo-labeling) with AMP (multi-label version)
def predict(model, data_loader, device, use_amp):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            # For unlabelled data, the batch has only (input_ids, attention_mask)
            if len(batch) == 3:
                input_ids, attention_mask, _ = [x.to(device) for x in batch]
            else:
                input_ids, attention_mask = [x.to(device) for x in batch]
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask)
            # Apply sigmoid and threshold to get binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(preds.cpu().numpy())
    return predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    pin_memory = True if device.type == "cuda" else False

    # ---- Data Loading ----
    # Load labelled data; assumes first column is 'Prompt' and the rest are label columns.
    df = pd.read_csv(LABELLED_FILE)
    texts = df["Prompt"].tolist()
    label_cols = df.columns[1:]
    # For multi-label, use the entire multi-hot vector rather than argmax.
    labels = df[label_cols].values.tolist()  # Each row is like [0, 1, 0, 0, 1, 0]
    # Ensure labels are float values
    labels = [[float(val) for val in row] for row in labels]

    # For stratification (if needed), we use the index of the first positive label
    stratify_labels = [np.argmax(row) for row in labels]

    # Split the labelled data into 70% train, 20% validation, and 10% test.
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=stratify_labels
    )
    stratify_temp = [np.argmax(row) for row in y_temp]
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=42, stratify=stratify_temp
    )

    # Create PyTorch datasets and dataloaders for the labelled data
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=MAX_LENGTH)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=pin_memory)

    # ---- Phase 1: Train Two Classifiers on Labelled Data ----
    print("\nPhase 1: Training two classifiers on labelled data")
    model1 = RobertaClassifier(NUM_LABELS).to(device)
    model2 = RobertaClassifier(NUM_LABELS).to(device)
    optimizer1 = optim.AdamW(model1.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.AdamW(model2.parameters(), lr=LEARNING_RATE)
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        # Train model1 for one epoch
        train_loss1, train_acc1 = train_epoch(model1, train_loader, optimizer1, device, criterion, USE_AMP)
        val_loss1, val_acc1 = eval_model(model1, val_loader, device, criterion, USE_AMP)
        # Train model2 for one epoch (using the same training data)
        train_loss2, train_acc2 = train_epoch(model2, train_loader, optimizer2, device, criterion, USE_AMP)
        val_loss2, val_acc2 = eval_model(model2, val_loader, device, criterion, USE_AMP)

        print(f"\nEpoch {epoch}/{EPOCHS_PHASE1}")
        print(f"Model1 -> Train Loss: {train_loss1:.4f} | Train Acc: {train_acc1:.4f} || Val Loss: {val_loss1:.4f} | Val Acc: {val_acc1:.4f}")
        print(f"Model2 -> Train Loss: {train_loss2:.4f} | Train Acc: {train_acc2:.4f} || Val Loss: {val_loss2:.4f} | Val Acc: {val_acc2:.4f}")

    # ---- Phase 2: Pseudo-Label Unlabelled Data ----
    print("\nPhase 2: Labeling unlabelled data with the two classifiers")
    df_unlabelled = pd.read_csv(UNLABELLED_FILE)
    unlabelled_texts = df_unlabelled["Prompt"].tolist()
    unlabelled_dataset = TextDataset(unlabelled_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH)
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=BATCH_SIZE, pin_memory=pin_memory)

    preds1 = predict(model1, unlabelled_loader, device, USE_AMP)
    preds2 = predict(model2, unlabelled_loader, device, USE_AMP)

    pseudo_texts = []
    pseudo_labels = []
    # Only add samples where both models agree exactly on the multi-label prediction
    for text, p1, p2 in zip(unlabelled_texts, preds1, preds2):
        if np.array_equal(p1, p2):
            pseudo_texts.append(text)
            pseudo_labels.append(p1.tolist())

    print(f"Pseudo-labelled {len(pseudo_texts)} out of {len(unlabelled_texts)} unlabelled samples (only where both models agree exactly).")

    # ---- Phase 3: Train Third Classifier on Combined Data ----
    print("\nPhase 3: Training a third classifier on all (labelled + pseudo-labelled) data")
    # Combine the original training data with pseudo-labelled data
    combined_texts = X_train + pseudo_texts
    combined_labels = y_train + pseudo_labels

    combined_dataset = TextDataset(combined_texts, combined_labels, tokenizer, max_length=MAX_LENGTH)
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)

    model3 = RobertaClassifier(NUM_LABELS).to(device)
    optimizer3 = optim.AdamW(model3.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS_PHASE3 + 1):
        train_loss, train_acc = train_epoch(model3, combined_loader, optimizer3, device, criterion, USE_AMP)
        val_loss, val_acc = eval_model(model3, val_loader, device, criterion, USE_AMP)
        print(f"\nEpoch {epoch}/{EPOCHS_PHASE3}")
        print(f"Model3 -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    #Optionally, save the final model weights
    torch.save(model3.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
