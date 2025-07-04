# -*- coding: utf-8 -*-
"""ClassifierRelevance.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EhuqXmMya33Rkp0zpM6lt_mLeToF5lgm
"""

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
# Hyperparameters and File Paths
EPOCHS_PHASE1 = 5         # Number of epochs to train the two initial classifiers
EPOCHS_PHASE3 = 3         # Number of epochs to train the third classifier
BATCH_SIZE = 16           # Batch size
LEARNING_RATE = 2e-5      # Learning rate
MAX_LENGTH = 128          # Maximum token length for RoBERTa
# File paths (change these to your local paths)
LABELLED_FILE = r"/data/RelevanceDataLabelled.csv"     # e.g., "./data/labeled_data.csv"
UNLABELLED_FILE = r"/data/RelevanceDataUnlabelled.csv" # e.g., "./data/unlabeled_data.csv"
# Model save path
MODEL_SAVE_PATH = "final_model_weights.pth"
USE_AMP = True
#############################################

# Enable benchmark for faster runtime if using GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Custom Dataset for text (with or without labels)
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        """
        texts: list of text strings
        labels: list of 0/1 if labeled, else None
        tokenizer: a HuggingFace tokenizer (e.g., RobertaTokenizer)
        max_length: integer, maximum sequence length
        """
        self.texts = texts
        self.labels = labels
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
        # Squeeze out the batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return input_ids, attention_mask, label
        else:
            return input_ids, attention_mask

class RobertaBinaryClassifier(nn.Module):
    def __init__(self):
        super(RobertaBinaryClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        # For a single binary label, output dimension is 1
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (index 0) as pooled representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits shape: [batch_size, 1]
        return logits

def train_epoch(model, data_loader, optimizer, device, criterion, use_amp):
    """
    Training loop for one epoch.
    We use BCEWithLogitsLoss, so 'logits' are raw scores,
    and we apply a sigmoid + threshold for accuracy.
    """
    model.train()
    losses = []
    total_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in data_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask).view(-1)  # shape: [batch_size]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        # Predictions: apply sigmoid, threshold at 0.5
        preds = torch.sigmoid(logits) > 0.5
        correct = (preds == (labels > 0.5)).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

    avg_loss = np.mean(losses)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def eval_model(model, data_loader, device, criterion, use_amp):
    """
    Evaluation loop for validation/test sets.
    """
    model.eval()
    losses = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask).view(-1)
                loss = criterion(logits, labels)

            losses.append(loss.item())

            preds = torch.sigmoid(logits) > 0.5
            correct = (preds == (labels > 0.5)).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

    avg_loss = np.mean(losses)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def predict_binary(model, data_loader, device, use_amp):
    """
    Get binary predictions for unlabeled data using the trained model.
    Returns a list of 0/1 predictions.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                input_ids, attention_mask, _ = [x.to(device) for x in batch]
            else:
                input_ids, attention_mask = [x.to(device) for x in batch]

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask).view(-1)

            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)

    return all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    pin_memory = (device.type == "cuda")

    #------------------------------------------------------
    # 1) LOAD LABELED DATA
    #   "Relevant" is 0 or 1.
    #------------------------------------------------------
    df = pd.read_csv(LABELLED_FILE)
    texts = df["Text"].tolist()
    labels = df["Relevance"].tolist()  # 0/1

    #------------------------------------------------------
    # 2) SPLIT INTO TRAIN/VAL/TEST
    #------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp
    )
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

    #------------------------------------------------------
    # 3) CREATE DATASETS/DATALOADERS
    #------------------------------------------------------
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=MAX_LENGTH)
    val_dataset   = TextDataset(X_val,   y_val,   tokenizer, max_length=MAX_LENGTH)
    test_dataset  = TextDataset(X_test,  y_test,  tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, pin_memory=pin_memory)

    #------------------------------------------------------
    # 4) PHASE 1: TRAIN TWO SEPARATE CLASSIFIERS ON LABELED DATA
    #------------------------------------------------------
    model1 = RobertaBinaryClassifier().to(device)
    model2 = RobertaBinaryClassifier().to(device)

    optimizer1 = optim.AdamW(model1.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.AdamW(model2.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print("\n==== PHASE 1: Train Two Classifiers on Labeled Data ====\n")
    for epoch in range(1, EPOCHS_PHASE1 + 1):
        # Train model1
        train_loss1, train_acc1 = train_epoch(model1, train_loader, optimizer1, device, criterion, USE_AMP)
        val_loss1, val_acc1 = eval_model(model1, val_loader, device, criterion, USE_AMP)

        # Train model2
        train_loss2, train_acc2 = train_epoch(model2, train_loader, optimizer2, device, criterion, USE_AMP)
        val_loss2, val_acc2 = eval_model(model2, val_loader, device, criterion, USE_AMP)

        print(f"Epoch {epoch}/{EPOCHS_PHASE1}")
        print(f"Model1 -> Train Loss: {train_loss1:.4f} | Train Acc: {train_acc1:.4f} | "
              f"Val Loss: {val_loss1:.4f} | Val Acc: {val_acc1:.4f}")
        print(f"Model2 -> Train Loss: {train_loss2:.4f} | Train Acc: {train_acc2:.4f} | "
              f"Val Loss: {val_loss2:.4f} | Val Acc: {val_acc2:.4f}\n")

    #------------------------------------------------------
    # 5) PHASE 2: PSEUDO-LABEL THE UNLABELED DATA
    #    Only keep samples for which model1 and model2 agree exactly.
    #------------------------------------------------------
    print("==== PHASE 2: Label Unlabeled Data with Both Classifiers ====\n")
    df_unlabeled = pd.read_csv(UNLABELLED_FILE)
    unlabeled_texts = df_unlabeled["Text"].tolist()

    unlabeled_dataset = TextDataset(unlabeled_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, pin_memory=pin_memory)

    preds1 = predict_binary(model1, unlabeled_loader, device, USE_AMP)  # list of 0/1
    preds2 = predict_binary(model2, unlabeled_loader, device, USE_AMP)  # list of 0/1

    pseudo_texts = []
    pseudo_labels = []
    for text, p1, p2 in zip(unlabeled_texts, preds1, preds2):
        if p1 == p2:
            pseudo_texts.append(text)
            pseudo_labels.append(p1)  # either 0 or 1

    print(f"Pseudo-labeled {len(pseudo_texts)} out of {len(unlabeled_texts)} unlabeled samples.\n")

    #------------------------------------------------------
    # 6) PHASE 3: TRAIN A THIRD CLASSIFIER ON
    #    (TRAINING DATA + PSEUDO-LABELED DATA)
    #------------------------------------------------------
    print("==== PHASE 3: Train Third Classifier on Combined Data ====\n")
    combined_texts = X_train + pseudo_texts
    combined_labels = y_train + pseudo_labels

    combined_dataset = TextDataset(combined_texts, combined_labels, tokenizer, max_length=MAX_LENGTH)
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)

    model3 = RobertaBinaryClassifier().to(device)
    optimizer3 = optim.AdamW(model3.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS_PHASE3 + 1):
        train_loss, train_acc = train_epoch(model3, combined_loader, optimizer3, device, criterion, USE_AMP)
        val_loss, val_acc = eval_model(model3, val_loader, device, criterion, USE_AMP)
        print(f"Epoch {epoch}/{EPOCHS_PHASE3}")
        print(f"Model3 -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")

    torch.save(model3.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model weights saved to {MODEL_SAVE_PATH}")

    #------------------------------------------------------
    # 7) Evaluate on Test Set
    #------------------------------------------------------
    test_loss, test_acc = eval_model(model3, test_loader, device, criterion, USE_AMP)
    print(f"\n==== Test Results ====\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")

    #------------------------------------------------------
    # 8) Demo: Load the saved model and print predictions
    #------------------------------------------------------
    model_loaded = RobertaBinaryClassifier().to(device)
    model_loaded.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model_loaded.eval()

    example_count = 5
    print(f"Binary predictions on {example_count} sample test inputs:")
    for i in range(min(example_count, len(X_test))):
        text = X_test[i]
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model_loaded(inputs["input_ids"], inputs["attention_mask"]).view(-1)
        pred = (torch.sigmoid(logits) > 0.5).long().item()
        print(f"Text: {text}\nPredicted Relevant?: {pred} (Actual={y_test[i]})\n")

if __name__ == "__main__":
    main()