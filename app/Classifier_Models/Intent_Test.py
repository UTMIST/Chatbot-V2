#!/usr/bin/env python
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel

# Define the same label names used in training
LABELS = ["Provide_Preference", "Accept_Recommendation", "Reject_Recommendation",
          "Inquire_Resources", "Club_Related_Inquiry", "Short_Answer_Inquiry"]
NUM_LABELS = len(LABELS)
MAX_LENGTH = 128
MODEL_SAVE_PATH = "model3_weights.pth"  # Make sure this matches the training save path

# Define the same model architecture
class RobertaClassifier(nn.Module):
    def __init__(self, num_labels):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaClassifier(NUM_LABELS)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("Enter your prompt (or type 'quit' to exit):")
    while True:
        prompt = input(">> ")
        if prompt.lower() == "quit":
            break

        inputs = tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1).item()

        # Return classification using the actual CSV column name
        print("Classification:", LABELS[pred])

if __name__ == "__main__":
    main()
