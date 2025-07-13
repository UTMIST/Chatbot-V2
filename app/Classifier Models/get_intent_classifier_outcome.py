import torch
from transformers import RobertaTokenizer
from Intent_Classifier import RobertaClassifier, LABELS, MAX_LENGTH, MODEL_SAVE_PATH

def initialize_intent_classifier():
    global model, tokenizer, device

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer (same as used in training)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Instantiate the model using the imported model definition
    model = RobertaClassifier(len(LABELS)).to(device)

    # Load the saved model weights
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


def get_binary_outcome(input_text: str) -> dict:
    """
    Given an input text (string), load the saved model (using definitions from the training code),
    process the text, and return a dictionary mapping each label to its binary outcome (0 or 1).
    """
    # Tokenize the input text
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Convert logits to probabilities and apply threshold (0.5) to get binary outcomes
    preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()[0]

    # Create and return a dictionary mapping each label to its binary prediction
    # binary_outcome = {label: int(pred) for label, pred in zip(LABELS, preds)}

    res = [LABELS[i] for i in range(len(preds)) if preds[i]]
    return res

# Example usage:
if __name__ == "__main__":
    sample_text = "Can you recommend some resources on machine learning?"
    outcome = get_binary_outcome(sample_text)
    print("Input text:", sample_text)
    print("Binary Outcome:", outcome)