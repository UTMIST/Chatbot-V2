import torch
from transformers import RobertaTokenizer
from classifierconstraint import RobertaClassifier, NUM_CLASSES, MAX_LENGTH, MODEL_SAVE_PATH

def initialize_constraint_classifier():
    global model, tokenizer, device

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer (same as used during training)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Instantiate the model using the imported model definition and load its weights
    model = RobertaClassifier(NUM_CLASSES).to(device)
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


def get_constraint_prediction(input_text: str) -> dict:
    """
    Given an input text (string), this function loads the saved constraint classifier,
    processes the text, and returns a dictionary with the predicted outcome.

    The prediction is a single label:
      0: soft constraint
      1: hard constraint
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

    # For single-label classification, choose the class with the highest logit
    pred_label = torch.argmax(outputs, dim=1).item()
    pred_label += 1 # CHANGING BACK TO 1 or 2

    # Optionally, map the numeric prediction to a human-readable format
    label_map = {1: 'soft', 2: 'hard'}
    return [label_map[pred_label]]

# Example usage:
if __name__ == "__main__":
    sample_text = "This text might impose a strict limitation on scheduling."
    prediction = get_constraint_prediction(sample_text)
    print("Input Text:", sample_text)
    print("Prediction:", prediction)