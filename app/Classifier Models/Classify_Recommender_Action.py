import json
import argparse
import openai
import os


openai.api_key = os.getenv("OPENAI-API-KEY")  # or: openai.api_key = "YOUR_API_KEY"

def load_constraints(json_file):
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_file}: {e}")
        exit(1)

def construct_prompt(user_input, constraints):
    # Extract hard and soft constraints from the JSON data
    hard = constraints.get("hard_constraints", {})
    soft = constraints.get("soft_constraints", {})

    # Check if there are any hard constraints
    has_hard_constraints = any(value not in [None, "N/A", ""] for value in hard.values())
    print(has_hard_constraints)
    # Define the list of possible intents
    intents_list = (
        "1. Provide_Preference, "
        "2. Accept_Recommendation, "
        "3. Reject_Recommendation, "
        "4. Inquire Resources, "
        "5. Club-related Inquiry, "
        "6. Short-Answer Inquiry"
    )

    # Build the prompt string
    prompt = (
        f"User Query: {user_input}\n\n"
        "### Hard Constraints ###\n"
        f"  1. Education Level: {hard.get('education_level', 'N/A')}\n"
        f"  2. Resource Type: {hard.get('resource_type', 'N/A')}\n"
        f"  3. Topic: {hard.get('topic', 'N/A')}\n"
        f"  4. Language: {hard.get('language', 'N/A')}\n"
        f"  5. Budget: {hard.get('budget', 'N/A')}\n\n"
        "### Soft Constraints ###\n"
        f"  1. Learning Style: {soft.get('learning_style', 'N/A')}\n"
        f"  2. Time Commitment: {soft.get('time_commitment', 'N/A')}\n"
        f"  3. Level of Depth: {soft.get('level_of_depth', 'N/A')}\n"
        f"  4. Preferred Topics: {soft.get('preferred_topics', 'N/A')}\n"
        f"  5. Format Preferences: {soft.get('format_preferences', 'N/A')}\n\n"
        f"### Possible User Intents (as implied by the query) ###\n"
        f"{intents_list}\n\n"
    )
    if has_hard_constraints:
        prompt += (
            "### Instructions ###\n"
            "Based on the user's query and the provided constraints, classify the system action "
            "into one of the following three exact phrases (with exact capitalization):\n"
            "  - 'Request Information'\n"
            "  - 'Give Recommendation'\n"
            "  - 'Answer a Question'\n\n"
            "### Mapping Guidance ###\n"
            "  - If the user's intent is Inquire Resources, Club-related Inquiry, or Short-Answer Inquiry, choose 'Answer a Question'.\n"
            "  - If the user's intent is Provide_Preference or the query is ambiguous, choose 'Request Information' or 'Give Recommendation' depending on the situation.\n\n"
        )
    else:
        prompt += (
            "### Instructions ###\n"
            "Based on the user's query and the provided constraints, classify the system action "
            "into one of the following three exact phrases (with exact capitalization):\n"
            "  - 'Request Information'\n"
            "  - 'Answer a Question'\n\n"
            "### Mapping Guidance ###\n"
            "  - If the user's intent is Inquire Resources, Club-related Inquiry, or Short-Answer Inquiry, choose 'Answer a Question'.\n"
            "  - If the user's intent is Provide_Preference or the query is ambiguous, choose 'Request Information' depending on the situation.\n\n"
        )

    # Ending reminder
    prompt += "Return only one of these phrases exactly as written."

    return prompt


def get_classified_action(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", 
                 "content": prompt}
            ],
            temperature=0.0  # Deterministic output
        )
        print(response)
        action = response["choices"][0]["message"]["content"].strip()
        return action
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        exit(1)

def main():
    print('Start the program')
    parser = argparse.ArgumentParser(description="Action Classifier using GPT API")
    parser.add_argument(
        "--json_file",
        type=str,
        default="constraints.json",
        help="Path to the JSON file containing constraints"
    )
    args = parser.parse_args()

    # Load the constraints from the specified JSON file.
    constraints = load_constraints(args.json_file)
    
    # Get user input.
    user_input = input("Please enter your query: ").strip()
    if not user_input:
        print("No user input provided.")
        exit(1)

    # Construct the prompt for the GPT API.
    prompt = construct_prompt(user_input, constraints)
    
    # Get the classified system action from the GPT API.
    action = get_classified_action(prompt)
    
    print(f"Recommended Action: {action}")
