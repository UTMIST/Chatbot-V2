import os
from openai import OpenAI

api_key = os.environ["OPENAI_API_KEY"] = "..."


def process_text_with_gpt3(input_file_path, output_file_path) :
    # Read input data from the text file
    with open ( input_file_path, 'r' ) as file :
        input_data = file.read ()

    client = OpenAI (
        # This is the default and can be omitted
        api_key=os.environ.get ( api_key ),
    )

    # Define the prompt for GPT-3
    prompt = f"Transform the following text into Q&A form. If not possible, provide it in point form:\n\n{input_data}"

    try :
        # Make a call to the OpenAI GPT-3 model
        chat_completion = client.chat.completions.create (
            model="gpt-3.5-turbo",  # You can use other models like "gpt-4" if available
            messages=[
                {"role" : "system", "content" : "You are a helpful assistant."},
                {"role" : "user", "content" : prompt}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Get the response text
        response_message_raw = chat_completion.choices[0].message.content
        response_message = response_message_raw.strip ()

        # Write the output data to a text file
        with open ( output_file_path, 'w' ) as file :
            file.write ( response_message )

        print ( f"Output has been written to {output_file_path}" )

    except Exception as e :
        print ( f"An error occurred: {e}" )


# Usage example
input_file = 'Copy of Immersion Night Planning.txt'  # Path to your input text file
output_file = 'refined.txt'  # Path to your output text file

process_text_with_gpt3 ( input_file, output_file)
