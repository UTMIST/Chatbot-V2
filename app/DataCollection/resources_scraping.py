import pandas as pd
from duckduckgo_search import DDGS
import openai
import requests
from bs4 import BeautifulSoup
import os
import os.path
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (including OPENAI_API_KEY)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

search_query = "Machine learning tutorials"
descriptions = {}

def generate_description(content):
    prompt = (
        "Generate a description for these machine learning resources. "
        "In the description identify constraints such as language requirements, "
        "system requirements, prerequisite knowledge, accessibility, budget, "
        "learning style, time commitment, level of depth, preferred topics and "
        "format preferences. Write the description as one long paragraph.:\n\n"
        f"{content}\n\nDescription:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    descriptions[content] = response.choices[0].message.content

# Perform DuckDuckGo search
results = DDGS().text(
    keywords=search_query,
    region='wt-wt',
    safesearch='off',
    timelimit='7d',
    max_results=200
)

results_df = pd.DataFrame(results)
second_column = results_df['href']

# Generate GPT descriptions for each URL
for url in second_column:
    generate_description(url)

# Build a DataFrame of Link + Description
new_data = []
for key, desc in descriptions.items():
    new_data.append({
        'Link': key,
        'Description': desc
    })
new_data = pd.DataFrame(new_data)

# Path to your output CSV
output_path = r"C:\Users\User\Desktop\chatbot\ChatbotV2\app\DataCollection\resources.csv"

# Write header on first run, append thereafter
if not os.path.isfile(output_path):
    new_data.to_csv(
        output_path,
        mode="w",       # create file
        header=True,    # write "Link,Description"
        index=False
    )
else:
    new_data.to_csv(
        output_path,
        mode="a",       # append to existing
        header=False,   # no header row
        index=False
    )

def extract(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator="\n", strip=True)
    return None

def extract_links(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if "https:" in href:
                links.append(href)
        return links
    return []

# Uncomment and adapt the following for UofT course descriptions:

# def generate_description_courses(content):
#     prompt = (
#         "Generate a description for these machine learning courses at the University of Toronto. "
#         "In the description identify constraints such as language requirements, system requirements, "
#         "prerequisite knowledge, accessibility, budget, learning style, time commitment, level of depth, "
#         "preferred topics and format preferences. Write the description as one long paragraph. "
#         "For these courses, make sure to include course prerequisites (course codes), location, "
#         "and department in the constraint categories.:\n\n"
#         f"{content}\n\nDescription:"
#     )
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     descriptions[content] = response.choices[0].message.content
#
# course_site = "https://utmist.gitlab.io/courses/"
# links = extract_links(course_site)
# links = links[:56]
# remove = [15, 28, 29, 33, 34, 48]
# links = [val for idx, val in enumerate(links) if idx not in remove]
#
# for url in links:
#     generate_description_courses(url)
#
# new_data = []
# for key, desc in descriptions.items():
#     new_data.append({
#         'Link': key,
#         'Description': desc
#     })
# new_data = pd.DataFrame(new_data)
#
# # write to CSV as above
# new_data.to_csv("path_to_csv", mode="a", header=False, index=False)
