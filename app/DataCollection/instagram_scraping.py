import instaloader
import logging
import os
import re

logging.basicConfig(level=logging.INFO)

INSTALOADER_INSTANCE = instaloader.Instaloader()
desktop = os.path.expanduser("~\Desktop")
instafilepath = os.path.join(desktop, "instauserpass.txt")
with open(instafilepath, "r") as file:
    lines = file.readlines()
    lines = [line.rstrip("\n") for line in lines]
    INSTAGRAM_USERNAME = lines[0]
    INSTAGRAM_PASSWORD = lines[1]

INSTALOADER_INSTANCE.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)

# Gets all captions in a list from the UTMIST profile
def get_captions() -> list:

    captions = []
    
    profile = instaloader.Profile.from_username(INSTALOADER_INSTANCE.context, "uoft_utmist")

    # Iterating through all posts
    count = 0
    for i, post in enumerate(profile.get_posts()):
        if (count == 30):
            break
        captions.append(post.caption)
        count +=1 
    logging.info(f"Got {len(captions)} captions.")

    return captions

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    return re.sub(emoj, '', data)

unfiltered_instagram_data = get_captions()
instagram_data = []
for i in range(len(unfiltered_instagram_data)):
    instagram_data.append(remove_emojis(unfiltered_instagram_data[i]))

file_path = os.path.join(os.getcwd(), "data\instagram_data.txt")

with open(file_path, 'w', encoding="utf-8") as file:
    for i in range(len(instagram_data)):
        file.write(instagram_data[i] + ' ')

