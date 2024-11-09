import instaloader
import logging

logging.basicConfig(level=logging.INFO)

INSTALOADER_INSTANCE = instaloader.Instaloader()

# NOTE: Logging in with instagram credentials may not be necessary depending on the machine you're on

# May want to replace with environment variables
INSTAGRAM_USERNAME = "Username"
INSTAGRAM_PASSWORD = "Password"

INSTALOADER_INSTANCE.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)

# Gets all captions in a list from the UTMIST profile
def get_captions() -> list:

    captions = []
    
    profile = instaloader.Profile.from_username(INSTALOADER_INSTANCE.context, "uoft_utmist")

    # Iterating through all posts
    for i, post in enumerate(profile.get_posts()):
        captions.append(post.caption)
    logging.info(f"Got {len(captions)} captions.")

    return captions

