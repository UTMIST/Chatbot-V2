import discord
from discord.ext import commands, tasks
# Modified for rag
from custom_query_with_PastChat import classifyRelevance, aiResponse
# from rag_handler import ai_response, save_unanswered_queries, update_vector_database  
import os
from pathlib import Path
from dotenv import load_dotenv

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Modified for rag
intents.reactions = True

client = discord.Client(intents=intents)

# Modified for rag
env_path = Path("..") / ".env"
load_dotenv(dotenv_path=env_path)

# Get target guild and channel IDs from environment variables or hard-code them
TARGET_GUILD_ID = int(os.environ.get("TARGET_GUILD_ID", "1208261679352250458"))
TARGET_CHANNEL_ID = int(os.environ.get("TARGET_CHANNEL_ID", "1243945628124446802"))

@client.event
async def on_ready():
    print("Bot is now online")
    # save_unanswered_queries_task.start()  # Modified for rag
    # update_vector_database_task.start()   # Modified for rag

@client.event
async def on_message(message):
    # Avoid responding to the bot's own messages
    if message.author == client.user:
        return

    # Check if the message is from the target guild and channel
    if message.guild and message.guild.id == TARGET_GUILD_ID and message.channel.id == TARGET_CHANNEL_ID:
        # Add reaction
        if message.content.lower() == 'thanks':
            await message.add_reaction('\U0001F970')

        # Respond
        elif message.content.lower() == "hello":
            await message.channel.send("Welcome to UTMIST!")
            
        else:
            relevance = classifyRelevance(message.content)  # Modified for rag
            print("Relevance:", relevance)                 # Modified for rag
            output = aiResponse(message.content)
            await message.channel.send(output)
    else:
        # Ignore messages not in the target guild and channel
        return

# Addressing edited messages
@client.event
async def on_message_edit(before, after):
    if before.guild and before.guild.id == TARGET_GUILD_ID and before.channel.id == TARGET_CHANNEL_ID:
        await before.channel.send(
            f'{before.author} edited a message.\n'
            f'Before: {before.content}\n'
            f'After: {after.content}'
        )

@client.event
async def on_reaction_add(reaction, user):
    if reaction.message.guild and reaction.message.guild.id == TARGET_GUILD_ID and reaction.message.channel.id == TARGET_CHANNEL_ID:
        await reaction.message.channel.send(f'{user} reacted with {reaction.emoji}')

# # Modified for rag
# @tasks.loop(hours=24)  
# async def save_unanswered_queries_task():
#     await client.wait_until_ready()
#     await save_unanswered_queries()

# # Modified for rag
# @tasks.loop(hours=24)  
# async def update_vector_database_task():
#     await client.wait_until_ready()
#     update_vector_database()

client.run("Your Key")