# This code takes the AI functions and incorporates into a discord Bot 
import discord
from discord.ext import commands, tasks
#modified for rag
from custom_query_with_PastChat import classifyRelevance, aiResponse
# from rag_handler import ai_response, save_unanswered_queries, update_vector_database  
import os
from pathlib import Path
from dotenv import load_dotenv

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True #modified for rag
intents.reactions = True

client = discord.Client(intents=intents)

#modified for rag
env_path = Path("..") / ".env"
load_dotenv(dotenv_path=env_path)

@client.event
async def on_ready():
    print("Bot is now online")
    # save_unanswered_queries_task.start()  #modified for rag
    # update_vector_database_task.start()   #modified for rag

@client.event
async def on_message(message):
    #avoid same message between user and the bot
    if message.author == client.user:
        return

    #Check if the message is from the target guild
    #add reaction
    if message.content == 'thanks':
        await message.add_reaction('\U0001F970')

    #respond
    elif message.content == "hello":
        await message.channel.send("Welcome to UTMIST!")
        
    else:
        relevance = classifyRelevance(message.content)  #modified for rag
        print("relevance: ", relevance)                 #modified for rag
        output = aiResponse(message.content)
        await message.channel.send(output)

#addressing edited messages
@client.event
async def on_message_edit(before, after):
    await before.channel.send(
        f'{before.author} edit a message.\n'
        f'Before: {before.content}\n'
        f'After:{after.content}'
    )


@client.event
async def on_reaction_add(reaction, user):
    await reaction.message.channel.send(f'{user} reacted with {reaction.emoji}')

# #modified for rag
# @tasks.loop(hours=24)  
# async def save_unanswered_queries_task():
#     await client.wait_until_ready()
#     await save_unanswered_queries()

# #modified for rag
# @tasks.loop(hours=24)  
# async def update_vector_database_task():
#     await client.wait_until_ready()
#     update_vector_database()


#client.run(os.environ.get("DISCORD_BOT_TOKEN")) #modified for rag
client.run("Your Key")