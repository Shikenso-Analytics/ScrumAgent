import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
import pytz
import requests

from config import scrum_promts
from PIL import Image
from io import BytesIO

import discord
import yaml
from discord import ChannelType
from discord.ext import commands, tasks
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage

from scrumagent.build_agent_graph import build_graph
from scrumagent.data_collector.discord_chat_collector import DiscordChatCollector
from scrumagent.tools.taiga_tool import get_entity_by_ref_tool, get_project
from scrumagent.utils import split_text_smart, init_discord_chroma_db, get_image_description_via_llama

mod_path = Path(__file__).parent

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_THREAD_TYPE = os.getenv("DISCORD_THREAD_TYPE")
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.members = True

with open(mod_path / "../config/taiga_discord_maps.yaml") as f:
    yaml_config = yaml.safe_load(f)
    logger.info(f"Read Config File: {yaml_config}")

    INTERACTABLE_DISCORD_CHANNELS = yaml_config["interactable_discord_channels"]
    TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP = yaml_config["taiga_slag_to_discord_channel_map"]
    DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP = {v: k for k, v in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.items()}

    TAIGA_USER_TO_DISCORD_USER_MAP = yaml_config["taiga_discord_user_map"]

# Initialize the Discord bot
bot = commands.Bot(command_prefix="!!!!", intents=intents)
logger.info("Discord Bot initialized.")

# Initialize the multi-agent graph for /ama requests
multi_agent_graph = build_graph()
logger.info("Multi-agent graph initialized.")

# Draw the graph for visualization purposes (optional)
multi_agent_graph.get_graph(xray=True).draw_mermaid_png(output_file_path="multi_agent_graph.png")

# daily_calculated_openai_cost = 0
summed_up_open_ai_cost = {"undefined": 0}  # per taiga_slug

# Initialize the data collector database
discord_chroma_db = init_discord_chroma_db()

# Initialize the data collectors. Deactivated datacollector for now. Only discord chat collector is active.
discord_chat_collector = DiscordChatCollector(bot, discord_chroma_db, filter_channels=INTERACTABLE_DISCORD_CHANNELS)
data_collector_list = [discord_chat_collector]


def run_agent_in_cb_context(messages: list, config: dict, cost_position=None) -> dict:
    with get_openai_callback() as cb:
        result = multi_agent_graph.invoke(
            {"messages": messages},
            config,
            # debug=True
        )
        logger.info(f"Total Cost (USD): {cb.total_cost}")
        if cost_position:
            if cost_position not in summed_up_open_ai_cost:
                summed_up_open_ai_cost[cost_position] = 0
            summed_up_open_ai_cost[cost_position] += cb.total_cost
        else:
            summed_up_open_ai_cost["undefined"] += cb.total_cost
    return result


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    logger.info(f"Message received from {message.author}: {message.content}")

    if type(message.channel) == discord.DMChannel:
        channel_name = message.author.name
    else:
        channel_name = message.channel.name
        with get_openai_callback() as cb:
            new_msg = discord_chat_collector.add_discord_messages_to_db(message.guild, message.channel, [message])
            summed_up_open_ai_cost["undefined"] += cb.total_cost

    # Config for stateful agents
    config = {"configurable": {"user_id": channel_name, "thread_id": channel_name}}

    # Prepare the question format
    question_format = (
        f"DiscordMsg: {message.content} (From user: {message.author}, channel_name: {channel_name}, "
        f"channel_id: {message.channel.id}, timestamp_sent: {message.created_at.timestamp()})")

    # Add the taiga slug and user story to the question format if the message is not a direct message
    if type(message.channel) != discord.DMChannel:
        taiga_slug = None
        if message.channel.id in DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP:
            taiga_slug = DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP[message.channel.id]
        elif message.channel.parent.id in DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP:
            taiga_slug = DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP[message.channel.parent.id]

        if taiga_slug:
            question_format += f" (Corresponding taiga slug: {taiga_slug})"

        if channel_name.startswith("#"):
            question_format += f" (Corresponding taiga user story id: {channel_name.split(' ')[0][1:]})"


    # Prepare the attachments. Currently only images and text files are supported.
    attachments = message.attachments
    attachments_prepared = []
    for attachment in attachments:
        response = requests.get(attachment.url)

        if response.status_code != 200:
            logger.error(f"Failed to retrieve the file. Status code: {response.status_code}. URL: {attachment.url}")
            continue

        if attachment.content_type.startswith("image"):
            image = Image.open(BytesIO(response.content))  # Open image from response content
            image.save("temp_image.jpg")  # Save the image to a temporary file
            description = get_image_description_via_llama("temp_image.jpg")  # Get the image description
            os.remove("temp_image.jpg")
            attachments_prepared.append(f"Attached Image (Description): {description}")
        #elif attachment.content_type.startswith("audio"):
            # Whisper transcription
        #    pass
        elif attachment.content_type.startswith("text"):
                text_content = response.text  # Get the content as a string
                attachments_prepared.append(f"Attached Textfile: {text_content}")
        else:
            logger.error(f"Unknown attachment type: {attachment.content_type} for {attachment.filename}: {attachment.url}")

    if attachments_prepared:
        question_format += "\n" + "Attachments:"
        question_format += "\n" + "\n".join(attachments_prepared)

    # If the bot is not mentioned in the message, add the question to the state of the multi-agent graph.
    if not bot.user.mentioned_in(message) and type(message.channel) != discord.DMChannel:
        current_messages_state = multi_agent_graph.get_state(config=config).values.get("messages", [])
        current_messages_state.append(HumanMessage(content=question_format))
        multi_agent_graph.update_state(config=config, values={"messages": current_messages_state})
        return

    # Invoke the multi-agent graph with the question.
    # And get the total cost of the conversation.
    # Offload the synchronous, blocking call to an executor.
    logger.info(f"Run Agent with question: {question_format}")
    async with message.channel.typing():
        loop = asyncio.get_running_loop()
        # Use run_in_executor to run the blocking invocation in a separate thread.
        result = await loop.run_in_executor(
            None,
            lambda: run_agent_in_cb_context([HumanMessage(content=question_format)], config)
        )

    str_result = result["messages"][-1].content
    logger.info(f"Result: {str_result}")


    # send multimple messages if the result is too long (2000 char is discord limit)
    str_results_segments = split_text_smart(str_result)
    for segment in str_results_segments:
        await message.reply(segment, suppress_embeds=True)

    # Deactivated for debugging purposes.
    # discord_chat_collector.get_links_from_messages(message.guild, message.channel, [message])
    # discord_chat_collector.get_files_from_messages(message.guild, message.channel, [message])

async def manage_user_story_threads(project_slug: str):
    logger.info("Manage user story threads started.")

    project = get_project(project_slug)
    taiga_thread_channel = bot.get_channel(int(TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP[project_slug]))

    # Get all threads in the channel
    thread_name_to_id = {}
    for thread in taiga_thread_channel.threads:
        thread_name_to_id[thread.name] = thread.id

    async def get_all_archived_threads(channel):
        private = DISCORD_THREAD_TYPE == "private_thread"
        threads = [archived_thread async for archived_thread in channel.archived_threads(private=private, joined=private, limit=100)]
        return threads
    all_archived_threads = await get_all_archived_threads(taiga_thread_channel)
    for thread in all_archived_threads:
        if thread.name not in thread_name_to_id:
            thread_name_to_id[thread.name] = thread.id

    async def manage_user_story(user_story):
        thread_name = f"#{user_story.ref} {user_story.subject}"

        us_full_infos = get_entity_by_ref_tool({"project_slug": project_slug,
                                                "entity_ref": user_story.ref,
                                                "entity_type": "userstory"})
        us_full_infos = json.loads(us_full_infos)

        if thread_name in thread_name_to_id:
            logger.info(f"Skipping '{thread_name}' as it already exists.")
            thread = taiga_thread_channel.get_thread(thread_name_to_id[thread_name])

            tread_pins = await thread.pins()
            if not tread_pins or len(tread_pins) == 0:
                # Pin the first message in the thread. Safety measures, when something went wrong while initializing.
                messages = [message async for message in thread.history(limit=1, oldest_first=True)]
                await messages[0].pin()
        else:
            logger.info(f"Creating thread {thread_name}")
            if DISCORD_THREAD_TYPE == "public_thread":
                thread = await taiga_thread_channel.create_thread(name=thread_name, type=ChannelType.public_thread)
            else:
                thread = await taiga_thread_channel.create_thread(name=thread_name, type=ChannelType.private_thread)
            msg = await thread.send(f"**{thread_name}**:\n"
                                    f"{us_full_infos['description']}\n"
                                    f"{us_full_infos['url']}")
            await msg.pin()


            init_user_story_thread_promt_format = scrum_promts.init_user_story_thread_promt.format(taiga_ref=user_story.ref,
                                                                                      taiga_name=user_story.subject,
                                                                                      project_slug=project_slug)

            config = {"configurable": {"user_id": thread.name, "thread_id": f"{thread.name} thread_init"}}
            async with thread.typing():
                loop = asyncio.get_running_loop()
                # Use run_in_executor to run the blocking invocation in a separate thread.
                result = await loop.run_in_executor(None,
                                                    lambda: run_agent_in_cb_context([
                                                        HumanMessage(content=init_user_story_thread_promt_format)
                                                    ],
                                                        config)
                                                    )

            str_result = result["messages"][-1].content
            logger.info(f"User Story Init Result: {str_result}")

            str_results_segments = split_text_smart(str_result)
            for segment in str_results_segments:
                msg = await thread.send(segment, suppress_embeds=True)
                # await msg.pin()

            thread_name_to_id[thread_name] = thread.id

        associated_users = [w["id"] for w in us_full_infos["watchers"]]
        if us_full_infos["assigned_to"]:
            associated_users += [us_full_infos["assigned_to"]["id"]]

        # associated_users += [20]

        for task in us_full_infos["related"]["tasks"]:
            if task.get("assigned_to"):
                associated_users += [task["assigned_to"]]
            if task.get("watchers"):
                associated_users += task["watchers"]

        associated_users = list(set(associated_users))
        logger.info(f"Adding the following associated users: {associated_users}")
        for user in associated_users:
            if user in TAIGA_USER_TO_DISCORD_USER_MAP:
                discord_user_name = TAIGA_USER_TO_DISCORD_USER_MAP[user]
                discord_user = discord.utils.get(taiga_thread_channel.members, name=discord_user_name)

                # TODO: For some reason, thread.members is empty. Need the check bot permissions
                if discord_user not in thread.members:
                    await thread.add_user(discord_user)
                    await asyncio.sleep(0.5) # Sleep for 0.5 second to avoid rate limiting

    if project.is_backlog_activated:
        sprints = project.list_milestones(closed=False)
        for sprint in sprints:
            for user_story in sprint.user_stories:
                if not user_story.is_closed:
                    await manage_user_story(user_story)
    else:
        for user_story in project.list_user_stories():
            if not user_story.is_closed and not user_story.status_extra_info.get("is_closed"):
                await manage_user_story(user_story)


@bot.event
async def on_guild_join():
    logger.info(f"Guild join: {bot.user} (ID: {bot.user.id})")
    await discord_chat_collector.check_all_unread_massages()


@bot.event
async def on_guild_remove():
    # Do nothing for now. Delete every msg from the guild from the DB??
    logger.info(f"Guild remove: {bot.user} (ID: {bot.user.id})")


@bot.event
async def on_guild_update():
    # DO nothing for now. Update the guild info in the DB??
    logger.info(f"Guild update: {bot.user} (ID: {bot.user.id})")


@tasks.loop(hours=1)
async def update_taiga_threads():
    logger.info("Updating taiga threads started.")
    for project_slug in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.keys():
        await manage_user_story_threads(project_slug)


@tasks.loop(hours=24)
async def daily_datacollector_task():
    logger.info("Daily data collector started.")
    # await blog_txt_collector.check_all_files_in_folder()


"""
@tasks.loop(time=datetime.time(hour=6, minute=0))
async def output_total_open_ai_cost():
    for taiga_slug, taiga_channel_id in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.items():
        taiga_thread_channel = bot.get_channel(int(taiga_channel_id))

        msg = f"To total cost of yesterdays OpenAI usage was: {summed_up_open_ai_cost['undefined']}"
        await taiga_thread_channel.send(msg)

        summed_up_open_ai_cost["undefined"] = 0
"""


@tasks.loop(time=datetime.time(hour=8, minute=0, tzinfo=pytz.timezone('Europe/Berlin')))
async def scrum_master_task():
    # Only run on weekdays
    if datetime.datetime.today().weekday() > 4:
        return

    logger.info("Scrum master started.")
    for project_slug in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.keys():
        await manage_user_story_threads(project_slug)


        taiga_thread_channel = bot.get_channel(TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP[project_slug])
        project = get_project(project_slug)
        for thread in taiga_thread_channel.threads:
            logger.info(f"Running Scrummaster for {thread.name}")
            taiga_ref, taiga_name = thread.name.split(" ", 1)
            taiga_ref = taiga_ref.replace("#", "")
            userstory = project.get_userstory_by_ref(taiga_ref)
            if userstory.is_closed or userstory.status_extra_info.get("is_closed", False):
                continue

            scrum_task_promt = scrum_promts.scrum_master_promt.format(taiga_ref=taiga_ref, taiga_name=taiga_name,
                                                         project_slug=project_slug)
            config = {"configurable": {"user_id": thread.name, "thread_id": f"{thread.name} scrum_master"}}

            logger.info(f"Scrum master promt: {scrum_task_promt}")
            async with thread.typing():
                loop = asyncio.get_running_loop()
                # Use run_in_executor to run the blocking invocation in a separate thread.
                result = await loop.run_in_executor(None,
                                                    lambda: run_agent_in_cb_context([
                                                        HumanMessage(content=scrum_task_promt)
                                                    ],
                                                        config)
                                                    )

            str_result = result["messages"][-1].content
            logger.info(f"Scrum master result: {str_result}")

            str_results_segments = split_text_smart(str_result)
            for segment in str_results_segments:
                await thread.send(segment, suppress_embeds=True)


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    for assistant in data_collector_list:
        await assistant.on_startup()
    logger.info("Bots are ready!")

    # Get all user_stories of active sprints
    for project_slug in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.keys():
        await manage_user_story_threads(project_slug)

    await bot.tree.sync()
    logger.info("Bot command tree synced!")

    scrum_master_task.start()
    daily_datacollector_task.start()
    update_taiga_threads.start()
    # await scrum_master_task()


if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
