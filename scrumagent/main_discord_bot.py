import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Any, List, Optional

import discord
import httpx
import pytz
import taiga
import yaml
from discord import ChannelType, Message
from discord.ext import commands, tasks
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from langchain_taiga.tools.taiga_tools import get_entity_by_ref_tool, get_project
from taiga.models import UserStory

from config import scrum_promts
from scrumagent import util_logging
from scrumagent.build_agent_graph import build_graph
from scrumagent.data_collector.discord_chat_collector import DiscordChatCollector
from scrumagent.utils import split_text_smart, init_discord_chroma_db

mod_path = Path(__file__).parent

load_dotenv()

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

    INTERACTABLE_DISCORD_CHANNELS = yaml_config["interactable_discord_channels"]
    TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP = yaml_config["taiga_slag_to_discord_channel_map"]

    DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP = {
        v: k for k, v in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.items()
    }
    if "other_discord_channel_to_taiga_slag_map" in yaml_config:
        other_discord_channel_to_taiga_slag_map = yaml_config[
            "other_discord_channel_to_taiga_slag_map"
        ]
    else:
        other_discord_channel_to_taiga_slag_map = {}
    DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP.update(other_discord_channel_to_taiga_slag_map)

    TAIGA_USER_TO_DISCORD_USER_MAP = yaml_config["taiga_discord_user_map"]

    DISCORD_LOG_CHANNEL = yaml_config["discord_log_channels"]

# Initialize the Discord bot
bot = commands.Bot(command_prefix="!!!!", intents=intents)

logger = util_logging.init_module_logger(__name__)
listener = util_logging.start_listener()

print("Discord Bot initialized.")

# Initialize the multi-agent graph for /ama requests
multi_agent_graph = build_graph()
print("Multi-agent graph initialized.")

# Draw the graph for visualization purposes (optional)
# multi_agent_graph.get_graph(xray=True).draw_mermaid_png(output_file_path="multi_agent_graph.png")

# daily_calculated_openai_cost = 0
summed_up_open_ai_cost = {"undefined": 0}  # per taiga_slug

# Initialize the data collector database
discord_chroma_db = init_discord_chroma_db()

# Initialize the data collectors. Deactivated datacollector for now. Only discord chat collector is active.
discord_chat_collector = DiscordChatCollector(
    bot, discord_chroma_db, filter_channels=INTERACTABLE_DISCORD_CHANNELS
)
data_collector_list = [discord_chat_collector]


# https://python.langchain.com/docs/how_to/trim_messages/#trimming-based-on-message-count


@util_logging.exception(__name__)
def run_agent_in_cb_context(
        messages: List[HumanMessage],
        config: dict,
        cost_position: Optional[str] = None,
) -> dict:
    """Run the agent graph and track token costs."""
    with get_openai_callback() as cb:
        result = multi_agent_graph.invoke(
            {"messages": messages},
            config,
            # debug=True
        )

        if cost_position:
            if cost_position not in summed_up_open_ai_cost:
                summed_up_open_ai_cost[cost_position] = 0
            summed_up_open_ai_cost[cost_position] += cb.total_cost
        else:
            summed_up_open_ai_cost["undefined"] += cb.total_cost
    return result


async def run_agent_async(
        typing_channel: discord.abc.Messageable,
        messages: List[HumanMessage],
        config: dict,
        cost_position: Optional[str] = None,
) -> str:
    """Run the agent graph in an executor while showing a typing indicator."""
    async with typing_channel.typing():
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_agent_in_cb_context(messages, config, cost_position),
        )
    return result["messages"][-1].content


def prepare_attachments(attachments: List[discord.Attachment]) -> List[str]:
    """Return string descriptions for Discord attachments."""
    prepared = []
    for attachment in attachments:
        response = httpx.get(attachment.url)
        if response.status_code != 200:
            print(
                f"Failed to retrieve the file. Status code: {response.status_code}. URL: {attachment.url}"
            )
            continue
        prepared.append(
            f"Attached File: {attachment.filename} (Type: {attachment.content_type}) - {attachment.url}"
        )
    return prepared


async def send_split_message(
        destination: discord.abc.Messageable, text: str, *, reply: bool = False
) -> None:
    """Send ``text`` split into Discord-sized segments."""
    for segment in split_text_smart(text):
        if reply and isinstance(destination, discord.Message):
            await destination.reply(segment, suppress_embeds=True)
        else:
            await destination.send(segment, suppress_embeds=True)


def build_question_format(message: discord.Message, channel_name: str) -> str:
    """Return the formatted question string for ``message``."""
    q = (
        f"DiscordMsg: {message.content} (From user: {message.author}, channel_name: {channel_name}, "
        f"channel_id: {message.channel.id}, timestamp_sent: {message.created_at.timestamp()})"
    )

    if type(message.channel) != discord.DMChannel:
        taiga_slug = None
        if message.channel.id in DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP:
            taiga_slug = DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP[message.channel.id]
        elif getattr(message.channel, "parent", None) is not None and (
                message.channel.parent.id in DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP
        ):
            taiga_slug = DISCORD_CHANNEL_TO_TAIGA_SLAG_MAP[message.channel.parent.id]

        if taiga_slug:
            q += f" (Corresponding taiga slug: {taiga_slug})"

        if channel_name.startswith("#"):
            q += f" (Corresponding taiga user story id: {channel_name.split(' ')[0][1:]})"
    return q


async def add_users_to_thread(
        discord_thread: discord.Thread,
        us_info: dict,
        guild_channel: discord.abc.GuildChannel,
) -> None:
    """Invite associated Taiga users to the Discord thread."""
    associated_users = [w["id"] for w in us_info["watchers"]]
    if us_info["assigned_to"]:
        associated_users.append(us_info["assigned_to"]["id"])

    for task in us_info["related"]["tasks"]:
        if task.get("assigned_to"):
            associated_users.append(task["assigned_to"])
        if task.get("watchers"):
            associated_users.extend(task["watchers"])

    associated_users = list(set(associated_users))
    for user in associated_users:
        if user not in TAIGA_USER_TO_DISCORD_USER_MAP:
            continue
        discord_user_name = TAIGA_USER_TO_DISCORD_USER_MAP[user]
        discord_user = discord.utils.get(guild_channel.members, name=discord_user_name)
        if discord_user and discord_user not in discord_thread.members:
            await discord_thread.add_user(discord_user)
            await asyncio.sleep(0.5)


async def ensure_user_story_thread(
        user_story: UserStory,
        project_slug: str,
        taiga_thread_channel: discord.abc.GuildChannel,
        thread_map: dict,
) -> None:
    """Create or update a Discord thread for the given user story."""
    thread_name = f"#{user_story.ref} {user_story.subject}"
    us_full_infos = json.loads(
        get_entity_by_ref_tool(
            {
                "project_slug": project_slug,
                "entity_ref": user_story.ref,
                "entity_type": "userstory",
            }
        )
    )

    if thread_name in thread_map:
        discord_thread = thread_map[thread_name]
        # Check if the thread is archived and unarchive it if necessary
        if discord_thread.archived:
            await discord_thread.edit(archived=False)
        pins = await discord_thread.pins()
        if not pins:
            messages: List[Message] = [
                m async for m in discord_thread.history(limit=1, oldest_first=True)
            ]
            # If there are no pinned messages, pin the first message if not system message
            if messages and not messages[0].is_system():
                await messages[0].pin()
    else:
        if DISCORD_THREAD_TYPE == "public_thread":
            discord_thread = await taiga_thread_channel.create_thread(
                name=thread_name,
                type=ChannelType.public_thread,
                auto_archive_duration=4320,
            )
        else:
            discord_thread = await taiga_thread_channel.create_thread(
                name=thread_name,
                type=ChannelType.private_thread,
                auto_archive_duration=4320,
            )
        header = f"**{thread_name}**:\n"
        body = f"{us_full_infos['description']}\n{us_full_infos['url']}"

        # 1) erstes Segment senden & pinnen
        segments = list(split_text_smart(header + body, max_length=3500))
        first_msg = await discord_thread.send(segments[0])
        await first_msg.pin()

        init_prompt = scrum_promts.init_user_story_thread_promt.format(
            taiga_ref=user_story.ref,
            taiga_name=user_story.subject,
            project_slug=project_slug,
        )
        config = {
            "configurable": {
                "user_id": discord_thread.name,
                "thread_id": f"{discord_thread.name} thread_init",
            }
        }
        result_text = await run_agent_async(
            discord_thread, [HumanMessage(content=init_prompt)], config
        )
        await send_split_message(discord_thread, result_text)

        thread_map[thread_name] = discord_thread

    await add_users_to_thread(discord_thread, us_full_infos, taiga_thread_channel)


@bot.event
@util_logging.exception(__name__)
async def on_message(message: discord.Message) -> None:
    """Handle incoming Discord messages."""
    if message.author == bot.user:
        return

    print(
        f"Message ({type(message.channel)}) received from {message.author}: {message.content}"
    )
    if type(message.channel) == discord.DMChannel:
        channel_name = message.author.name
    else:
        channel_name = message.channel.name
        with get_openai_callback() as cb:
            discord_chat_collector.add_discord_messages_to_db(
                message.guild, message.channel, [message]
            )
            summed_up_open_ai_cost["undefined"] += cb.total_cost

    # Config for stateful agents
    config = {"configurable": {"user_id": channel_name, "thread_id": channel_name}}

    # Prepare the question format
    question_format = build_question_format(message, channel_name)

    # Prepare the attachments. Currently only images and text files are supported.
    attachments_prepared = prepare_attachments(message.attachments)

    if attachments_prepared:
        question_format += "\nAttachments:\n" + "\n".join(attachments_prepared)

    # Determine if the bot was explicitly mentioned in the message. ``User.mentioned_in``
    # also returns ``True`` for ``@here`` or ``@everyone`` mentions, which should not
    # trigger the bot. Therefore we explicitly check if the bot user is in the
    # ``message.mentions`` list.
    bot_explicitly_mentioned = bot.user in message.mentions

    # If the bot is not explicitly mentioned in the message and it is not a DM,
    # just add the question to the state of the multi-agent graph without
    # generating a reply.
    if (
            not bot_explicitly_mentioned
            and type(message.channel) != discord.DMChannel
    ):
        print(f"Add question to state: {question_format}")
        # I don't think it is needed to update the state manuel with alle msg before the question.
        # https://python.langchain.com/docs/how_to/message_history/
        # Check

        # current_messages_state = multi_agent_graph.get_state(config=config).values.get("messages", [])
        # current_messages_state.append(HumanMessage(content=question_format))
        # multi_agent_graph.update_state(config=config, values={"messages": current_messages_state})

        multi_agent_graph.update_state(
            config=config, values={"messages": HumanMessage(content=question_format)}
        )

        return

    # Invoke the multi-agent graph with the question.
    # And get the total cost of the conversation.
    # Offload the synchronous, blocking call to an executor.
    print(f"Run Agent with question: {question_format}")
    str_result = await run_agent_async(
        message.channel, [HumanMessage(content=question_format)], config
    )
    print(f"Result: {str_result}")

    # send multimple messages if the result is too long (2000 char is discord limit)
    await send_split_message(message, str_result, reply=True)

    # Deactivated for debugging purposes.
    # discord_chat_collector.get_links_from_messages(message.guild, message.channel, [message])
    # discord_chat_collector.get_files_from_messages(message.guild, message.channel, [message])


@util_logging.exception(__name__)
async def manage_user_story_threads(project_slug: str) -> None:
    """Ensure that each Taiga user story has a corresponding Discord thread."""
    print("Manage user story threads started.")

    project = get_project(project_slug)
    if not project:
        print(f"Project '{project_slug}' not found: {project}")
        return

    taiga_thread_channel = bot.get_channel(
        int(TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP[project_slug])
    )
    thread_map = await get_discord_thread_map(taiga_thread_channel)

    for us in get_active_user_stories(project):
        await ensure_user_story_thread(
            us, project_slug, taiga_thread_channel, thread_map
        )


@bot.event
@util_logging.exception(__name__)
async def on_guild_join() -> None:
    """Called when the bot joins a guild."""
    print(f"Guild join: {bot.user} (ID: {bot.user.id})")
    await discord_chat_collector.check_all_unread_messages()


@bot.event
@util_logging.exception(__name__)
async def on_guild_remove() -> None:
    # Do nothing for now. Delete every msg from the guild from the DB??
    print(f"Guild remove: {bot.user} (ID: {bot.user.id})")
    pass


@bot.event
@util_logging.exception(__name__)
async def on_guild_update() -> None:
    # DO nothing for now. Update the guild info in the DB??
    print(f"Guild update: {bot.user} (ID: {bot.user.id})")
    pass


@tasks.loop(hours=1)
@util_logging.exception(__name__)
async def update_taiga_threads() -> None:
    """Periodic task that syncs Taiga user stories with Discord threads."""
    print("Updating taiga threads started.")
    for project_slug in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.keys():
        await manage_user_story_threads(project_slug)


@tasks.loop(hours=24)
@util_logging.exception(__name__)
async def daily_datacollector_task() -> None:
    """Run daily housekeeping routines for data collection."""
    print("Daily data collector started.")
    # await blog_txt_collector.check_all_files_in_folder()
    pass


"""
@tasks.loop(time=datetime.time(hour=6, minute=0))
async def output_total_open_ai_cost():
    for taiga_slug, taiga_channel_id in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.items():
        taiga_thread_channel = bot.get_channel(int(taiga_channel_id))

        msg = f"To total cost of yesterdays OpenAI usage was: {summed_up_open_ai_cost['undefined']}"
        await taiga_thread_channel.send(msg)

        summed_up_open_ai_cost["undefined"] = 0
"""

BERLIN_TZ = pytz.timezone("Europe/Berlin")


def extract_tags(us: Any) -> set[str]:
    """Return all tag names of a Taiga user story in lowercase.

    Args:
        us (dict | taiga.models.UserStory): The user story object or its JSON representation.

    Returns:
        set[str]: All tags associated with the user story.
    """
    if isinstance(us, dict):  # JSON from the tool
        return {t[0].lower() for t in us.get("tags", []) if t}
    if isinstance(us, UserStory):
        return {t[0].lower() for t in us.tags if t}
    # Python client object
    return {t.lower() for t in us.tags}


def standup_due_today(us: Any) -> bool:
    """Return ``True`` if a stand‑up should be sent today.

    The decision is based on the tags attached to the Taiga user
    story:

    - ``"daily stand-up"``    – a message is posted every day, including
      weekends.
    - ``"weekly stand-up"``   – only post on Mondays.
    - no tag                  – default behaviour; post Monday through
      Friday.

    Args:
        us (dict | taiga.models.UserStory): The user story object or its
            JSON representation.

    Returns:
        bool: ``True`` if a stand‑up should be posted today, otherwise
        ``False``.
    """
    tags = extract_tags(us)
    today_idx = datetime.datetime.now(BERLIN_TZ).weekday()  # 0 = Monday
    if "daily stand-up" in tags:
        return True
    if "weekly stand-up" in tags:
        return today_idx == 0
    return today_idx < 5


def get_active_user_stories(project: Any) -> List[UserStory]:
    """Return all active user stories for a Taiga project."""
    if project.is_backlog_activated:
        sprints = [
            m
            for m in project.list_milestones(closed=False)
            if not getattr(m, "is_closed", False) and getattr(m, "is_active", True)
        ]
        user_stories = [
            us for sprint in sprints for us in sprint.user_stories if not us.is_closed
        ]
    else:
        status_map = {
            status.id: status.name.lower()
            for status in project.list_user_story_statuses()
        }
        user_stories = []
        for us in project.list_user_stories():
            status_id = (
                us.status.get("id")
                if isinstance(us.status, dict)
                else getattr(us.status, "id", us.status)
            )
            status_name = status_map.get(status_id, "")
            if (
                    not us.is_closed
                    and not us.status_extra_info.get("is_closed")
                    and status_name in ["ready", "in progress", "ready for test"]
            ):
                user_stories.append(us)
    return user_stories


async def get_discord_thread_map(
        channel: discord.abc.GuildChannel,
) -> dict[str, discord.Thread]:
    """Return mapping of thread names to Discord thread objects, including archived threads."""
    thread_map = {t.name: t for t in channel.threads}

    async def collect(private: bool) -> List[discord.Thread]:
        return [
            t
            async for t in channel.archived_threads(
                private=private, joined=private, limit=100
            )
        ]

    archived = await collect(True)
    archived += await collect(False)

    for t in archived:
        thread_map.setdefault(t.name, t)
    return thread_map


@tasks.loop(time=datetime.time(hour=8, minute=0, tzinfo=pytz.timezone("Europe/Berlin")))
@util_logging.exception(__name__)
async def scrum_master_task() -> None:
    """Daily task that posts stand-up messages."""
    print(f"Scrum master task started at {datetime.datetime.now()}")
    for project_slug in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.keys():
        await manage_user_story_threads(project_slug)

        taiga_thread_channel = bot.get_channel(
            TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP[project_slug]
        )
        project = get_project(project_slug)
        thread_map = await get_discord_thread_map(taiga_thread_channel)

        for us in get_active_user_stories(project):
            thread_name = f"#{us.ref} {us.subject}"
            thread = thread_map.get(thread_name)
            if not thread:
                print(f"Discord thread for {thread_name} not found.")
                continue

            if not standup_due_today(us):
                print(f"Skip stand‑up for {thread_name} (tags rule)")
                continue

            print(f"Running Scrummaster for {thread_name}")

            scrum_task_promt = scrum_promts.scrum_master_promt.format(
                taiga_ref=us.ref, taiga_name=us.subject, project_slug=project_slug
            )
            config = {
                "configurable": {
                    "user_id": thread.name,
                    "thread_id": f"{thread.name} scrum_master",
                }
            }

            str_result = await run_agent_async(
                thread, [HumanMessage(content=scrum_task_promt)], config
            )
            print(f"Scrum master result: {str_result}")
            await send_split_message(thread, str_result)


@bot.event
@util_logging.exception(__name__)
async def on_ready() -> None:
    """Called when the Discord bot is fully ready."""
    channel_list = [bot.get_channel(x) for x in DISCORD_LOG_CHANNEL]
    util_logging.override_defaults(override=channel_list)
    # discord_log_worker.start()

    print(f"Logged in as {bot.user} (ID: {bot.user.id})")

    for assistant in data_collector_list:
        await assistant.on_startup()

    # Runs with start later
    # Get all user_stories of active sprints
    for project_slug in TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.keys():
        await manage_user_story_threads(project_slug)

    await bot.tree.sync()

    scrum_master_task.start()
    daily_datacollector_task.start()
    update_taiga_threads.start()
    print(f"Tasks started.")


@tasks.loop(seconds=10)
async def discord_log_worker() -> None:
    """Forward log records from the queue to Discord."""
    try:
        subject, rec, discord_channels = util_logging.discord_log_queue.get_nowait()
    except util_logging.queue.Empty:
        return

    print(f"Captured log message: {subject}: {rec}")

    for ch in discord_channels:
        await ch.send(f"**{subject}**:\n\n" f"{rec}")


if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
