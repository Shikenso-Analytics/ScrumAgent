import json
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

from scrumagent.utils import init_discord_chroma_db
from scrumagent.tools.timeframe_parser_tool import interpret_timeframe_tool

load_dotenv()

_chroma_db_inst = None


def get_chroma_db():
    """Return a lazily initialized Chroma DB instance."""
    global _chroma_db_inst
    if _chroma_db_inst is None:
        _chroma_db_inst = init_discord_chroma_db()
    return _chroma_db_inst

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")


@tool(parse_docstring=True)
def discord_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search for Discord messages that are semantically similar to the given query.

    This tool uses the Chroma database to find relevant posts from the Discord server
    based on similarity to the provided query. The returned string includes:
      - The message content (with newlines replaced by spaces)
      - The author's name
      - The channel name where the message was posted
      - A human-readable timestamp

    Use this tool when you need to find messages that match a certain query or topic.

    Args:
        query (str): The search query to use for finding similar Discord messages.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        str: A formatted summary of the matched messages or a notice if no relevant results were found.
    """
    chroma_db = get_chroma_db()
    results = chroma_db.similarity_search(query, k=max_results)
    str_format = ""
    for result in results:
        content = result.page_content.replace("\n", " ")
        timestamp_format = datetime.fromtimestamp(result.metadata['timestamp'])

        str_format += (
            f"{content} (User: {result.metadata['author_name']}, "
            f"Channel: {result.metadata['channel_name']}, "
            f"Timestamp: {timestamp_format})\n"
        )
    if len(str_format) == 0:
        str_format = "No good Discord Chat Result was found"
    return str_format


@tool(parse_docstring=True)
def discord_channel_msgs_tool(channel_name: str = None, timeframe: str = None) -> str:
    """
    Use this tool to retrieve historical Discord messages from a specific channel or thread,
    optionally filtering by a time range.

    Example usage:
        discord_channel_msgs_tool(
            channel_name="#1234 My Channel",
            timeframe="yesterday"
        )

    Args:
        channel_name (str): The name of the Discord channel to search in. Threads start with #. Does not have to be URL encoded.
        timeframe (str, optional): A time range to filter messages, such as "yesterday" or "last week".

    Returns:
        str: A formatted string of matching messages, including user, channel, and timestamp.

    When to call:
        - If the user requests something like "show me the messages from #my-channel yesterday".
        - If you need to gather conversation context from a certain channel/time.
    """
    where_filter = [{"source": "discord_chat"}]
    if channel_name:
        where_filter.append({"channel_name": channel_name})

    timeframes = json.loads(interpret_timeframe_tool(timeframe))
    before = timeframes.get("before")
    after = timeframes.get("after")

    # https://docs.trychroma.com/docs/querying-collections/metadata-filtering
    if before:
        where_filter.append({"timestamp": {"$lte": before}})  # less than or equal
    if after:
        where_filter.append({"timestamp": {"$gte": after}})  # greater than or equal

    chroma_db = get_chroma_db()
    results_dict = chroma_db.get(where={"$and": where_filter})
    str_format = ""
    # json_results = []
    for content, metadata in zip(results_dict["documents"], results_dict["metadatas"]):
        timestamp_format = datetime.fromtimestamp(metadata["timestamp"])

        # str_format += f"User: {metadata['author_name']} said at {timestamp_format}: {content}\n"
        content = content.replace("\n", " ")
        str_format += f"{content} (User: {metadata['author_name']}, Channel: {metadata['channel_name']}, Timestamp: {timestamp_format})\n"
        # json_results.append({"content": content, "meta_data": metadata})

    if len(str_format) == 0:
        str_format = "No good Discord Chat Result was found"
    return str_format


@tool(parse_docstring=True)
def discord_get_recent_messages_tool(channel_id: str, limit: int = 200) -> str:
    """
    Decepricated
    Use this tool to fetch the most recent messages from a specific Discord channel or thread via the Discord API.

    Example usage:
        discord_get_recent_messages_tool(
            channel_id="123456789012345678",
            limit=200
        )

    Args:
        channel_id (str): The unique ID of the Discord channel or thread to retrieve messages from.
        limit (int, optional): Maximum number of recent messages to retrieve. Defaults to 200.

    Returns:
        str: A formatted string of the most recent messages, including content, user, and timestamp.

    When to call:
        - If the user asks for the latest updates or recent conversation details.
        - When you need to quickly obtain the latest messages for context.

    Parameter Suggestions:
        - if the channel name is known but not the ID, consider calling discord_list_channels_with_threads_tool.
    """
    if not DISCORD_TOKEN:
        return "Error: DISCORD_BOT_TOKEN is not set in environment variables."

    # The Discord API endpoint for fetching channel messages:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={limit}"

    headers = {
        "Authorization": f"Bot {DISCORD_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()  # Will raise an HTTPError if non-2xx status
    except httpx.HTTPError as e:
        return f"Error fetching messages from Discord API: {str(e)}"

    messages = response.json()

    # If the response is not an array of messages, handle error:
    if not isinstance(messages, list):
        return f"Unexpected response from Discord: {messages}"

    if len(messages) == 0:
        return "No recent messages found."

    # Build a formatted string of messages
    formatted_output = ""
    # Discord returns messages in reverse chronological order by default (newest first).
    # If you prefer oldest first, you can reverse them: messages = reversed(messages)
    for msg in messages:
        content = msg.get("content", "").replace("\n", " ")
        author = msg.get("author", {}).get("username", "Unknown User")
        # Convert Discord's ISO8601 timestamp to local datetime
        # The 'timestamp' field is in ISO8601 (e.g., "2023-01-29T20:31:10.527000+00:00")
        msg_timestamp = msg.get("timestamp")
        dt_str = msg_timestamp
        if msg_timestamp:
            try:
                # Attempt to parse the ISO8601 string
                dt_obj = datetime.fromisoformat(msg_timestamp.replace("Z", "+00:00"))
                dt_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        formatted_output += (
            f"{content} "
            f"(User: {author}, Timestamp: {dt_str})\n"
        )

    return formatted_output


@tool(parse_docstring=True)
def discord_send_message_tool(channel_id: str, message: str) -> str:
    """
    Use this tool to send a message to a specified Discord channel or thread via the Discord API.

    Example usage:
        discord_send_message_tool(
            channel_id="123456789012345678",
            message="Hello, world!"
        )

    Args:
        channel_id (str): The unique ID of the Discord channel or thread where the message will be sent.
        message (str): The text content of the message to send.

    Returns:
        str: A confirmation message including the sent message content and message ID, or an error message if sending fails.

    When to call:
        - If the user instructs the bot to post a message (e.g., "send this to #general").
        - When programmatically triggering a notification or update in a Discord channel.

    Parameter Suggestions:
        - If either "channel_id" or "message" is missing, prompt the user to provide the missing information.
          For instance, if "message" is missing, ask: "What message should I send?".
        - If the channel name is known but not the ID, consider calling discord_list_channels_with_threads_tool.
    """
    if not DISCORD_TOKEN:
        return "Error: DISCORD_BOT_TOKEN is not set in environment variables."

    # The Discord API endpoint for sending messages:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"

    headers = {
        "Authorization": f"Bot {DISCORD_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {"content": message}

    try:
        response = httpx.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if a non-2xx status is returned
    except httpx.HTTPError as e:
        return f"Error sending message via Discord API: {str(e)}"

    sent_message = response.json()

    # Optionally, you can format the response or extract specific fields.
    content_sent = sent_message.get("content", "")
    message_id = sent_message.get("id", "N/A")
    return f"Message sent successfully: '{content_sent}' (ID: {message_id})"


@tool(parse_docstring=True)
def discord_list_channels_with_threads_tool(guild_id: str) -> str:
    """
    List all channels in a Discord guild, embedding active threads within their respective channel.

    This tool fetches all channels from a guild and assumes that every channel is writable.
    It then fetches active threads in the guild and nests each thread under its parent channel
    in the resulting JSON structure.

    Args:
        guild_id (str): The ID of the Discord guild (server) to fetch channels and threads from.

    Returns:
        str: A JSON-formatted string representing a list of channels. Each channel contains:
             - id: The channel ID.
             - name: The channel name.
             - threads: A list of active threads (each with its id and name) that belong to the channel.
             In case of an error, an error message is returned as a JSON object.
    """
    if not DISCORD_TOKEN:
        return json.dumps({"error": "DISCORD_BOT_TOKEN is not set in environment variables."})

    headers = {
        "Authorization": f"Bot {DISCORD_TOKEN}",
        "Content-Type": "application/json"
    }

    # --- Fetch all guild channels ---
    channels_url = f"https://discord.com/api/v10/guilds/{guild_id}/channels"
    try:
        channels_response = httpx.get(channels_url, headers=headers)
        channels_response.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": f"Error fetching channels from Discord API: {str(e)}"})

    channels_data = channels_response.json()
    if not isinstance(channels_data, list):
        return json.dumps({"error": f"Unexpected response from Discord channels: {channels_data}"})

    # Build a list of channel objects and a lookup dictionary by channel ID
    channels_list = []
    channels_dict = {}
    for channel in channels_data:
        channel_obj = {
            "id": channel.get("id"),
            "name": channel.get("name"),
            "threads": []  # Placeholder for threads belonging to this channel
        }
        channels_list.append(channel_obj)
        channels_dict[channel.get("id")] = channel_obj

    # --- Fetch active threads in the guild ---
    threads_url = f"https://discord.com/api/v10/guilds/{guild_id}/threads/active"
    try:
        threads_response = httpx.get(threads_url, headers=headers)
        threads_response.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": f"Error fetching active threads from Discord API: {str(e)}"})

    threads_data = threads_response.json()
    active_threads = threads_data.get("threads", [])

    # Place each active thread under its parent channel in our channels_dict
    for thread in active_threads:
        parent_id = thread.get("parent_id")
        thread_obj = {
            "id": thread.get("id"),
            "name": thread.get("name")
        }
        if parent_id and parent_id in channels_dict:
            channels_dict[parent_id]["threads"].append(thread_obj)
        else:
            # If the parent channel is not found, the thread is ignored.
            # Alternatively, you might choose to add it to an "unknown" category.
            pass

    return json.dumps(channels_list)
