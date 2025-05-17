import datetime
import itertools
import os
import re
from typing import Tuple, List

import discord
from discord import Thread
from langchain_chroma import Chroma

from .base_collector import BaseCollector
from scrumagent import util_logging


# Regular expression to match URLs
URL_REGEX = re.compile(r'https?://\S+')


logger = util_logging.init_module_logger(__name__)
listener = util_logging.start_listener()


class DiscordChatCollector(BaseCollector):
    """Collect Discord messages, files and links into a Chroma database."""

    DB_IDENTIFIER = "discord_chat"
    # Filter out new_member and chat_input_command messages. Add more if needed
    FILTERED_MSG_TYPES = [discord.MessageType.new_member, discord.MessageType.chat_input_command]

    def __init__(self, bot: discord.Client, chroma_db: Chroma, filter_channels: [str] = None):
        """Create the collector.

        Args:
            bot (discord.Client): Discord client instance.
            chroma_db (Chroma): Database handle.
            filter_channels (list[str] | None): Optional list of channel names to limit collection.
        """

        super().__init__(bot, chroma_db)
        self.filter_channels = filter_channels

    @util_logging.exception(__name__)
    async def on_startup(self):
        """Collect unread messages when the bot starts."""

        await self.check_all_unread_massages()

    @util_logging.exception(__name__)
    async def check_all_unread_massages(self):
        """Iterate over all channels and collect new messages."""

        for guild in self.bot.guilds:
            channel_que = [guild.channels, guild.threads]
            for channel in itertools.chain(*channel_que):
                print(f"Checking channel: {channel.name}, type: {channel.type}, id: {channel.id}")
                if (self.filter_channels and channel.name not in self.filter_channels and
                        (type(channel) != Thread or channel.parent.name not in self.filter_channels)):
                    continue
                if isinstance(channel, discord.TextChannel) or isinstance(channel, discord.Thread):
                    last_timestamp = self.get_last_msg_timestamps_in_db(guild, channel)
                    try:
                        after = datetime.datetime.fromtimestamp(last_timestamp) if last_timestamp else None
                        messages = [msg async for msg in channel.history(limit=None, after=after)] if channel else []
                        self.add_discord_messages_to_db(guild, channel, messages)
                    except discord.Forbidden:
                        print(f"  - No access to channel: {channel.name}")

    @util_logging.exception(__name__)
    def get_last_msg_timestamps_in_db(self, guild, channel) -> float:
        """Return the timestamp of the newest stored message."""
        chats = self.db.get(
            where={"$and": [{"guild_id": guild.id}, {"channel_id": channel.id}, {"source": self.DB_IDENTIFIER}]})

        if len(chats["ids"]) == 0:
            return None

        last_timetamp = max([chat["timestamp"] for chat in chats["metadatas"]]) if chats else None
        return last_timetamp

    @util_logging.exception(__name__)
    def add_discord_messages_to_db(self, guild, channel, messages: [discord.Message]):
        """Add Discord messages to the database."""
        ids, texts, metadatas = [], [], []

        for msg in messages:
            # print(msg)
            if len(msg.content) > 0 and msg.type not in self.FILTERED_MSG_TYPES:
                # Filter out empty announcements like MessageType.new_member
                ids.append(f"{self.DB_IDENTIFIER}_{msg.id}")
                texts.append(msg.content)
                metadatas.append({"guild_id": guild.id, "guild_name": guild.name,
                                  "channel_id": channel.id, "channel_name": channel.name,
                                  "timestamp": msg.created_at.timestamp(),
                                  "author_id": msg.author.id, "author_name": msg.author.name,
                                  "source": self.DB_IDENTIFIER, "msg_type": str(msg.type),
                                  "flags": str(msg.flags),
                                  "msg_reference": f"{self.DB_IDENTIFIER}_{msg.reference.message_id}" if msg.reference else "None",
                                  "attachments": f"{[attachment.to_dict() for attachment in msg.attachments] if msg.attachments else []}"})

        if len(ids) > 0:
            print(f"Adding {len(ids)} messages to the database")
            print(metadatas)
            return self.add_to_db_batch(ids=ids, texts=texts, metadatas=metadatas)

    @util_logging.exception(__name__)
    def get_files_from_messages(self, guild, channel, messages: [discord.Message]):
        """Store attachments of ``messages`` in the database."""
        ids, texts, metadatas = [], [], []
        for msg in messages:
            if msg.attachments and msg.type not in self.FILTERED_MSG_TYPES:
                # Create the 'images' directory if it doesn't exist
                if not os.path.exists('images'):
                    os.makedirs('images')
                    print("Created 'images' directory.")

                # Append metadata and IDs
                ids.append(f"{self.DB_IDENTIFIER}_{msg.id}")
                metadatas.append({
                    "guild_id": guild.id,
                    "guild_name": guild.name,
                    "channel_id": channel.id,
                    "channel_name": channel.name,
                    "timestamp": msg.created_at.timestamp(),
                    "author_id": msg.author.id,
                    "author_name": msg.author.name,
                    "source": self.DB_IDENTIFIER,
                    "msg_type": str(msg.type),
                    "flags": str(msg.flags),
                    "msg_reference": f"{self.DB_IDENTIFIER}_{msg.reference.message_id}" if msg.reference else "None"
                })

                # Process attachments
                for attachment in msg.attachments:
                    if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.pdf', '.xlsx',
                                                             '.pptx', '.docx', '.txt', '.xls')):
                        # Save the attachment to the 'images' directory
                        file_path = f'images/{attachment.filename}'
                        attachment.save(file_path)
                        # Append the file path as text
                        texts.append(file_path)

        if len(ids) > 0:
            return self.add_to_db_batch(ids=ids, texts=texts, metadatas=metadatas)

    @util_logging.exception(__name__)
    def get_links_from_messages(self, guild, channel, messages: [discord.Message]):
        """Extract and store URLs from ``messages``."""
        ids, texts, metadatas = [], [], []
        for msg in messages:
            if len(msg.content) > 0 and msg.type not in self.FILTERED_MSG_TYPES:
                urls = URL_REGEX.findall(msg.content)
                if urls:  # Only proceed if URLs are found
                    # Join the list of URLs into a single string
                    urls_text = " ".join(urls)
                    ids.append(f"{self.DB_IDENTIFIER}_{msg.id}")
                    texts.append(urls_text)  # Append the joined string, not the list
                    metadatas.append({
                        "guild_id": guild.id,
                        "guild_name": guild.name,
                        "channel_id": channel.id,
                        "channel_name": channel.name,
                        "timestamp": msg.created_at.timestamp(),
                        "author_id": msg.author.id,
                        "author_name": msg.author.name,
                        "source": self.DB_IDENTIFIER,
                        "msg_type": str(msg.type),
                        "flags": str(msg.flags),
                        "msg_reference": f"{self.DB_IDENTIFIER}_{msg.reference.message_id}" if msg.reference else "None"
                    })
        if len(ids) > 0:
            return self.add_to_db_batch(ids=ids, texts=texts, metadatas=metadatas)

    @util_logging.exception(__name__)
    def get_surrounding_docs(self, doc_metadata, num_before=3, num_after=3) -> Tuple[List, List]:
        """Return messages surrounding a given document.

        Args:
            doc_metadata: Metadata of the reference message.
            num_before (int, optional): How many previous messages to fetch.
            num_after (int, optional): How many following messages to fetch.

        Returns:
            Tuple[List, List]: Messages before and after the reference message.
        """
        max_search_window = int(datetime.timedelta(weeks=1).total_seconds())

        guild_id = doc_metadata["guild_id"]
        channel_id = doc_metadata["channel_id"]
        doc_ts = doc_metadata["timestamp"]  # float, e.g., message.created_at.timestamp()

        # 1) Build time range
        min_ts_for_older = doc_ts - max_search_window
        max_ts_for_newer = doc_ts + max_search_window

        # 2) Fetch older messages within 1 week
        #    where: older => $lt doc_ts, but >= (doc_ts - 1 week)
        older_docs = self.db.get(
            where={
                "$and": [
                    {"source": self.DB_IDENTIFIER},
                    {"guild_id": guild_id},
                    {"channel_id": channel_id},
                    {"timestamp": {"$gt": min_ts_for_older}},
                    {"timestamp": {"$lt": doc_ts}}
                ]
            }
        )

        # Sort in descending order so the messages closest to doc_ts are first
        older_docs_zipped = list(zip(
            older_docs["documents"], older_docs["metadatas"], older_docs["ids"]
        ))
        older_docs_sorted = sorted(
            older_docs_zipped,
            key=lambda x: x[1]["timestamp"],  # use metadatas["timestamp"]
            reverse=True
        )
        prev_docs = reversed(older_docs_sorted[:num_before])

        # 3) Fetch newer messages within 1 week
        #    where: newer => $gt doc_ts, but <= (doc_ts + 1 week)
        newer_docs = self.db.get(
            where={
                "$and": [
                    {"source": self.DB_IDENTIFIER},
                    {"guild_id": guild_id},
                    {"channel_id": channel_id},
                    {"timestamp": {"$gt": doc_ts}},
                    {"timestamp": {"$lt": max_ts_for_newer}}
                ]
            }
        )

        newer_docs_zipped = list(zip(
            newer_docs["documents"], newer_docs["metadatas"], newer_docs["ids"]
        ))
        # Sort in ascending order so the messages closest after doc_ts are first
        newer_docs_sorted = sorted(
            newer_docs_zipped,
            key=lambda x: x[1]["timestamp"],
            reverse=False
        )
        next_docs = newer_docs_sorted[:num_after]

        return prev_docs, next_docs
