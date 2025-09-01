import datetime
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CHROMA_DB_PATH", "chroma")
os.environ.setdefault("CHROMA_DB_DISCORD_CHAT_DATA_NAME", "discord-test")

from scrumagent.main_discord_bot import standup_due_today


class Monday(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2024, 12, 30, tzinfo=tz)


def test_no_standup_tag_skips_posting():
    """User stories tagged with ``no stand-up`` should not trigger messages."""
    us = SimpleNamespace(tags=["no stand-up"])
    with patch("scrumagent.main_discord_bot.datetime.datetime", Monday):
        assert standup_due_today(us) is False
