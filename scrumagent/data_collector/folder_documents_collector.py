import hashlib
from pathlib import Path
from typing import Union

import discord
import nltk
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader

from .base_collector import BaseCollector


class DirectoryCollector(BaseCollector):
    """Collect text files from a folder and store them in Chroma."""

    DB_IDENTIFIER = "folder_doc"

    def __init__(self, bot: discord.Client, chroma_db: Chroma, folder_path: Union[Path, str]):
        """Create the directory collector.

        Args:
            bot (discord.Client): Discord client instance.
            chroma_db (Chroma): Database handle.
            folder_path (Path | str): Path to the folder to scan.
        """

        self.folder_path = str(folder_path)

        # Used by DirectoryLoader ...
        # The NLTK dataset names were incorrect which caused runtime
        # download errors. Use the proper identifiers instead.
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        super().__init__(bot, chroma_db)

    async def on_startup(self):
        """Load all files when the bot starts."""

        await self.check_all_files_in_folder()

    async def check_all_files_in_folder(self):
        """Load all text files from the configured folder."""

        loader = DirectoryLoader(self.folder_path, glob="**/*.txt")
        docs = loader.load()
        ids = [f"{self.DB_IDENTIFIER}_{hashlib.md5(doc.metadata['source'].encode('UTF-8')).hexdigest()}" for doc in docs]

        for doc in docs:
            doc.metadata["source"] = self.DB_IDENTIFIER

        self.add_to_db_docs(docs, ids=ids)
