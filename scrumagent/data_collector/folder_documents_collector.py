import hashlib
from pathlib import Path
from typing import Union

import discord
import nltk
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader

from .base_collector import BaseCollector


class DirectoryCollector(BaseCollector):
    DB_IDENTIFIER = "folder_doc"

    def __init__(self, bot: discord.Client, chroma_db: Chroma, folder_path: Union[Path, str]):
        self.folder_path = str(folder_path)

        # Used by DirectoryLoader ...
        # The NLTK dataset names were incorrect which caused runtime
        # download errors. Use the proper identifiers instead.
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        super().__init__(bot, chroma_db)

    async def on_startup(self):
        await self.check_all_files_in_folder()

    async def check_all_files_in_folder(self):
        loader = DirectoryLoader(self.folder_path, glob="**/*.txt")
        docs = loader.load()
        ids = [f"{self.DB_IDENTIFIER}_{hashlib.md5(doc.metadata['source'].encode('UTF-8')).hexdigest()}" for doc in
               docs]

        for doc in docs:
            doc.metadata["source"] = self.DB_IDENTIFIER

        self.add_to_db_docs(docs, ids=ids)
