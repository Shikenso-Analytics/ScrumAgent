import os
import re
from pathlib import Path

import ollama
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

mod_path = Path(__file__).parent


def get_image_description_via_llama(image_path: str) -> str:
    """Return a short description of the image at ``image_path``.

    The function sends the image to an Ollama model and returns the
    generated description.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The text description produced by the model.
    """

    response = ollama.chat(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [image_path],
            }
        ],
    )
    return response["message"]["content"]


def init_discord_chroma_db():
    """Initialise and return the Chroma database for Discord logs."""

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    CHROMA_PATH = str(mod_path / os.getenv("CHROMA_DB_PATH"))
    CHROMA_DB_DISCORD_CHAT_DATA_NAME = os.getenv("CHROMA_DB_DISCORD_CHAT_DATA_NAME")

    # embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    persistent_chromadb = chromadb.PersistentClient(CHROMA_PATH)
    persistent_chromadb.get_or_create_collection(CHROMA_DB_DISCORD_CHAT_DATA_NAME)

    chroma_db_inst = Chroma(
        client=persistent_chromadb,
        collection_name=CHROMA_DB_DISCORD_CHAT_DATA_NAME,
        embedding_function=embeddings,
    )
    return chroma_db_inst


def split_text_smart(text, max_length=2000):
    """Split ``text`` into chunks of at most ``max_length`` characters.

    The function tries to keep paragraphs and list items intact when
    splitting the text.

    Args:
        text (str): The input string.
        max_length (int, optional): Maximum length of each chunk. Defaults to
            ``2000``.

    Returns:
        list[str]: The resulting list of text segments.
    """
    if len(text) <= max_length:
        return [text]

    # Split by double newlines first (paragraphs)
    paragraphs = re.split(r'(\n\n+)', text)
    sections = []
    current_section = ""

    for part in paragraphs:
        if len(current_section) + len(part) <= max_length:
            current_section += part
        else:
            if current_section:
                sections.append(current_section)
                current_section = ""
            if len(part) > max_length:
                # Hard split for long segments
                sub_parts = [part[i: i + max_length] for i in range(0, len(part), max_length)]
                sections.extend(sub_parts[:-1])
                current_section = sub_parts[-1]
            else:
                current_section = part

    if current_section:
        sections.append(current_section)

    return sections

