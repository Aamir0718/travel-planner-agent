from __future__ import annotations

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from llama_index.core import Settings
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq as LlamaGroq

# Chat runs on Groq; embeddings run remotely on the Hugging Face Inference API (nothing local).
DEFAULT_LLM = "openai/gpt-oss-120b"
DEFAULT_EMBED = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_langchain_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )


@lru_cache(maxsize=1)
def get_langchain_embeddings() -> HuggingFaceEndpointEmbeddings:
    model = os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED)
    return HuggingFaceEndpointEmbeddings(model=model, huggingfacehub_api_token=os.getenv("HF_TOKEN"))


@lru_cache(maxsize=1)
def get_llama_llm() -> LlamaGroq:
    return LlamaGroq(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )


@lru_cache(maxsize=1)
def get_llama_embed_model() -> HuggingFaceInferenceAPIEmbedding:
    model = os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED)
    return HuggingFaceInferenceAPIEmbedding(model_name=model, token=os.getenv("HF_TOKEN"))


def configure_settings() -> None:
    Settings.llm = get_llama_llm()
    Settings.embed_model = get_llama_embed_model()


def complete(prompt: str, max_tokens: int = 700) -> str:
    llm = get_langchain_llm().bind(max_tokens=max_tokens)
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()
