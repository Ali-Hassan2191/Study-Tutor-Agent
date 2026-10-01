import os

import streamlit as st
from crewai import LLM, Memory
from groq import Groq
from sklearn.feature_extraction.text import HashingVectorizer


MODEL_NAME = "openai/gpt-oss-120b"


class GroqCrewLLM(LLM):
    """
    CrewAI-compatible LLM that sends requests directly to Groq.

    This avoids CrewAI's custom_openai model-prefix handling, which strips
    the "openai/" prefix that Groq requires for GPT-OSS 120B.
    """

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        **kwargs,
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add it to Streamlit secrets."
            )

        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages,
                }
            ]

        client = Groq(api_key=api_key)

        request = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.3,
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        if "max_tokens" in kwargs and kwargs["max_tokens"]:
            request["max_tokens"] = kwargs["max_tokens"]

        response = client.chat.completions.create(**request)

        message = response.choices[0].message

        if message.tool_calls:
            return message.tool_calls

        return message.content or ""

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 131072


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit secrets."
        )

    return GroqCrewLLM(
        model=MODEL_NAME,
        temperature=0.3,
    )


_vectorizer = HashingVectorizer(
    n_features=256,
    alternate_sign=False,
    norm="l2",
)


def embed_texts(texts):
    matrix = _vectorizer.transform(texts)
    return matrix.toarray().tolist()


@st.cache_resource
def get_memory():
    return Memory(
        llm=get_llm(),
        embedder=embed_texts,
        storage="./.crewai/memory",
        semantic_weight=0.5,
        recency_weight=0.3,
        importance_weight=0.2,
    )


def get_student_memory(session_id: str):
    return get_memory().scope(f"/student/{session_id}")


def remember(session_id: str, question: str, answer: str, subject: str):
    memory = get_student_memory(session_id)

    content = f"""
Subject: {subject}

Student question:
{question}

Tutor response:
{answer}
"""

    memory.remember(
        content,
        categories=["study_session"],
        importance=0.5,
    )


def recall(session_id: str, question: str, limit: int = 5):
    memory = get_student_memory(session_id)

    matches = memory.recall(
        question,
        limit=limit,
        depth="shallow",
    )

    return [match.record.content for match in matches]
