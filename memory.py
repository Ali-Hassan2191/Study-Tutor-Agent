import os

import streamlit as st
from crewai import LLM, Memory
from sklearn.feature_extraction.text import HashingVectorizer


MODEL_NAME = "gemini/gemini-3.8-flash"


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add it to Streamlit secrets."
        )

    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
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
