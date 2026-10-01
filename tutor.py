import json
import re
from io import BytesIO

from crewai import Crew, Process, Task
from pypdf import PdfReader

from agent import create_study_tutor
from memory import get_student_memory, recall, remember


def extract_text(file_name, file_bytes):
    extension = file_name.lower().split(".")[-1]

    if extension == "pdf":
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    if extension in {"txt", "md", "markdown"}:
        return file_bytes.decode("utf-8", errors="ignore")

    return ""


def chunk_text(text, chunk_size=700, overlap=100):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def format_history(messages, limit=12):
    if not messages:
        return "No previous conversation."

    recent = messages[-limit:]
    return "\n".join(
        f'{message["role"].upper()}: {message["content"]}'
        for message in recent
    )


def run_crew(agent, description, expected_output):
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return str(crew.kickoff())


def ask_tutor(
    session_id,
    question,
    subject,
    level,
    history,
    material_chunks=None,
):
    material_chunks = material_chunks or []

    student_memory = get_student_memory(session_id)
    memories = recall(session_id, question, limit=5)
    memory_context = "\n\n".join(memories) or "No relevant long-term memory found."

    conversation = format_history(history)

    material_instruction = ""
    if material_chunks:
        material_instruction = """
The student has uploaded study material.
When the question can be answered from that material, use the
Study Material Search tool before answering.
Do not invent facts that conflict with the material.
If the material does not contain enough information, say so clearly.
"""

    prompt = f"""
You are the Study Tutor Agent.

SUBJECT:
{subject}

STUDENT LEVEL:
{level}

STUDENT QUESTION:
{question}

PREVIOUS CONVERSATION:
{conversation}

RELEVANT STUDENT MEMORY:
{memory_context}

{material_instruction}

TEACHING RULES:
1. Adapt the explanation to the student's level.
2. Use simple language for beginners.
3. Break difficult ideas into steps.
4. Use examples or analogies when helpful.
5. Correct misconceptions politely.
6. Avoid unnecessary length.
7. Use the Calculator tool for numerical calculations.
8. Use Study Material Search when uploaded material is relevant.
9. Maintain context from the conversation.
10. Encourage understanding rather than memorization.
11. If useful, finish with one short check-for-understanding question.

Return only the tutor's response.
"""

    agent = create_study_tutor(
        memory=student_memory,
        material_chunks=material_chunks,
    )

    response = run_crew(
        agent=agent,
        description=prompt,
        expected_output=(
            "A clear, accurate, educational tutoring response "
            "appropriate for the student's level."
        ),
    )

    remember(
        session_id=session_id,
        question=question,
        answer=response,
        subject=subject,
    )

    return response


def generate_quiz(
    session_id,
    topic,
    subject,
    level,
    number_of_questions,
    material_chunks=None,
):
    material_chunks = material_chunks or []

    student_memory = get_student_memory(session_id)
    memories = recall(session_id, topic, limit=3)
    memory_context = "\n\n".join(memories) or "No relevant student memory found."

    material_instruction = ""
    if material_chunks:
        material_instruction = """
The student has uploaded study material.
Use the Study Material Search tool first.
The quiz should be based primarily on the uploaded material when relevant.
"""

    prompt = f"""
Create a multiple-choice quiz for a student.

SUBJECT:
{subject}

TOPIC:
{topic}

LEVEL:
{level}

NUMBER OF QUESTIONS:
{number_of_questions}

RELEVANT STUDENT MEMORY:
{memory_context}

{material_instruction}

Return ONLY valid JSON. Do not use markdown fences.

The JSON must have exactly this structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct_answer": "Option A",
      "explanation": "Short explanation"
    }}
  ]
}}

Rules:
- Exactly {number_of_questions} questions.
- Exactly four options per question.
- Only one correct answer per question.
- correct_answer must exactly match one option.
- Questions must match the student's level.
- Keep explanations short and educational.
- Do not add text outside the JSON.
"""

    agent = create_study_tutor(
        memory=student_memory,
        material_chunks=material_chunks,
    )

    raw = run_crew(
        agent=agent,
        description=prompt,
        expected_output="Valid JSON containing the requested quiz.",
    )

    quiz = parse_json_response(raw)

    if not isinstance(quiz, dict) or not isinstance(quiz.get("questions"), list):
        raise ValueError("The tutor returned an invalid quiz structure.")

    return quiz


def parse_json_response(text):
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for start, char in enumerate(cleaned):
        if char not in "{[":
            continue

        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("The tutor returned invalid JSON.")
