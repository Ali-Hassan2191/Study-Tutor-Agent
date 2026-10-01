# Study Tutor Agent

A single-agent AI Study Tutor MVP built with:

- Streamlit frontend
- CrewAI agent, tools, and memory
- Groq's OpenAI-compatible API
- GPT-OSS 120B
- PDF/TXT/Markdown study material search
- Adaptive explanations
- Quiz generation
- Deterministic quiz scoring
- Conversation context
- Basic progress tracking

## Features

### 1. Ask questions
Students can ask academic questions through the chat interface.

### 2. Explain concepts
The tutor adapts explanations to:

- Beginner
- Intermediate
- Advanced

### 3. Conversation context
Recent conversation messages are included in tutor prompts.

### 4. CrewAI memory
The app also stores study interactions in scoped CrewAI memory for the current session.

### 5. Study material upload
Upload:

- PDF
- TXT
- Markdown

The uploaded material is chunked and searchable with a custom CrewAI tool.

### 6. Calculator tool
The tutor has a safe calculator tool for arithmetic.

### 7. Quiz generation
The agent generates multiple-choice quizzes as JSON.

### 8. Quiz evaluation
Multiple-choice answers are scored deterministically in the UI so the score is consistent and does not require an extra LLM call for every question.

### 9. Progress tracking
The sidebar shows:

- Questions asked
- Quiz attempts
- Last quiz accuracy

## Project structure

```text
study-tutor-agent/
├── app.py
├── agent.py
├── tutor.py
├── tools.py
├── memory.py
├── requirements.txt
├── README.md
├── .python-version
└── .gitignore
```

## Environment variable

The app needs:

```text
GROQ_API_KEY
```

Never commit the API key to GitHub.

## Browser-only deployment on GitHub + Render

### 1. Create a GitHub repository

Create a new repository on GitHub, then add all project files through the GitHub web interface.

### 2. Add the files

Create these files in the repository:

```text
app.py
agent.py
tutor.py
tools.py
memory.py
requirements.txt
README.md
.python-version
.gitignore
```

Paste the corresponding code into each file and commit the changes.

### 3. Create a Render Web Service

Connect the GitHub repository to Render.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

### 4. Add the Groq API key

In Render, add an environment variable:

```text
GROQ_API_KEY = your_real_groq_api_key
```

Do not put the key in GitHub.

## Python version

The project includes:

```text
3.13
```

This is intentional because the pinned CrewAI version requires Python below 3.14.

## Important MVP limitation

CrewAI memory is stored under:

```text
.crewai/memory
```

That local filesystem is suitable for an MVP/demo. On hosted infrastructure, local storage should not be treated as durable production storage.

For a production version, move memory/document storage to a persistent external database or vector store.

## Local development

Local development is optional. The intended deployment workflow for this project is:

```text
ChatGPT → GitHub web UI → Render
```

No local Git, Python, or Render CLI is required for deployment.
