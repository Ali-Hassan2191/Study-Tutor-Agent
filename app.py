import uuid
import streamlit as st

from tutor import (
    ask_tutor,
    extract_text,
    chunk_text,
    generate_quiz,
)

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        border: 1px solid rgba(128,128,128,.18);
        background: linear-gradient(135deg, rgba(80,120,255,.10), rgba(120,80,220,.08));
        margin-bottom: 1.25rem;
    }
    .hero h1 {margin: 0 0 .35rem 0;}
    .hero p {margin: 0; opacity: .78;}
    .card {
        padding: 1rem 1.1rem; border-radius: 16px;
        border: 1px solid rgba(128,128,128,.18);
        background: rgba(128,128,128,.035); margin-bottom: .8rem;
    }
    .metric-card {
        padding: .9rem 1rem; border-radius: 14px;
        border: 1px solid rgba(128,128,128,.18);
        text-align: center;
    }
    .metric-value {font-size: 1.6rem; font-weight: 700;}
    .metric-label {font-size: .82rem; opacity: .7;}
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "material_chunks": [],
        "material_name": "",
        "subject": "General",
        "level": "Beginner",
        "quiz": None,
        "quiz_submitted": False,
        "quiz_answers": {},
        "quiz_score": 0,
        "quiz_total": 0,
        "questions_asked": 0,
        "quiz_attempts": 0,
        "correct_answers": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

with st.sidebar:
    st.header("🎓 Study Tutor")
    st.caption("AI-powered learning workspace")

    st.session_state.subject = st.text_input(
        "Subject",
        value=st.session_state.subject,
        placeholder="e.g. Physics, Biology, Python",
    )

    st.session_state.level = st.selectbox(
        "Difficulty level",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(
            st.session_state.level
        ),
    )

    st.divider()
    st.subheader("📚 Study material")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        help="The tutor can search the uploaded material when answering questions and creating quizzes.",
    )

    if uploaded is not None:
        if uploaded.name != st.session_state.material_name:
            with st.spinner("Reading study material..."):
                text = extract_text(uploaded.name, uploaded.getvalue())
                chunks = chunk_text(text)

            st.session_state.material_name = uploaded.name
            st.session_state.material_chunks = chunks
            st.session_state.quiz = None
            st.session_state.quiz_submitted = False

            if chunks:
                st.success(f"Loaded {len(chunks)} searchable chunks.")
            else:
                st.warning("No readable text was found in this file.")

    if st.session_state.material_name:
        st.caption(f"Loaded: **{st.session_state.material_name}**")
        if st.button("Remove material", use_container_width=True):
            st.session_state.material_name = ""
            st.session_state.material_chunks = []
            st.session_state.quiz = None
            st.rerun()

    st.divider()
    st.subheader("📊 Progress")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Questions", st.session_state.questions_asked)
    with c2:
        st.metric("Quizzes", st.session_state.quiz_attempts)

    if st.session_state.quiz_total:
        accuracy = round(
            st.session_state.correct_answers
            / st.session_state.quiz_total
            * 100
        )
        st.metric("Last quiz", f"{accuracy}%")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="hero">
    <h1>🎓 Study Tutor Agent</h1>
    <p>Ask questions, learn from your material, and test your understanding with adaptive quizzes.</p>
</div>
""", unsafe_allow_html=True)

learn_tab, quiz_tab = st.tabs(["💬 Learn", "🧠 Quiz"])

with learn_tab:
    if st.session_state.material_name:
        st.info(
            f"📎 The tutor can use **{st.session_state.material_name}** "
            "when your question relates to it."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your subject...",
        key="chat_input",
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_tutor(
                    session_id=st.session_state.session_id,
                    question=question,
                    subject=st.session_state.subject,
                    level=st.session_state.level,
                    history=st.session_state.messages,
                    material_chunks=st.session_state.material_chunks,
                )
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.questions_asked += 1

with quiz_tab:
    st.subheader("🧠 Generate a quiz")

    topic = st.text_input(
        "Quiz topic",
        placeholder="e.g. Newton's laws, cell biology, Python functions",
    )
    number_of_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button("Generate quiz", type="primary"):
        if not topic.strip():
            st.warning("Enter a quiz topic first.")
        else:
            with st.spinner("Creating your quiz..."):
                try:
                    quiz = generate_quiz(
                        session_id=st.session_state.session_id,
                        topic=topic,
                        subject=st.session_state.subject,
                        level=st.session_state.level,
                        number_of_questions=number_of_questions,
                        material_chunks=st.session_state.material_chunks,
                    )
                    st.session_state.quiz = quiz
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                except Exception as exc:
                    st.error(f"Could not generate the quiz: {exc}")

    quiz = st.session_state.quiz

    if quiz and quiz.get("questions"):
        questions = quiz["questions"]

        st.divider()
        st.caption(
            f"{len(questions)} questions • "
            f"{st.session_state.subject} • {st.session_state.level}"
        )

        with st.form("quiz_form"):
            answers = {}

            for index, item in enumerate(questions, start=1):
                st.markdown(
                    f'<div class="card"><strong>Question {index}</strong><br>{item["question"]}</div>',
                    unsafe_allow_html=True,
                )
                answers[index - 1] = st.radio(
                    "Choose an answer",
                    item["options"],
                    key=f"quiz_{id(quiz)}_{index}",
                    index=None,
                )

            submitted = st.form_submit_button(
                "Submit quiz",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            score = 0
            for index, item in enumerate(questions):
                if answers.get(index) == item.get("correct_answer"):
                    score += 1

            st.session_state.quiz_answers = answers
            st.session_state.quiz_score = score
            st.session_state.quiz_total = len(questions)
            st.session_state.correct_answers = score
            st.session_state.quiz_attempts += 1
            st.session_state.quiz_submitted = True
            st.rerun()

        if st.session_state.quiz_submitted:
            score = st.session_state.quiz_score
            total = st.session_state.quiz_total
            percentage = round(score / total * 100) if total else 0

            st.success(f"Score: {score}/{total} ({percentage}%)")
            st.progress(percentage / 100)

            for index, item in enumerate(questions):
                selected = st.session_state.quiz_answers.get(index)
                correct = item.get("correct_answer")

                if selected == correct:
                    st.success(f"Q{index + 1}: Correct")
                else:
                    st.error(
                        f"Q{index + 1}: Incorrect. "
                        f"Your answer: {selected or 'No answer'}"
                    )
                    st.write(f"**Correct answer:** {correct}")

                explanation = item.get("explanation")
                if explanation:
                    st.caption(f"Explanation: {explanation}")
