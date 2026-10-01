from crewai import Agent

from memory import get_llm
from tools import CalculatorTool, StudyMaterialSearchTool


def create_study_tutor(memory, material_chunks=None):
    material_chunks = material_chunks or []

    calculator = CalculatorTool()
    material_search = StudyMaterialSearchTool(chunks=material_chunks)

    return Agent(
        role="Study Tutor Agent",
        goal=(
            "Help students understand academic concepts clearly and deeply. "
            "Adapt explanations to the student's level, use examples, check "
            "understanding, generate practice questions, and give useful feedback."
        ),
        backstory=(
            "You are a patient and knowledgeable academic tutor. "
            "You break difficult ideas into manageable steps, use examples "
            "and analogies, correct misconceptions politely, and encourage "
            "students to understand rather than memorize."
        ),
        llm=get_llm(),
        tools=[calculator, material_search],
        memory=memory,
        allow_delegation=False,
        verbose=False,
    )
