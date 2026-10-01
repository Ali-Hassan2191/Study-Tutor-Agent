import ast
import operator
from typing import List

from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A mathematical expression, such as 25 * 4 + 10."
    )


class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = (
        "Calculate mathematical expressions accurately. "
        "Use this tool whenever a numerical calculation is required."
    )
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            return str(self._safe_calculate(expression))
        except Exception:
            return "Unable to calculate this expression."

    @staticmethod
    def _safe_calculate(expression):
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
        }

        def evaluate(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Unsupported constant")

            if isinstance(node, ast.BinOp):
                left = evaluate(node.left)
                right = evaluate(node.right)
                operation = allowed_operators.get(type(node.op))
                if operation is None:
                    raise ValueError("Unsupported operator")
                return operation(left, right)

            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand)
                if isinstance(node.op, ast.USub):
                    return -value
                if isinstance(node.op, ast.UAdd):
                    return value

            raise ValueError("Unsupported expression")

        tree = ast.parse(expression, mode="eval")
        return evaluate(tree.body)


class MaterialSearchInput(BaseModel):
    query: str = Field(
        description="The student's question or search query about uploaded material."
    )


class StudyMaterialSearchTool(BaseTool):
    name: str = "Study Material Search"
    description: str = (
        "Search the student's uploaded study material and return the most "
        "relevant passages. Use this tool when uploaded material is available."
    )
    args_schema = MaterialSearchInput
    chunks: List[str] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        if not self.chunks:
            return "No study material is currently available."

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
            )
            matrix = vectorizer.fit_transform(self.chunks)
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, matrix)[0]

            ranked_indexes = similarities.argsort()[::-1]
            results = []

            for index in ranked_indexes[:3]:
                score = similarities[index]
                if score <= 0:
                    continue

                results.append(
                    f"[Relevant passage | similarity={score:.2f}]\n"
                    f"{self.chunks[index]}"
                )

            if not results:
                return "No relevant passage was found in the uploaded material."

            return "\n\n".join(results)

        except Exception as exc:
            return f"Material search failed: {exc}"
