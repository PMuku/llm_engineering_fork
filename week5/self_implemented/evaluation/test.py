import json
from pathlib import Path

from pydantic import BaseModel, Field


TEST_FILE = Path(__file__).parents[2] / "evaluation" / "tests.jsonl"


class TestQuestion(BaseModel):
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category")


def load_tests(test_file: str | Path | None = None) -> list[TestQuestion]:
    path = Path(test_file) if test_file else TEST_FILE
    if test_file and not path.is_absolute():
        path = TEST_FILE.parent / path

    with path.open(encoding="utf-8") as file:
        return [TestQuestion(**json.loads(line)) for line in file if line.strip()]