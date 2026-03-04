from typing import List, Dict, Tuple
from pydantic import BaseModel, ConfigDict

from ollama_client import chat_once


class QAAnswer(BaseModel):
    model_config = ConfigDict(strict=True)
    answer: str
    citations: list[str]
    confidence: float


def build_prompt(question: str, evidence: List[Tuple[Dict[str, str], float]]) -> str:
    blocks = []
    for item, score in evidence:
        blocks.append(f"[{item['id']}] (score={score:.3f}) {item['text']}")
    evidence_text = "\n".join(blocks)

    return f"""
You must output JSON only. Do not output any additional text.

You must answer strictly based on the provided Evidence. 
If the Evidence is insufficient to answer the question, set:
- answer to "UNKNOWN"
- citations to an empty array
- confidence to 0.0

Evidence:
{evidence_text}

JSON schema:
{{
  "answer": "...",
  "citations": ["note-001", "..."],
  "confidence": 0-1
}}

Question: {question}
""".strip()


def answer(question: str, evidence: List[Tuple[Dict[str, str], float]]) -> QAAnswer:
    prompt = build_prompt(question, evidence)
    raw = chat_once(prompt)
    return QAAnswer.model_validate_json(raw)