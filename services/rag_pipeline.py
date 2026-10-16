# services/rag_pipeline.py
from groq import Groq
from config import settings
from typing import List, Dict, Any
import json
import re
import logging

logger = logging.getLogger(__name__)


class GroundedRAGEngine:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.active_model = settings.LLM_MODEL

    @staticmethod
    def _extract_query_labels(query: str) -> set:
        labels = set()
        labels |= {m.upper() for m in re.findall(r"\bQ\d+\b", query, re.IGNORECASE)}
        labels |= set(re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", query))
        return labels

    @staticmethod
    def _labels_present_in_context(labels: set, contexts: List[Dict[str, Any]]) -> set:
        context_text = " ".join(c.get("text", "") for c in contexts).lower()
        return {lbl for lbl in labels if lbl.lower() in context_text}

    async def generate_answer(
        self, query: str, contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        if not contexts:
            return {
                "answer": "Information not located in document archives.",
                "citations": [],
            }

        # ---- Pre-LLM label short-circuit ----
        query_labels = self._extract_query_labels(query)
        if query_labels:
            present = self._labels_present_in_context(query_labels, contexts)
            missing = query_labels - present
            is_asking_about = any(
                marker in query.lower()
                for marker in ["what is q", "what are q", "define q", "explain q"]
            )
            if missing and is_asking_about:
                ms = ", ".join(sorted(missing))
                logger.warning(f"[/rag] labels {ms} not in context")
                return {
                    "answer": (
                        f"Information not located in document archives. "
                        f"The label(s) {ms} do not appear in the retrieved context."
                    ),
                    "citations": [],
                }

        # ---- Build formatted context ----
        blocks = []
        for c in contexts:
            page = c.get("page", "N/A")
            section = c.get("section", "")
            text = c.get("text", "")
            header = f"[SOURCE: Page {page}"
            if section:
                header += f", Section {section}"
            header += "]"
            blocks.append(f"{header}\n{text}\n[END OF PAGE {page}]")
        formatted_context = "\n\n".join(blocks)

        # ---- System prompt ----
        system_prompt = (
            "You are a strict technical verification system. "
            "Answer using ONLY the provided context blocks. "
            "You MUST respond with valid JSON at all times.\n\n"
            "RULES:\n"
            "1. If the context CONTAINS the answer, answer directly with [Page X] citations.\n"
            "2. If the query asks about a SPECIFIC ITEM (a language, a section number, "
            "a named entity) and the context explicitly mentions a list/section that "
            "DOES NOT contain that item, respond with a CONFIRMED NEGATIVE:\n"
            "   'X is not present in the document. Specifically, [source] lists [actual items] "
            "and X is not among them.'\n"
            "3. Only use 'Information not located in document archives.' if you have NO "
            "context that relates to the query at all.\n"
            "4. If a query asks about a specific section number and the context includes "
            "that section, answer with the section's content. Do NOT refuse.\n"
            "5. NEVER invent columns, list items, or section numbers not in the context.\n"
            "6. Return ONLY this JSON shape: {\"answer\": str, \"citations\": [int]}\n"
            "7. When a query mentions a specific section number (like 3.3), answer ONLY from "
            "the chunk whose [SOURCE: Section X.Y] tag matches. Do NOT mix content from "
            "adjacent sections."
            "8. When reproducing tables or lists, output EXACTLY the items in the context. "
            "Do NOT add items from memory or inference."
        )

        user_prompt = f"Context Blocks:\n{formatted_context}\n\nQuery: {query}"

        # ---- Call Groq ----
        try:
            response = self.groq_client.chat.completions.create(
                model=self.active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.exception("Groq API call failed")
            return {
                "answer": f"LLM service error: {type(e).__name__}: {e}",
                "citations": [],
            }

        # ---- Extract raw content ----
        try:
            raw_content = response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            return {"answer": f"Malformed LLM response: {e}", "citations": []}

        raw_content = (raw_content or "").strip()

        # ---- Parse JSON ----
        parsed = None
        try:
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))

            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]

            if not isinstance(parsed, dict):
                raise ValueError("Model returned non-object JSON")
        except Exception as e:
            logger.exception(f"JSON parse failed. Raw: {raw_content[:300]!r}")
            return {
                "answer": f"System encountered a parsing anomaly: {e}",
                "citations": [],
            }

        # ---- Guard: parsed must be a dict ----
        if parsed is None or not isinstance(parsed, dict):
            logger.error(f"[/rag] parsed is not a dict: {type(parsed)}")
            return {
                "answer": "Information not located in document archives.",
                "citations": [],
            }

        # ---- Citation validation ----
        valid_pages = {c.get("page") for c in contexts if c.get("page") is not None}

        raw_citations = parsed.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []
        parsed["citations"] = [p for p in raw_citations if p in valid_pages]

        # ---- Inline citation cleanup ----
        def _clean_inline(text: str) -> str:
            def repl(m):
                n = int(m.group(1))
                if n not in valid_pages:
                    logger.warning(f"[/rag] stripped invalid citation [Page {n}]")
                    return ""
                return m.group(0)
            return re.sub(r"\[Page\s+(\d+)\]", repl, text)

        parsed["answer"] = _clean_inline(parsed.get("answer", ""))
        parsed.setdefault("answer", "")
        parsed.setdefault("citations", [])

        return parsed