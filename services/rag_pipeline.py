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

        # ---- Guard: no contexts ----
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

        # ---- Build formatted context (STRIPPED SECTIONS) ----
        # We completely remove the section labels because Chandra OCR dropped "16. Limitations",
        # causing the chunk to be mislabeled as "15 Results", which confuses the LLM.
        # ---- Build formatted context (STRIPPED SECTIONS) ----
        # We completely remove the section labels because OCR drops some headings,
        # causing chunks to be mislabeled. We force the LLM to trust the content.
        blocks = []
        for c in contexts:
            page = c.get("page", "N/A")
            text = c.get("text", "")
            header = f"[SOURCE: Page {page}]"
            blocks.append(f"{header}\n{text}\n[END OF PAGE {page}]")
        formatted_context = "\n\n".join(blocks)

        # ---- System prompt ----
        system_prompt = (
            "You are a strict technical verification system. "
            "Answer using ONLY the provided context blocks. "
            "You MUST respond with valid JSON at all times.\n\n"
            "CRITICAL RULES:\n"
            "1. The section labels have been intentionally removed because OCR errors "
            "often mislabel them. You MUST answer based purely on the TEXT CONTENT.\n"
            "2. If the user asks for 'limitations' and the text contains a numbered "
            "list of limitations (e.g., '1. In-memory likes... 2. No real authentication...'), "
            "you MUST extract the first item from that list. Do NOT refuse.\n"
            "3. If the user asks for 'future scope' or 'solutions' and the text contains "
            "a future scope list (e.g., '17. Future Scope 1. Migrate... 2. Real Redis...'), "
            "extract the matching solution. Do NOT refuse.\n"
            "4. Only say 'Information not located in document archives.' if the context "
            "contains absolutely NO relevant text. A numbered list containing the keywords "
            "from the query is ALWAYS relevant.\n"
            "5. For every factual claim, include an inline [Page X] citation.\n"
            "6. Return ONLY this JSON shape: {\"answer\": str, \"citations\": [int]}\n"
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

        # ---- Derive citations from [Page X] tags in the answer ----
        valid_pages = {c.get("page") for c in contexts if c.get("page") is not None}
        answer_text = parsed.get("answer", "")

        inline_pages = re.findall(r"\[Page\s+(\d+)\]", answer_text)
        derived_citations = sorted({
            int(p) for p in inline_pages if int(p) in valid_pages
        })

        # Fallback: if no inline citations, use the LLM's own citations array
        if not derived_citations:
            raw = parsed.get("citations", [])
            if isinstance(raw, list):
                derived_citations = sorted([
                    int(p) for p in raw if isinstance(p, int) and p in valid_pages
                ])

        parsed["citations"] = derived_citations

        # ---- Strip invalid inline citations ----
        def _clean_inline(text: str) -> str:
            def repl(m):
                n = int(m.group(1))
                if n not in valid_pages:
                    logger.warning(f"[/rag] stripped invalid citation [Page {n}]")
                    return ""
                return m.group(0)
            return re.sub(r"\[Page\s+(\d+)\]", repl, text)

        parsed["answer"] = _clean_inline(answer_text)
        parsed.setdefault("answer", "")
        parsed.setdefault("citations", [])

        return parsed