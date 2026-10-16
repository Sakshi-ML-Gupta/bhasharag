# services/indic_translator.py
import re
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


# All 22 scheduled Indian languages + English (FLORES-200 codes)
LANG_CODE_MAP = {
    "as":  "asm_Beng",
    "bn":  "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "en":  "eng_Latn",
    "gom": "gom_Deva",
    "gu":  "guj_Gujr",
    "hi":  "hin_Deva",
    "kn":  "kan_Knda",
    "ks":  "kas_Arab",
    "mai": "mai_Deva",
    "ml":  "mal_Mlym",
    "mr":  "mar_Deva",
    "mni": "mni_Beng",
    "ne":  "npi_Deva",
    "or":  "ory_Orya",
    "pa":  "pan_Guru",
    "sa":  "san_Deva",
    "sat": "sat_Olck",
    "sd":  "snd_Arab",
    "ta":  "tam_Taml",
    "te":  "tel_Telu",
    "ur":  "urd_Arab",
}

FLORES_TO_ISO = {v: k for k, v in LANG_CODE_MAP.items()}

CITATION_RE = re.compile(r"\[Page \d+\]")

REFUSAL_EN = "Information not located in document archives."


class IndicTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[IndicTranslator] Loading on device: {self.device}")

        # Pre-cached refusal message for all target languages
        self._refusal_cache = {"en": REFUSAL_EN}

        self.processor = IndicProcessor(inference=True)

        # --- Indic -> English ---
        print("[IndicTranslator] Loading indic-en model...")
        self.ind2en_tok = AutoTokenizer.from_pretrained(
            "ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True
        )
        self.ind2en_model = AutoModelForSeq2SeqLM.from_pretrained(
            "ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True
        ).to(self.device)

        # --- English -> Indic ---
        print("[IndicTranslator] Loading en-indic model...")
        self.en2ind_tok = AutoTokenizer.from_pretrained(
            "ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True
        )
        self.en2ind_model = AutoModelForSeq2SeqLM.from_pretrained(
            "ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True
        ).to(self.device)

        # --- IndicLID for accurate language identification ---
        self.lid_model = self._load_lid_model()

        print("[IndicTranslator] Ready.")

    # ---------------- IndicLID loading ----------------

    def _load_lid_model(self):
        """Load IndicLID fasttext model — prefer native-script, fall back to roman."""
        if not FASTTEXT_AVAILABLE:
            print("[IndicTranslator] fasttext not installed — LID disabled")
            return None

        native_paths = [
            Path(__file__).parent.parent / "IndicLID" / "models" / "model_baseline_native.bin",
            Path.cwd() / "IndicLID" / "models" / "model_baseline_native.bin",
            Path(__file__).parent.parent / "IndicLID" / "models" / "indiclid_ftn.bin",
        ]
        roman_paths = [
            Path(__file__).parent.parent / "IndicLID" / "models" / "model_baseline_roman.bin",
            Path.cwd() / "IndicLID" / "models" / "model_baseline_roman.bin",
            Path(__file__).parent.parent / "IndicLID" / "models" / "indiclid_ftr.bin",
        ]

        for p in native_paths + roman_paths:
            if p.exists():
                try:
                    model = fasttext.load_model(str(p))
                    kind = "native" if "native" in p.name or "ftn" in p.name else "roman"
                    print(f"[IndicTranslator] IndicLID loaded ({kind}) from {p}")
                    return model
                except Exception as e:
                    print(f"[IndicTranslator] Failed to load from {p}: {e}")

        print("[IndicTranslator] IndicLID model file not found — using script fallback")
        return None

    # ---------------- Citation protection ----------------

    def _protect_citations(self, text: str):
        placeholders = {}

        def repl(m):
            key = f"CITATIONMARKER{len(placeholders)}X"
            placeholders[key] = m.group(0)
            return key

        return CITATION_RE.sub(repl, text), placeholders

    def _restore_citations(self, text: str, placeholders: dict):
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    # ---------------- Core translation ----------------

    @torch.inference_mode()
    def _translate_batch(self, texts: list, src_flores: str, tgt_flores: str) -> list:
        if not texts:
            return []

        is_ind2en = (src_flores != "eng_Latn")
        tok = self.ind2en_tok if is_ind2en else self.en2ind_tok
        model = self.ind2en_model if is_ind2en else self.en2ind_model

        batch = self.processor.preprocess_batch(
            texts, src_lang=src_flores, tgt_lang=tgt_flores
        )
        inputs = tok(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        outputs = model.generate(
            **inputs,
            num_beams=5,
            min_length=0,
            max_length=512,
            num_return_sequences=1,
            use_cache=False,
        )

        decoded = tok.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return self.processor.postprocess_batch(decoded, lang=tgt_flores)

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        protect_citations: bool = False,
    ) -> str:
        if not text or not text.strip():
            return text

        if src_lang == tgt_lang:
            return text

        # ---- Fast path: refusal message ----
        if text.strip() == REFUSAL_EN and tgt_lang != "en":
            if tgt_lang in self._refusal_cache:
                return self._refusal_cache[tgt_lang]
            # Translate once, cache forever
            try:
                tgt_flores = LANG_CODE_MAP.get(tgt_lang, tgt_lang)
                translated = self._translate_batch(
                    [text], "eng_Latn", tgt_flores
                )[0]
                self._refusal_cache[tgt_lang] = translated
                return translated
            except Exception:
                return text

        # ---- Normal translation path ----
        src_flores = LANG_CODE_MAP.get(src_lang, src_lang)
        tgt_flores = LANG_CODE_MAP.get(tgt_lang, tgt_lang)

        print(
            f"[TRANSLATE-CALL] {src_lang}({src_flores}) -> "
            f"{tgt_lang}({tgt_flores}) | protect={protect_citations} | "
            f"len={len(text)} | preview={text[:80]!r}"
        )

        placeholders = {}
        working_text = text
        if protect_citations:
            working_text, placeholders = self._protect_citations(text)

        result = working_text
        try:
            translated_list = self._translate_batch(
                [working_text], src_flores, tgt_flores
            )
            result = translated_list[0] if translated_list else working_text
            print(f"[TRANSLATE-OK] result preview={result[:80]!r}")
        except Exception as e:
            print("=" * 80)
            print("[TRANSLATE-FAIL] Translation threw an exception")
            print(f"  src={src_lang}({src_flores}) tgt={tgt_lang}({tgt_flores})")
            print(f"  text[:120]={text[:120]!r}")
            print(f"  error={type(e).__name__}: {e}")
            traceback.print_exc()
            print("=" * 80)

        if protect_citations and placeholders:
            result = self._restore_citations(result, placeholders)

        return result

    # ---------------- Language detection ----------------

    def detect_language(self, text: str) -> str:
        """
        Detect language using IndicLID (if loaded), else script-based fallback.
        """
        if not text:
            return "en"

        if self.lid_model is not None:
            try:
                labels, scores = self.lid_model.predict(text.replace("\n", " "), k=1)
                label = labels[0].replace("__label__", "")
                iso = FLORES_TO_ISO.get(label, "en")
                print(f"[LID] IndicLID: {label} -> {iso} (conf={scores[0]:.2f})")
                return iso
            except Exception as e:
                print(f"[LID] IndicLID failed: {e} — falling back to script detection")

        for ch in text:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F: return "hi"
            if 0x0980 <= code <= 0x09FF: return "bn"
            if 0x0A00 <= code <= 0x0A7F: return "pa"
            if 0x0A80 <= code <= 0x0AFF: return "gu"
            if 0x0B00 <= code <= 0x0B7F: return "or"
            if 0x0B80 <= code <= 0x0BFF: return "ta"
            if 0x0C00 <= code <= 0x0C7F: return "te"
            if 0x0C80 <= code <= 0x0CFF: return "kn"
            if 0x0D00 <= code <= 0x0D7F: return "ml"
            if 0x0600 <= code <= 0x06FF: return "ur"
        return "en"

    # ---------------- Script helpers ----------------

    def _is_devanagari(self, text: str) -> bool:
        return any(0x0900 <= ord(c) <= 0x097F for c in text)

    def _is_bengali_script(self, text: str) -> bool:
        return any(0x0980 <= ord(c) <= 0x09FF for c in text)

    def _is_arabic_script(self, text: str) -> bool:
        return any(0x0600 <= ord(c) <= 0x06FF for c in text)