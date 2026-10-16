# services/ocr_engine.py
import os
import tempfile
from typing import List, Dict, Any

import fitz  # PyMuPDF
from datalab_sdk import AsyncDatalabClient, ConvertOptions


class DocumentProcessor:
    _client = None

    @classmethod
    async def _get_client(cls):
        if cls._client is None:
            api_key = os.getenv("DATALAB_API_KEY")
            if not api_key:
                raise ValueError("DATALAB_API_KEY not set in environment")
            cls._client = AsyncDatalabClient(api_key=api_key)
            print("[ocr] Datalab async client ready.")
        return cls._client

    @staticmethod
    async def extract_layout_and_text(
        file_bytes: bytes, file_name: str
    ) -> List[Dict[str, Any]]:
        ext = file_name.split(".")[-1].lower()
        filetype = "pdf" if ext == "pdf" else ext

        doc = fitz.open(stream=file_bytes, filetype=filetype)
        num_pages = len(doc)

        # Step 1: Extract pymupdf text layer for every page
        page_texts: List[str] = []
        pages_to_ocr: List[int] = []
        for i in range(num_pages):
            page = doc.load_page(i)
            raw = page.get_text("text").strip()
            page_texts.append(raw)
            if len(raw) < 500:
                pages_to_ocr.append(i)

        doc.close()

        print(f"[ocr] {num_pages} pages total, "
              f"{len(pages_to_ocr)} pages need Chandra OCR: {[p+1 for p in pages_to_ocr]}")

        # Step 2: For thin pages, call Chandra on individual page images
        if pages_to_ocr:
            client = await DocumentProcessor._get_client()
            doc = fitz.open(stream=file_bytes, filetype=filetype)

            for i in pages_to_ocr:
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name

                try:
                    options = ConvertOptions(
                        output_format="markdown",
                        mode="accurate",
                    )
                    result = await client.convert(tmp_path, options=options)
                    if result.success and result.markdown:
                        page_texts[i] = result.markdown.strip()
                        print(f"[ocr] page {i+1}: Chandra extracted "
                              f"{len(result.markdown)} chars")
                    else:
                        print(f"[ocr] page {i+1}: Chandra returned no content")
                except Exception as e:
                    print(f"[ocr] page {i+1}: Chandra failed: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            doc.close()
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:
                pass

        # Step 3: Assemble pages
        parsed_pages = []
        for i, text in enumerate(page_texts):
            parsed_pages.append({
                "page_number": i + 1,
                "text_content": text if text else "[Empty]",
                "is_scanned": False,
                "tables": [],
                "image_count": 0,
                "layout_blocks": [],
            })

        return parsed_pages