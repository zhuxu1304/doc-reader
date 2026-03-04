import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


@dataclass
class Chunk:
    source: str
    page: int
    idx: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.source}#p{self.page:03d}-c{self.idx:03d}"


def clean_text(s: str) -> str:
    return " ".join(s.replace("\u00a0", " ").split()).strip()


def chunk_text(text: str, *, max_chars: int = 900, overlap_chars: int = 150) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap_chars)

    return chunks


def iter_pdf_chunks(pdf_path: Path, *, max_chars: int = 900, overlap_chars: int = 150) -> Iterable[Chunk]:
    reader = PdfReader(str(pdf_path))
    source = pdf_path.name

    for page_idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        raw = raw.strip()
        if not raw:
            continue

        parts = chunk_text(raw, max_chars=max_chars, overlap_chars=overlap_chars)
        for i, part in enumerate(parts, start=1):
            yield Chunk(source=source, page=page_idx, idx=i, text=part)


def write_jsonl(chunks: Iterable[Chunk], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            rec = {
                "id": ch.id,
                "source": ch.source,
                "page": ch.page,
                "text": ch.text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_dir", type=str, help="Directory containing PDF files")
    ap.add_argument("--out", type=str, default="data/corpus.jsonl")
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--overlap-chars", type=int, default=150)
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)

    all_chunks = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name}")

        chunks = list(
            iter_pdf_chunks(
                pdf_file,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
        )

        all_chunks.extend(chunks)

    n = write_jsonl(all_chunks, out_path)
    print(f"Wrote {n} chunks to {out_path}")


if __name__ == "__main__":
    main()