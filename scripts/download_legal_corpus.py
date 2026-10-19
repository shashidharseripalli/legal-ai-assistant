"""Download official Indian criminal-law PDFs into data/raw/.

The source URLs are linked from the Ministry of Home Affairs' "New Criminal
Laws" page. Run this before scripts/ingest_documents.py.
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_PAGES = [
    "https://www.mha.gov.in/en/commoncontent/new-criminal-laws",
    "https://www.legislative.gov.in/document/constitution-of-india-in-english",
]
OFFICIAL_PDFS = {
    "constitution_of_india.pdf": "https://www.legislative.gov.in/static/uploads/2025/07/359f70a69695affb9d72f8393102bd2e.pdf",
    "bns.pdf": "https://www.mha.gov.in/sites/default/files/2024-04/250883_english_01042024.pdf",
    "bnss.pdf": "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
    "bsa.pdf": "https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
}
DATA_DIR = Path("data/raw")


def download_pdf(filename: str, url: str, *, force: bool = False) -> Path:
    destination = DATA_DIR / filename
    if destination.exists() and not force:
        print(f"Skipping {filename} (already exists; use --force to replace)")
        return destination

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Legal-AI-Assistant corpus downloader"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP status {response.status}")
            content_type = response.headers.get_content_type()
            if content_type not in {"application/pdf", "application/octet-stream"}:
                raise RuntimeError(f"Expected a PDF, got Content-Type {content_type!r}")

            first_chunk = response.read(1024 * 1024)
            if not first_chunk.startswith(b"%PDF-"):
                raise RuntimeError("Downloaded response does not have a PDF signature")

            with temporary.open("wb") as output:
                output.write(first_chunk)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        temporary.replace(destination)
        print(f"Downloaded {filename} ({destination.stat().st_size:,} bytes)")
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing PDFs instead of skipping them",
    )
    args = parser.parse_args()

    print("Downloading official PDFs from these government sources:")
    for source_page in SOURCE_PAGES:
        print(f"  - {source_page}")
    failures = []
    for filename, url in OFFICIAL_PDFS.items():
        try:
            download_pdf(filename, url, force=args.force)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            failures.append((filename, exc))
            print(f"ERROR: {filename}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"{len(failures)} download(s) failed. Check your internet connection "
            "or download the PDFs manually from the official source page.",
            file=sys.stderr,
        )
        return 1

    print(
        "\nNext: add constitution_of_india.pdf and any other authoritative "
        "documents required for your use case, then run "
        "'python scripts/ingest_documents.py'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
