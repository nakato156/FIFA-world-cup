#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_pdf_to_md.py input.pdf [output.md]")
        sys.exit(2)
    inp = Path(sys.argv[1])
    if not inp.exists():
        print(f"ERROR: input not found: {inp}")
        sys.exit(3)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_suffix('.md')
    try:
        from PyPDF2 import PdfReader
    except Exception:
        print("MISSING_PYPDF2")
        sys.exit(4)
    reader = PdfReader(str(inp))
    with out.open('w', encoding='utf-8') as f:
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ''
            except Exception:
                text = ''
            # Simple separation between pages
            f.write(text)
            if i != len(reader.pages) - 1:
                f.write('\n\n---\n\n')
    print(f"WROTE:{out}")

if __name__ == '__main__':
    main()
