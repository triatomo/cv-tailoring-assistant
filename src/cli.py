"""
Entry point: takes a JD (text file, PDF file, or pasted text), runs the full
pipeline, and outputs a structured tailored draft (matched + rewritten bullets
grouped by requirement, plus an honest gap list) as markdown.

Usage:
    python src/cli.py --jd path/to/jd.txt
    python src/cli.py --jd path/to/jd.pdf
    python src/cli.py --jd-text "paste the JD text directly"
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from classify_jd import classify_jd
from retrieve import retrieve_for_requirement
from rewrite import rewrite_achievement
from verify import verify_rewrite, get_final_bullet
from gaps import assess_gap
from embed import EMBEDDING_MODEL

load_dotenv()


def _read_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(
            "Error: reading PDF files requires the 'pypdf' package. "
            "Install it with: pip install pypdf",
            file=sys.stderr,
        )
        sys.exit(1)

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        print(
            f"Warning: no extractable text found in {path}. "
            "The PDF may be scanned/image-based rather than text-based -- "
            "in that case, paste the JD text manually via --jd-text instead.",
            file=sys.stderr,
        )
    return text


def read_jd_text(args) -> str:
    if args.jd:
        path = Path(args.jd)
        if not path.exists():
            print(f"Error: JD file not found: {args.jd}", file=sys.stderr)
            sys.exit(1)
        if path.suffix.lower() == ".pdf":
            return _read_pdf_text(path)
        return path.read_text(encoding="utf-8")
    return args.jd_text


def run_pipeline(jd_text: str, top_k: int = 3) -> str:
    print("Loading embedding model...", file=sys.stderr)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Classifying JD...", file=sys.stderr)
    classification = classify_jd(jd_text)

    output_lines = [
        f"# Tailored CV Draft",
        f"",
        f"**Detected role category:** {classification.role_category}",
        f"",
        f"**Key terminology to mirror:** {', '.join(classification.key_terminology)}",
        f"",
        f"---",
        f"",
    ]

    matched_section = ["## Matched & Rewritten Bullets", ""]
    gaps_section = ["## Gaps", ""]

    for requirement in classification.key_requirements:
        print(f"Processing requirement: {requirement}", file=sys.stderr)
        retrieved = retrieve_for_requirement(requirement, classification, embedding_model, top_k=top_k)

        gap = assess_gap(requirement, retrieved)
        if not gap.has_strong_match:
            gaps_section.append(f"- **{requirement}**: {gap.gap_statement}")
            continue

        matched_section.append(f"### Requirement: {requirement}")
        for candidate in retrieved[:2]:  # show top 2 candidates per requirement
            rewrite_result = rewrite_achievement(
                candidate.achievement,
                requirement=requirement,
                terminology=classification.key_terminology,
            )
            verification = verify_rewrite(rewrite_result)
            final_text = get_final_bullet(rewrite_result, verification)

            matched_section.append(f"- {final_text}")
            if not verification.is_supported:
                matched_section.append(
                    f"  - *(Note: rewrite was reverted to original -- flagged issues: "
                    f"{'; '.join(verification.issues)})*"
                )
        matched_section.append("")

    output_lines.extend(matched_section)
    output_lines.append("")
    output_lines.extend(gaps_section)

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="Tailor CV bullets to a job description.")
    parser.add_argument("--jd", type=str, help="Path to a JD text file.")
    parser.add_argument("--jd-text", type=str, help="JD text pasted directly.")
    parser.add_argument("--top-k", type=int, default=3, help="Candidates to retrieve per requirement.")
    parser.add_argument("--out", type=str, default=None, help="Optional output markdown file path.")
    args = parser.parse_args()

    if not args.jd and not args.jd_text:
        print("Error: provide either --jd <file> or --jd-text \"<text>\"", file=sys.stderr)
        sys.exit(1)

    jd_text = read_jd_text(args)
    result_md = run_pipeline(jd_text, top_k=args.top_k)

    if args.out:
        Path(args.out).write_text(result_md, encoding="utf-8")
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(result_md)


if __name__ == "__main__":
    main()
