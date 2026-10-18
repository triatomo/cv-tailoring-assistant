"""
Entry point: takes a JD (text file, PDF file, or pasted text), runs the full
pipeline, and outputs:
  - a tailored CV as PDF + HTML (per-JD summary, page-1 roles filled with the
    most relevant rewritten bullets and fitted to one page, then the remaining
    roles, education, and skills on page 2), and
  - review notes as markdown (rewrites, anything dropped to fit, honest gaps).

Usage:
    python src/cli.py --jd path/to/jd.txt
    python src/cli.py --jd path/to/jd.pdf
    python src/cli.py --jd-text "paste the JD text directly"
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from classify_jd import classify_jd
from retrieve import retrieve_for_requirement, score_against_jd
from rewrite import rewrite_achievement
from verify import verify_rewrite, get_final_bullet
from gaps import assess_gap, MATCH_THRESHOLD
from embed import EMBEDDING_MODEL
from ingest import load_achievements
from cv_profile import Profile, load_profile, skill_sets_for, with_relocation, with_skills
from summary import generate_summary
from render_cv import CVBullet, CVRole, fit_page_one, write_cv

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


class PipelineResult(NamedTuple):
    review_md: str
    profile: Profile
    headline: str
    summary: str
    roles: List[CVRole]


def run_pipeline(jd_text: str, top_k: int = 3) -> PipelineResult:
    print("Loading embedding model...", file=sys.stderr)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Classifying JD...", file=sys.stderr)
    classification = classify_jd(jd_text)

    profile = with_skills(with_relocation(load_profile(), jd_text), classification.role_category)
    achievements = load_achievements()

    # For each achievement, remember the JD requirement it matched best, so it
    # can be ranked for page 1 and rewritten toward that requirement.
    best_match: Dict[str, Tuple[float, str]] = {}
    gaps_section = ["## Gaps", ""]

    for requirement in classification.key_requirements:
        print(f"Job requirement: {requirement}", file=sys.stderr)
        retrieved = retrieve_for_requirement(requirement, classification, embedding_model, top_k=top_k)

        gap = assess_gap(requirement, retrieved)
        if not gap.has_strong_match:
            gaps_section.append(f"- **{requirement}**: {gap.gap_statement}")
            continue

        for candidate in retrieved:
            if candidate.score < MATCH_THRESHOLD:
                continue
            current = best_match.get(candidate.achievement.id)
            if current is None or candidate.score > current[0]:
                best_match[candidate.achievement.id] = (candidate.score, requirement)

    # Score every page-1 achievement against the JD, not just the ones that made
    # a requirement's top-k, so each page-1 role keeps its most relevant bullets.
    page_one_roles = {(e.company, e.role) for e in profile.experience if e.page == 1}
    relevance = score_against_jd(
        classification.key_requirements,
        [a for a in achievements if (a.company, a.role) in page_one_roles],
    )

    rewrite_notes = ["## Rewrites", ""]
    roles: List[CVRole] = []
    for entry in profile.experience:
        role_achievements = [a for a in achievements if a.company == entry.company and a.role == entry.role]
        limit = entry.max_bullets
        if entry.page == 1:
            role_achievements.sort(key=lambda a: relevance.get(a.id, float("-inf")), reverse=True)
            if entry.categories and classification.role_category not in entry.categories:
                limit = entry.min_bullets  # role is off-target for this JD: keep it short
        if limit is not None:
            role_achievements = role_achievements[:limit]

        bullets = []
        for achievement in role_achievements:
            text = achievement.text
            match = best_match.get(achievement.id)
            if match is not None:
                print(f"Rewriting {achievement.id} toward: {match[1]}", file=sys.stderr)
                rewrite_result = rewrite_achievement(
                    achievement,
                    requirement=match[1],
                    terminology=classification.key_terminology,
                )
                verification = verify_rewrite(rewrite_result)
                text = get_final_bullet(rewrite_result, verification)
                rewrite_notes.append(f"- `{achievement.id}` ({entry.role}) toward *{match[1]}*: {text}")
                if not verification.is_supported:
                    rewrite_notes.append(
                        f"  - *(Note: rewrite was reverted to original -- flagged issues: "
                        f"{'; '.join(verification.issues)})*"
                    )
            bullets.append(
                CVBullet(
                    text=text,
                    highlights=achievement.highlights,
                    relevance=relevance.get(achievement.id),
                    achievement_id=achievement.id,
                )
            )
        roles.append(
            CVRole(
                company=entry.company,
                role=entry.role,
                location=entry.location,
                date_range=entry.date_range,
                page=entry.page,
                min_bullets=entry.min_bullets,
                bullets=bullets,
            )
        )

    print("Writing summary...", file=sys.stderr)
    cv_summary = generate_summary(
        classification,
        profile,
        achievements,
        matched_requirements={achievement_id: req for achievement_id, (_, req) in best_match.items()},
    )

    print("Fitting page 1...", file=sys.stderr)
    dropped, fits = fit_page_one(profile, cv_summary.headline, cv_summary.summary, roles)

    output_lines = [
        f"# Tailored CV Draft -- Review Notes",
        f"",
        f"**Detected role category:** {classification.role_category}",
        f"",
        f"**Key terminology to mirror:** {', '.join(classification.key_terminology)}",
        f"",
        f"**Contact line:** {' • '.join(profile.contact)}",
        f"",
        f"**Skill set:** {' + '.join(skill_sets_for(profile, classification.role_category))}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"**{cv_summary.headline}**",
        f"",
        cv_summary.summary,
        f"",
    ]
    if not cv_summary.is_generated:
        output_lines.append(
            f"*(Note: using the default summary from data/profile.json -- "
            f"{'; '.join(cv_summary.issues)})*"
        )
        output_lines.append("")
    if not fits:
        output_lines.append(
            "**Warning:** page 1 still overflows with every page-1 role at its min_bullets -- "
            "shorten the summary or lower min_bullets in data/profile.json."
        )
        output_lines.append("")
    output_lines.extend(rewrite_notes)
    output_lines.append("")
    if dropped:
        output_lines.extend(["## Dropped to fit page 1", ""])
        output_lines.extend(f"- `{b.achievement_id}`: {b.text}" for b in dropped)
        output_lines.append("")
    output_lines.extend(gaps_section)

    return PipelineResult(
        review_md="\n".join(output_lines),
        profile=profile,
        headline=cv_summary.headline,
        summary=cv_summary.summary,
        roles=roles,
    )


def main():
    parser = argparse.ArgumentParser(description="Tailor CV bullets to a job description.")
    parser.add_argument("--jd", type=str, help="Path to a JD text file.")
    parser.add_argument("--jd-text", type=str, help="JD text pasted directly.")
    parser.add_argument("--top-k", type=int, default=3, help="Candidates to retrieve per requirement.")
    parser.add_argument("--out", type=str, default=None, help="Optional output markdown file path (review notes).")
    parser.add_argument(
        "--cv",
        type=str,
        default=None,
        help="Output CV PDF path (an .html is written next to it). Defaults to --out with a .pdf suffix, else tailored_cv.pdf.",
    )
    args = parser.parse_args()

    if not args.jd and not args.jd_text:
        print("Error: provide either --jd <file> or --jd-text \"<text>\"", file=sys.stderr)
        sys.exit(1)

    jd_text = read_jd_text(args)
    result = run_pipeline(jd_text, top_k=args.top_k)

    if args.out:
        Path(args.out).write_text(result.review_md, encoding="utf-8")
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(result.review_md)

    if args.cv:
        cv_path = Path(args.cv)
    elif args.out:
        cv_path = Path(args.out).with_suffix(".pdf")
    else:
        cv_path = Path("tailored_cv.pdf")
    html_path = write_cv(result.profile, result.headline, result.summary, result.roles, cv_path)
    print(f"CV written to {cv_path} (and {html_path})", file=sys.stderr)


if __name__ == "__main__":
    main()
