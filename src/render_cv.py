"""
Render the tailored CV to HTML (Jinja2 template) and PDF (headless Chrome).

Page 1 holds the summary plus the roles marked page=1 in data/profile.json.
To keep them on one page, fit_page_one() renders page 1 alone and drops the
least JD-relevant bullets (respecting each role's min_bullets) until it fits.
Everything else starts on page 2, followed by education and skills.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape
from pydantic import BaseModel
from pypdf import PdfReader

from cv_profile import Profile

TEMPLATE_DIR = Path(__file__).parent / "templates"
CHROME_PATH = os.getenv("CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


class CVBullet(BaseModel):
    text: str
    highlights: List[str] = []
    relevance: Optional[float] = None  # best rerank score against any JD requirement; None = not retrieved
    achievement_id: str


class CVRole(BaseModel):
    company: str
    role: str
    location: str
    date_range: str
    page: int
    min_bullets: int = 1
    bullets: List[CVBullet]


def _bold_markdown(text: str) -> Markup:
    """Escape text, then turn **x** into <strong>x</strong>."""
    escaped = str(escape(text))
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped))


def _render_bullet(bullet: CVBullet) -> Markup:
    """Escape bullet text, bold its highlight phrases, and prettify arrows."""
    text = re.sub(r"\*\*", "", bullet.text)  # drop stray markdown the LLM may add
    html = str(escape(text))
    for phrase in sorted(bullet.highlights, key=len, reverse=True):
        pattern = re.escape(str(escape(phrase)))
        html = re.sub(pattern, lambda m: f"<strong>{m.group(0)}</strong>", html, count=1, flags=re.IGNORECASE)
    return Markup(html.replace("-&gt;", "→"))


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    env.filters["bold_markdown"] = _bold_markdown
    env.filters["render_bullet"] = _render_bullet
    return env


def render_html(profile: Profile, headline: str, summary: str, roles: List[CVRole], page_one_only: bool = False) -> str:
    template = _environment().get_template("cv.html.j2")
    return template.render(
        profile=profile,
        headline=headline,
        summary=summary,
        roles=[r for r in roles if r.page == 1] if page_one_only else roles,
        page_one_only=page_one_only,
    )


def html_to_pdf(html_path: Path, pdf_path: Path):
    if not Path(CHROME_PATH).exists():
        raise FileNotFoundError(
            f"Chrome not found at {CHROME_PATH}. Install Google Chrome or set CHROME_PATH."
        )
    subprocess.run(
        [
            CHROME_PATH,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=5000",  # give the web font time to load
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def _page_count(html: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "page1.html"
        pdf_path = Path(tmp) / "page1.pdf"
        html_path.write_text(html, encoding="utf-8")
        html_to_pdf(html_path, pdf_path)
        return len(PdfReader(str(pdf_path)).pages)


def fit_page_one(profile: Profile, headline: str, summary: str, roles: List[CVRole]) -> Tuple[List[CVBullet], bool]:
    """Drop the least relevant page-1 bullets until page 1 fits. Returns (dropped bullets, fits)."""
    dropped: List[CVBullet] = []
    while _page_count(render_html(profile, headline, summary, roles, page_one_only=True)) > 1:
        candidates = [
            (role, bullet)
            for role in roles
            if role.page == 1 and len(role.bullets) > role.min_bullets
            for bullet in role.bullets
        ]
        if not candidates:
            return dropped, False  # every role is at its minimum and page 1 still overflows
        role, bullet = min(
            candidates,
            key=lambda rb: rb[1].relevance if rb[1].relevance is not None else float("-inf"),
        )
        role.bullets.remove(bullet)
        dropped.append(bullet)
    return dropped, True


def write_cv(profile: Profile, headline: str, summary: str, roles: List[CVRole], pdf_path: Path) -> Path:
    """Write <name>.html and <name>.pdf side by side; returns the HTML path."""
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(render_html(profile, headline, summary, roles), encoding="utf-8")
    html_to_pdf(html_path, pdf_path)
    return html_path
