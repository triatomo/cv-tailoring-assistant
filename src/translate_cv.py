"""
Translate the finished English CV (summary, bullets, contact note, education,
projects, hobbies, section labels) into another language with the LLM.

Tailoring and every honesty check run on English text; translation is the last
content step, before page-1 fitting (German runs longer than English). Names,
tools, numbers, and bold phrases must come through unchanged -- checked in code,
since a small model translating "$1M" or "Airflow" freely changes the claim. A
text whose translation keeps failing the checks stays in English and is
reported, rather than shipping a translation that says something else.
"""
import re
from typing import Dict, List, Optional, Tuple

from cv_profile import Profile
from llm_client import call_llm
from render_cv import CVBullet, CVRole

LANGUAGES = {"de": "German"}

# How a CV bullet ("Built X ...") is phrased in each language, so every bullet
# follows one pattern instead of mixing styles.
STYLE = {
    "de": """  - A work-experience bullet starting with a past-tense verb ("Built ...", "Reduced ...")
    becomes German nominal style starting with the action noun: "Aufbau ...",
    "Entwicklung ...", "Senkung ...", "Mitgestaltung ..." (for "Co-designed").
    Never start with a bare participle ("Entwickelt ein ...", "Implementiert ...").
  - Summary sentences ("Skilled at ...", "Experienced in ...") become fluent German
    ("Versiert in ...", "Erfahren in ..."); "ambiguous problems" are "unklare Problemstellungen".
  - Use common German tech anglicisms where German recruiters use them (Pipeline,
    Stakeholder, Deployment), but translate ordinary words.""",
}

# Fixed template text, translated by hand rather than by the LLM.
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "html_lang": "en", "work_experience": "Work Experience", "education": "Education",
        "personal_projects": "Personal Projects", "skills": "Skills", "hobbies": "Hobbies",
        "thesis": "Thesis", "present": "Present",
    },
    "de": {
        "html_lang": "de", "work_experience": "Berufserfahrung", "education": "Ausbildung",
        "personal_projects": "Persönliche Projekte", "skills": "Kenntnisse", "hobbies": "Hobbys",
        "thesis": "Abschlussarbeit", "present": "heute",
    },
}

TRANSLATE_PROMPT = """Translate this text from a CV from English into {language}.

Rules:
- Write natural, professional {language} CV style, as a native recruiter would expect.
{style}
- Keep these exactly as written, untranslated: names of tools, technologies, products,
  and companies (e.g. {keep_examples}), acronyms, and job titles.
- Keep every number, metric, percentage, and currency amount exactly as written
  (e.g. "$1M", "75%", "p50: 33ms").
- Keep **double asterisks** around the same phrases (translate the phrase inside).
  Never add ** where the text has none.
- Do not add, drop, soften, or strengthen any information.
- Reply with only the translation, no quotes or notes.

Text:
{text}
"""

MAX_ATTEMPTS = 2
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 0.6, 2.0


def _numbers(text: str) -> List[str]:
    return sorted(re.findall(r"\d+(?:[.,]\d+)?", text))


def _bold(text: str) -> List[str]:
    return re.findall(r"\*\*(.+?)\*\*", text)


def _check_translation(source: str, translation: str, tools: List[str]) -> List[str]:
    issues = []
    if _numbers(source) != _numbers(translation):
        issues.append(f"numbers changed: {_numbers(source)} -> {_numbers(translation)}")
    if len(_bold(source)) != len(_bold(translation)):
        issues.append(f"bold phrases changed: {len(_bold(source))} -> {len(_bold(translation))}")
    for tool in tools:
        # A hyphen after the name is fine: German compounds ("MLOps-Pipelines").
        pattern = rf"(?<![\w-]){re.escape(tool)}(?!\w)"
        if re.search(pattern, source, re.IGNORECASE) and not re.search(pattern, translation, re.IGNORECASE):
            issues.append(f"tool name '{tool}' was changed or dropped")
    ratio = len(translation) / max(1, len(source))
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        issues.append(f"length changed too much ({ratio:.1f}x)")
    return issues


class Translator:
    def __init__(self, lang: str, profile: Profile):
        self.lang = lang
        self.language = LANGUAGES[lang]
        self.tools = sorted(
            {re.sub(r"^Apache\s+", "", skill) for s in profile.skill_sets.values() for skill in s.skills},
            key=str.lower,
        )
        self.cache: Dict[str, str] = {}
        self.failures: List[str] = []  # texts kept in English, with why

    def __call__(self, text: Optional[str]) -> Optional[str]:
        if not text or not text.strip():
            return text
        if text in self.cache:
            return self.cache[text]
        prompt = TRANSLATE_PROMPT.format(
            language=self.language, style=STYLE[self.lang], keep_examples=", ".join(self.tools[:8]), text=text,
        )
        issues: List[str] = []
        for _ in range(MAX_ATTEMPTS):
            attempt = prompt
            if issues:
                attempt += "\nA previous translation was rejected -- fix these problems:\n" + "\n".join(f"- {i}" for i in issues)
            translation = call_llm(attempt, max_tokens=1024).strip().strip('"')
            if not _bold(text):
                translation = translation.replace("**", "")  # bold the source never had
            issues = _check_translation(text, translation, self.tools)
            if not issues:
                self.cache[text] = translation
                return translation
        self.failures.append(f"{text[:80]}... ({'; '.join(issues)})")
        self.cache[text] = text
        return text


def _translate_bullet(bullet: CVBullet, translate: Translator) -> CVBullet:
    """Highlights are English phrases bolded at render time; mark them with ** so the
    translation carries them over, then turn the translated ** phrases back into highlights."""
    marked = bullet.text
    for phrase in sorted(bullet.highlights, key=len, reverse=True):
        marked = re.sub(re.escape(phrase), lambda m: f"**{m.group(0)}**", marked, count=1, flags=re.IGNORECASE)
    translated = translate(marked)
    return bullet.model_copy(update={"text": re.sub(r"\*\*", "", translated), "highlights": _bold(translated)})


def translate_cv(
    lang: str, profile: Profile, headline: str, summary: str, roles: List[CVRole],
) -> Tuple[Profile, str, List[CVRole], List[str]]:
    """Returns (profile, summary, roles, failures) in `lang`. The headline is the JD's
    own role title and job titles stay as held, so neither is translated."""
    translate = Translator(lang, profile)
    present = LABELS[lang]["present"]
    roles = [
        role.model_copy(update={
            "date_range": role.date_range.replace("Present", present),
            "bullets": [_translate_bullet(b, translate) for b in role.bullets],
        })
        for role in roles
    ]
    summary = translate(summary)
    contact = [translate(c) if not re.search(r"[@.]\w", c) else c for c in profile.contact]
    education = [
        e.model_copy(update={"degree": translate(e.degree)})  # thesis titles stay as written
        for e in profile.education
    ]
    projects = [
        p.model_copy(update={
            "subtitle": translate(p.subtitle),
            "description": translate(p.description),
            "date_range": p.date_range.replace("Present", present) if p.date_range else p.date_range,
        })
        for p in profile.personal_projects
    ]
    profile = profile.model_copy(update={
        "contact": contact,
        "education": education,
        "personal_projects": projects,
        "hobbies": translate(profile.hobbies),
    })
    return profile, summary, roles, translate.failures
