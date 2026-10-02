"""
Load the static parts of the CV (header, experience layout, education,
skills) from data/profile.json. Achievements hold the bullets; the profile
decides which roles exist, in what order, and which page each role lives on.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel


class ExperienceEntry(BaseModel):
    company: str
    role: str
    location: str
    date_range: str
    page: int = 2  # 1 = tailored + fitted onto the first page; 2 = everything after
    min_bullets: int = 1
    max_bullets: Optional[int] = None  # None = include all bullets for this role


class EducationEntry(BaseModel):
    degree: str
    school: str
    date_range: str
    thesis: Optional[str] = None


class RelocationTarget(BaseModel):
    note: str  # contact-line text used when the JD is in this country
    keywords: List[str]  # country/city names that place the JD in this country


class Profile(BaseModel):
    name: str
    contact: List[str]
    # Country -> contact-line note + location keywords to look for in the JD.
    # These are the only countries the CV will ever claim relocation to.
    relocation: Dict[str, RelocationTarget] = {}
    default_location: Optional[str] = None  # used when the JD names none of the relocation countries
    default_headline: str
    default_summary: str
    # Fixed first/last sentences of every tailored summary; the LLM only writes
    # what goes between them, so the seniority framing never drifts.
    summary_opening: str
    summary_closing: str = ""
    experience: List[ExperienceEntry]
    education: List[EducationEntry]
    skills: List[str]
    hobbies: Optional[str] = None


def _first_mention(target: RelocationTarget, jd_text: str) -> Optional[int]:
    positions = [
        m.start()
        for k in target.keywords
        if (m := re.search(rf"\b{re.escape(k)}\b", jd_text, re.IGNORECASE))
    ]
    return min(positions) if positions else None


def location_note(profile: Profile, jd_text: str) -> Optional[str]:
    """Note for the relocation country the JD mentions first, else the default location."""
    mentions = [
        (pos, target.note)
        for target in profile.relocation.values()
        if (pos := _first_mention(target, jd_text)) is not None
    ]
    return min(mentions)[1] if mentions else profile.default_location


def with_relocation(profile: Profile, jd_text: str) -> Profile:
    """Copy of the profile whose contact line ends with the JD-specific location note."""
    note = location_note(profile, jd_text)
    if not note:
        return profile
    return profile.model_copy(update={"contact": profile.contact + [note]})


def load_profile(path: str = "data/profile.json") -> Profile:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return Profile(**json.load(f))


if __name__ == "__main__":
    profile = load_profile()
    print(f"{profile.name}: {len(profile.experience)} roles, {len(profile.education)} degrees")
