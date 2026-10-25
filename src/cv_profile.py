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
    # JD role categories (see classify_jd.ROLE_CATEGORIES) this role is relevant
    # to. For any other JD, a page-1 role is cut to min_bullets. Empty = always relevant.
    categories: List[str] = []
    # JD role categories this role is the core of: it is filled to max_bullets and
    # page-1 fitting trims the other roles first. Must be a subset of categories.
    full_categories: List[str] = []


class EducationEntry(BaseModel):
    degree: str
    school: str
    date_range: str
    thesis: Optional[str] = None


class PersonalProject(BaseModel):
    name: str
    subtitle: str
    date_range: Optional[str] = None
    description: str


class SkillSet(BaseModel):
    categories: List[str]  # JD role categories (see classify_jd.ROLE_CATEGORIES) this set is used for
    skills: List[str]


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
    # Years of (data) experience, e.g. "5+"; given to the summary writer.
    years_of_experience: str
    # The industry the candidate worked in, in plain words; given to the summary writer.
    domain: str = ""
    # Languages spoken fluently; given to the summary writer.
    languages: List[str] = []
    # JD term -> phrases in the achievements that honestly back it, for terms the
    # material means but never spells out ("test automation" <- "integration tests").
    term_evidence: Dict[str, List[str]] = {}
    experience: List[ExperienceEntry]
    education: List[EducationEntry]
    personal_projects: List[PersonalProject] = []  # page 2, after work experience; not retrieved or rewritten
    # Skill set name -> skills + the JD role categories it is used for.
    skill_sets: Dict[str, SkillSet]
    default_skill_set: str  # used when no set lists the JD's role category
    skills: List[str] = []  # the set chosen for this JD; filled by with_skills()
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


def skill_sets_for(profile: Profile, role_category: str) -> List[str]:
    """Names of every skill set listing the JD's role category, else just the default set."""
    names = [name for name, s in profile.skill_sets.items() if role_category in s.categories]
    return names or [profile.default_skill_set]


def with_skills(profile: Profile, role_category: str) -> Profile:
    """Copy of the profile whose skills are the chosen sets combined (e.g. AI Data Engineer),
    deduplicated case-insensitively and sorted alphabetically."""
    merged: Dict[str, str] = {}
    for name in skill_sets_for(profile, role_category):
        for skill in profile.skill_sets[name].skills:
            merged.setdefault(skill.lower(), skill)
    return profile.model_copy(update={"skills": sorted(merged.values(), key=str.lower)})


def load_profile(path: str = "data/profile.json") -> Profile:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return Profile(**json.load(f))


if __name__ == "__main__":
    profile = load_profile()
    print(f"{profile.name}: {len(profile.experience)} roles, {len(profile.education)} degrees")
