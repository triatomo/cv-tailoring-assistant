"""
Load the static parts of the CV (header, experience layout, education,
skills) from data/profile.json. Achievements hold the bullets; the profile
decides which roles exist, in what order, and which page each role lives on.
"""
import json
from pathlib import Path
from typing import List, Optional
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


class Profile(BaseModel):
    name: str
    contact: List[str]
    default_headline: str
    default_summary: str
    experience: List[ExperienceEntry]
    education: List[EducationEntry]
    skills: List[str]
    hobbies: Optional[str] = None


def load_profile(path: str = "data/profile.json") -> Profile:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return Profile(**json.load(f))


if __name__ == "__main__":
    profile = load_profile()
    print(f"{profile.name}: {len(profile.experience)} roles, {len(profile.education)} degrees")
