from pydantic import BaseModel, Field
from typing import List, Optional

class ContactInfo(BaseModel):
    email: Optional[str]
    phone: Optional[str]
    linkedin: Optional[str] = None

class WorkExperience(BaseModel):
    company: str
    job_title: str
    dates: str
    responsibilities: Optional[str]

class Education(BaseModel):
    degree: Optional[str]
    institution: Optional[str]
    dates: Optional[str]
    additional_info: Optional[str]

class CandidateProfile(BaseModel):
    full_name: str
    contact_info: ContactInfo  
    summary: Optional[str] = None
    skills: List[str] = []
    work_experience: List[WorkExperience] = []
    education: List[Education] = []
    certifications: Optional[List[str]] = []
    projects: Optional[List[str]] = []
    languages: Optional[List[str]] = []
