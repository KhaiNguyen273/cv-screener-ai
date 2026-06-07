from dataclasses import dataclass


@dataclass
class Job:
    id: str
    title: str
    url: str
    company: str
    company_url: str
    salary: str
    location: str
    experience: str
    tags: list[str]
    time_posted: str
    image_url: str

    detail: str = ""  # markdown detail

    def toMarkdown(self):
        md = f"# {self.title}\n\n"
        md += f"**Company:** [{self.company}]({self.company_url})\n\n"
        md += f"**Location:** {self.location}\n\n"
        md += f"**Salary:** {self.salary}\n\n"
        md += f"**Experience:** {self.experience}\n\n"
        md += f"**Tags:** {', '.join(self.tags)}\n\n"
        md += f"**Posted:** {self.time_posted}\n\n"
        md += f"![Company Logo]({self.image_url})\n\n"
        md += f"## Job Description\n\n"
        md += self.detail
        return md

    def toJson(self):
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "company": self.company,
            "company_url": self.company_url,
            "salary": self.salary,
            "location": self.location,
            "experience": self.experience,
            "tags": self.tags,
            "time_posted": self.time_posted,
            "image_url": self.image_url,
            "detail": self.detail,
        }
