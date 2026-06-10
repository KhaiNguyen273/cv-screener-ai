from .db import get_connection


def load_skills():
    conn = get_connection()

    try:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                c.name AS category,
                s.skill_name
            FROM skills s
            JOIN skill_categories c
                ON c.id = s.category_id
        """)

        rows = cursor.fetchall()

        skills_by_category = {}

        for row in rows:
            category = row["category"]
            skill = row["skill_name"].lower()

            skills_by_category.setdefault(
                category,
                set()
            ).add(skill)

        return skills_by_category

    finally:
        cursor.close()
        conn.close()


def load_aliases():
    conn = get_connection()

    try:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                s.skill_name,
                a.alias_name
            FROM skill_aliases a
            JOIN skills s
                ON s.id = a.skill_id
        """)

        rows = cursor.fetchall()

        aliases = {}

        for row in rows:
            canonical = row["skill_name"].lower()
            alias = row["alias_name"].lower()

            aliases.setdefault(
                canonical,
                set()
            ).add(alias)

            # thêm canonical vào luôn
            aliases[canonical].add(canonical)

        return {
            skill: list(aliases_set)
            for skill, aliases_set in aliases.items()
        }

    finally:
        cursor.close()
        conn.close()