from pathlib import Path

import pandas as pd
import requests
import yaml

CURRENT_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.yaml"
HISTORICAL_URL = "https://unitedstates.github.io/congress-legislators/legislators-historical.yaml"

OUTPUT_PATHS = [
    "datastore/inference/congress_legislators.csv",
    "datastore/inference/daily_harmonized/congress_legislators.csv",
]


def read_yaml(url: str) -> list[dict]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.content.decode("utf-8", errors="replace")
    return yaml.safe_load(text)


def congress_from_year(year: int) -> int:
    return ((year - 1789) // 2) + 1


def get_term_congress_range(term: dict) -> tuple[int, int] | None:
    start_date = term.get("start")
    end_date = term.get("end")

    if not start_date:
        return None

    start_year = int(start_date.split("-")[0])
    start_congress = congress_from_year(start_year)

    if not end_date:
        return start_congress, start_congress

    end_year = int(end_date.split("-")[0])
    end_congress = ((end_year - 1789) // 2) if end_year > start_year else start_congress

    return start_congress, end_congress


def normalize_chamber(term_type: str | None) -> str | None:
    chamber_map = {
        "rep": "house",
        "sen": "senate",
    }
    return chamber_map.get(term_type, term_type)


def make_district_code(state_abbrev: str | None, district: int | None) -> str | None:
    if not state_abbrev:
        return None
    if district is None:
        return state_abbrev
    return f"{state_abbrev}{district:02d}"


def congress_date_range(congress: int) -> tuple[str, str]:
    start_year = 1789 + (congress - 1) * 2
    end_year = start_year + 1
    return f"{start_year}-01-03", f"{end_year}-12-31"


def leadership_roles_for_congress(
    leadership_roles: list[dict],
    chamber: str | None,
    congress: int,
) -> str:
    congress_start_date, congress_end_date = congress_date_range(congress)
    titles = []

    for role in leadership_roles:
        role_start = role.get("start")
        role_end = role.get("end") or "9999-12-31"
        role_chamber = role.get("chamber")

        overlaps_congress = role_start and role_start <= congress_end_date and role_end >= congress_start_date
        chamber_matches = role_chamber == chamber or role_chamber is None

        if overlaps_congress and chamber_matches:
            titles.append(role.get("title", ""))

    return "/".join(titles)


def extract_legislator_metadata(legislator: dict) -> dict:
    name = legislator.get("name", {})
    bio = legislator.get("bio", {})
    ids = legislator.get("id", {})

    return {
        "first_name": name.get("first"),
        "last_name": name.get("last"),
        "nickname": name.get("nickname"),
        "icpsr": ids.get("icpsr"),
        "gender": bio.get("gender"),
    }


def build_row(
    metadata: dict,
    chamber: str | None,
    congress: int,
    state_abbrev: str | None,
    district_code: str | None,
    leadership_roles: str,
) -> dict:
    return {
        "first_name": metadata["first_name"],
        "last_name": metadata["last_name"],
        "nickname": metadata["nickname"],
        "chamber": chamber,
        "congress": congress,
        "icpsr": metadata["icpsr"],
        "district_code": district_code,
        "state_abbrev": state_abbrev,
        "leadership_roles": leadership_roles,
        "gender": metadata["gender"],
    }


def process_legislator(legislator: dict) -> list[dict]:
    rows = []
    metadata = extract_legislator_metadata(legislator)
    leadership_roles = legislator.get("leadership_roles", [])

    for term in legislator.get("terms", []):
        congress_range = get_term_congress_range(term)
        if congress_range is None:
            continue

        start_congress, end_congress = congress_range
        chamber = normalize_chamber(term.get("type"))
        state_abbrev = term.get("state")
        district = term.get("district")
        district_code = make_district_code(state_abbrev, district)

        for congress in range(start_congress, end_congress + 1):
            leadership_str = leadership_roles_for_congress(
                leadership_roles=leadership_roles,
                chamber=chamber,
                congress=congress,
            )
            rows.append(
                build_row(
                    metadata=metadata,
                    chamber=chamber,
                    congress=congress,
                    state_abbrev=state_abbrev,
                    district_code=district_code,
                    leadership_roles=leadership_str,
                )
            )

    return rows


def process_legislators(data: list[dict]) -> pd.DataFrame:
    rows = []
    for legislator in data:
        rows.extend(process_legislator(legislator))
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, output_paths: list[str]) -> None:
    for path_str in output_paths:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def main() -> None:
    current_data = read_yaml(CURRENT_URL)
    historical_data = read_yaml(HISTORICAL_URL)

    current_df = process_legislators(current_data)
    historical_df = process_legislators(historical_data)

    all_df = pd.concat([current_df, historical_df], ignore_index=True)
    save_dataframe(all_df, OUTPUT_PATHS)


if __name__ == "__main__":
    main()
