#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd



PG_RE = re.compile(r"Pg([A-Z])(\d+)(?:-(\d+))?$")          # from id stem tail: PgH4437-2
FILE_RE = re.compile(r"Pg([A-Z])(\d+)(?:-(\d+))?\.json$")  # from filename


def parse_pg_from_stem(stem: str) -> Tuple[str, int, int]:
    """
    stem like 'CREC-2022-04-11-pt1-PgH4438-10' -> ('H', 4438, 10)
    """
    m = re.search(r"Pg[A-Z]\d+(?:-\d+)?$", stem)
    if not m:
        return ("", 0, 0)
    tail = m.group(0)  # e.g. PgH4438-10

    m2 = PG_RE.match(tail)
    if not m2:
        return ("", 0, 0)

    letter = m2.group(1)
    page = int(m2.group(2))
    seg = int(m2.group(3)) if m2.group(3) else 0
    return (letter, page, seg)


def sort_key_for_json_path(p: Path, prefix_rank: Dict[str, int]) -> tuple:
    m = FILE_RE.search(p.name)
    if not m:
        return (99, 0, 0, p.name)
    letter = m.group(1)
    page = int(m.group(2))
    seg = int(m.group(3)) if m.group(3) else 0
    return (prefix_rank.get(letter, 99), page, seg, p.name)


def normalize_speaker(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return "" if s.lower() == "none" else s


def strip_leading_speaker_label(text: str, speaker: str) -> str:
    """
    Daily 'speech' text often begins with e.g. 'Mr. DURBIN.' which duplicates `speaker`.
    Historical pipeline kept speaker separate, so we remove the leading label when it matches.
    """
    if not text:
        return ""
    if not speaker:
        return text

    # Remove exact speaker label at start, with optional period
    pat = r"^\s*" + re.escape(speaker) + r"\s*\.?\s*"
    return re.sub(pat, "", text, count=1)


def split_paragraphs(text: str) -> List[str]:
    """
    Convert daily text into "paragraph_text" rows.
    Heuristic: split on blank lines; collapse single newlines to spaces inside paragraphs.
    """
    if not text:
        return []
    t = text.replace("\r", "")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []
    parts = re.split(r"\n\s*\n+", t)
    out = []
    for p in parts:
        p = re.sub(r"\s*\n\s*", " ", p).strip()
        if p:
            out.append(p)
    return out


def split_title_heading_body(text: str) -> Tuple[str, str]:
    """
    For kind=='title', treat first non-empty line as the "section_name".
    Remainder (if any) becomes paragraph text (e.g. rollcall lists under YEAS--61).
    """
    if not text:
        return ("", "")
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        return ("", "")
    heading = lines[0]
    body = "\n".join(lines[1:]).strip()
    return (heading, body)


# ----------------------------
# Event stream from one JSON file
# ----------------------------

def iter_daily_blocks(crdoc: dict) -> Iterator[dict]:
    """
    Yield normalized events in order:
      - {'kind':'title_section', 'text': '...'}
      - {'kind':'speaker', 'speaker_name': '...','speaker_bioguide': '...'}
      - {'kind':'paragraph', 'text': '...'}
    """
    content = crdoc.get("content") or []
    last_turn: Optional[int] = None

    for item in content:
        kind = (item.get("kind") or "").strip().lower()
        speaker = normalize_speaker(item.get("speaker"))
        bioguide = item.get("speaker_bioguide")
        turn = item.get("turn", None)
        text = (item.get("text") or "").strip()

        # Skip pure spacing artifacts
        if kind == "linebreak":
            continue

        # Titles: section boundaries (but may include body lists)
        if kind == "title":
            heading, body = split_title_heading_body(text)
            if heading:
                yield {"kind": "title_section", "text": heading}
            if body:
                yield {"kind": "paragraph", "text": body}
            last_turn = None
            continue

        # Speech-like kinds
        if kind in {"speech", "recorder"} and speaker:
            # Use 'turn' to avoid repeating speaker within the same turn
            if isinstance(turn, int) and turn >= 0:
                if turn != last_turn:
                    yield {"kind": "speaker", "speaker_name": speaker, "speaker_bioguide": bioguide}
                    last_turn = turn
            else:
                # turn==-1 or missing: treat each as standalone
                yield {"kind": "speaker", "speaker_name": speaker, "speaker_bioguide": bioguide}
                last_turn = None

            cleaned = strip_leading_speaker_label(text, speaker)
            for para in split_paragraphs(cleaned):
                yield {"kind": "paragraph", "text": para}
            continue

        # Metacharacters often contain [[Page ...]] and headers; usually skip.
        if kind == "metacharacters":
            continue

        # Any other kind: keep text only if it is meaningful.
        if text:
            for para in split_paragraphs(text):
                yield {"kind": "paragraph", "text": para}



@dataclass
class Counters:
    section_id: int = 0
    speech_id: int = 0
    paragraph_id: int = 0
    speaker_seq: int = 0
    speech_order: int = 1
    paragraph_order: int = 1


def day_to_year_part(day_id: str) -> Tuple[int, int, str]:
    """
    day_id like 'CREC-2022-04-11' -> (2022, 20220411, '2022-04-11')
    """
    m = re.match(r"CREC-(\d{4})-(\d{2})-(\d{2})$", day_id)
    if not m:
        raise ValueError(f"Unexpected day_id: {day_id}")
    y, mm, dd = m.group(1), m.group(2), m.group(3)
    return int(y), int(f"{y}{mm}{dd}"), f"{y}-{mm}-{dd}"


def harmonize_day(day_dir: Path, out_root: Path) -> None:
    """
    Input:
      day_dir/.../CREC-YYYY-MM-DD/json/*.json
    Output:
      out_root/<year>/{sections,speeches,paragraphs,speakers}_{year}_pt{YYYYMMDD}.csv
    """
    day_id = day_dir.name
    year, part, date_str = day_to_year_part(day_id)

    json_dir = day_dir / "json"
    paths = sorted(json_dir.glob("*.json"))

    # Choose a deterministic order across H/S/E pages.
    prefix_rank = {"H": 0, "S": 1, "E": 2, "D": 3}
    paths.sort(key=lambda p: sort_key_for_json_path(p, prefix_rank))

    counters = Counters()

    sections_rows: List[dict] = []
    speeches_rows: List[dict] = []
    paragraphs_rows: List[dict] = []
    speakers_unique: Dict[str, dict] = {}  # speaker_name -> {speaker_id, speaker_name, speaker_bioguide?}

    current_section_id: Optional[str] = None
    current_speech_id: Optional[str] = None
    last_doc_section_name: Optional[str] = None  # avoid repeating doc_title per page

    for part_page_idx, p in enumerate(paths, start=1):
        crdoc = json.loads(p.read_text())
        stem = p.stem  # includes PgH4437-2, etc.
        letter, page_num, seg = parse_pg_from_stem(stem)
        volume_page = f"{letter}{page_num}" if letter else (crdoc.get("header", {}).get("pages") or "")

        # File-level “section” from title/doc_title when it changes
        doc_section_name = (crdoc.get("title") or crdoc.get("doc_title") or "").strip()
        if doc_section_name and doc_section_name != last_doc_section_name:
            section_id_text = f"{year}_{part}_{counters.section_id}"
            sections_rows.append({
                "year": year,
                "part": part,
                "part_page": part_page_idx,
                "date": date_str,
                "volume_page": volume_page,
                "section_name": doc_section_name,
                "section_id": section_id_text,
            })
            current_section_id = section_id_text
            counters.section_id += 1
            counters.speech_order = 1
            counters.paragraph_order = 1
            current_speech_id = None
            last_doc_section_name = doc_section_name

        # Per-item events
        for ev in iter_daily_blocks(crdoc):
            if ev["kind"] == "title_section":
                section_name = ev["text"].strip()
                if not section_name:
                    continue

                section_id_text = f"{year}_{part}_{counters.section_id}"
                sections_rows.append({
                    "year": year,
                    "part": part,
                    "part_page": part_page_idx,
                    "date": date_str,
                    "volume_page": volume_page,
                    "section_name": section_name,
                    "section_id": section_id_text,
                })
                current_section_id = section_id_text
                counters.section_id += 1
                counters.speech_order = 1
                counters.paragraph_order = 1
                current_speech_id = None

            elif ev["kind"] == "speaker":
                speaker_name = (ev.get("speaker_name") or "").strip()
                speaker_bioguide = ev.get("speaker_bioguide", None)

                # Ensure some section exists
                if current_section_id is None:
                    section_id_text = f"{year}_{part}_{counters.section_id}"
                    sections_rows.append({
                        "year": year,
                        "part": part,
                        "part_page": part_page_idx,
                        "date": date_str,
                        "volume_page": volume_page,
                        "section_name": "(implicit)",
                        "section_id": section_id_text,
                    })
                    current_section_id = section_id_text
                    counters.section_id += 1
                    counters.speech_order = 1
                    counters.paragraph_order = 1

                # Unique speaker table (matches historical speakers_{year}_pt{part}.csv)
                if speaker_name and speaker_name not in speakers_unique:
                    counters.speaker_seq += 1
                    speakers_unique[speaker_name] = {
                        "speaker_id": f"{year}_{part}_{counters.speaker_seq}",
                        "speaker_name": speaker_name,
                        # not used for identifier script, but potentially useful for future xwalk
                        "speaker_bioguide": speaker_bioguide,
                    }

                speaker_id = speakers_unique.get(speaker_name, {}).get("speaker_id", "")

                speech_id_text = f"{year}_{part}_{counters.speech_id}"
                speeches_rows.append({
                    "section_id": current_section_id,
                    "speech_order": counters.speech_order,
                    "speaker_name": speaker_name,
                    "speaker_id": speaker_id,
                    "speech_id": speech_id_text,
                })
                current_speech_id = speech_id_text

                counters.speech_id += 1
                counters.speech_order += 1
                counters.paragraph_order = 1

            elif ev["kind"] == "paragraph":
                para = (ev.get("text") or "").strip()
                if not para:
                    continue

                # If a paragraph appears before any speaker, create a dummy speech row (schema preservation)
                if current_speech_id is None:
                    if current_section_id is None:
                        section_id_text = f"{year}_{part}_{counters.section_id}"
                        sections_rows.append({
                            "year": year,
                            "part": part,
                            "part_page": part_page_idx,
                            "date": date_str,
                            "volume_page": volume_page,
                            "section_name": "(implicit)",
                            "section_id": section_id_text,
                        })
                        current_section_id = section_id_text
                        counters.section_id += 1
                        counters.speech_order = 1
                        counters.paragraph_order = 1

                    speech_id_text = f"{year}_{part}_{counters.speech_id}"
                    speeches_rows.append({
                        "section_id": current_section_id,
                        "speech_order": counters.speech_order,
                        "speaker_name": "",
                        "speaker_id": "",
                        "speech_id": speech_id_text,
                    })
                    current_speech_id = speech_id_text
                    counters.speech_id += 1
                    counters.speech_order += 1
                    counters.paragraph_order = 1

                paragraph_id_text = f"{year}_{part}_{counters.paragraph_id}"
                paragraphs_rows.append({
                    "speech_id": current_speech_id,
                    "paragraph_text": para,
                    "paragraph_order": counters.paragraph_order,
                    "paragraph_id": paragraph_id_text,
                })
                counters.paragraph_id += 1
                counters.paragraph_order += 1

    # Write outputs
    out_year_dir = out_root / str(year)
    out_year_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(sections_rows).to_csv(out_year_dir / f"sections_{year}_pt{part}.csv", index=False)
    pd.DataFrame(speeches_rows).to_csv(out_year_dir / f"speeches_{year}_pt{part}.csv", index=False)
    pd.DataFrame(paragraphs_rows).to_csv(out_year_dir / f"paragraphs_{year}_pt{part}.csv", index=False)

    # identifier script expects a file with at least `speaker_name` column.
    speakers_df = pd.DataFrame(list(speakers_unique.values()))
    speakers_df.to_csv(out_year_dir / f"speakers_{year}_pt{part}.csv", index=False)


def harmonize_range(daily_root: Path, out_root: Path, start: str, end: str) -> None:
    """
    start/end: 'YYYY-MM-DD' inclusive
    """
    import datetime as dt

    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)

    # Directories look like: daily_root/2022/CREC-2022-04-11
    for year_dir in sorted(daily_root.iterdir()):
        if not year_dir.is_dir():
            continue
        for day_dir in sorted(year_dir.iterdir()):
            if not day_dir.is_dir() or not day_dir.name.startswith("CREC-"):
                continue
            try:
                day = dt.date.fromisoformat(day_dir.name.replace("CREC-", ""))
            except ValueError:
                continue
            if s <= day <= e:
                harmonize_day(day_dir, out_root)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-root", required=True)   # e.g. datastore/inference/daily
    ap.add_argument("--out-root", required=True)     # e.g. datastore/inference/daily_harmonized
    ap.add_argument("--start", required=True)        # YYYY-MM-DD
    ap.add_argument("--end", required=True)          # YYYY-MM-DD
    args = ap.parse_args()

    harmonize_range(Path(args.daily_root), Path(args.out_root), args.start, args.end)
