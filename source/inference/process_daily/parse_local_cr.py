#!/usr/bin/env python
import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from zipfile import ZipFile

from congressionalrecord.govinfo.cr_parser import ParseCRDir, ParseCRFile


def get_day_metadata(zip_path: Path, out_root: Path) -> tuple[str, Path, Path, Path]:
    day_id = zip_path.stem
    parts = day_id.split("-")
    year = parts[1]

    day_dir = out_root / year / day_id
    html_dir = day_dir / "html"
    json_dir = day_dir / "json"

    return day_id, day_dir, html_dir, json_dir


def extract_zip_if_needed(zip_path: Path, day_dir: Path) -> None:
    if not day_dir.exists():
        logging.info("Extracting %s to %s", zip_path, day_dir.parent)
        day_dir.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(day_dir.parent)
    else:
        logging.info("Directory %s already exists, skipping extract", day_dir)


def build_crdir(day_dir: Path) -> ParseCRDir:
    return ParseCRDir(str(day_dir))


def should_skip_html(parse_path: Path) -> bool:
    pstr = str(parse_path)
    return any(s in pstr for s in ("-PgD", "FrontMatter", "-Pgnull"))


def iter_html_files(html_dir: Path):
    for fname in sorted(os.listdir(html_dir)):
        if fname.lower().endswith(".htm"):
            yield fname, html_dir / fname


def parse_html_file(parse_path: Path, crdir: ParseCRDir):
    try:
        return ParseCRFile(str(parse_path), crdir)
    except Exception as e:
        logging.exception("Failed to parse %s: %s", parse_path, e)
        return None


def write_json_output(json_dir: Path, fname: str, crfile: ParseCRFile) -> None:
    out_name = Path(fname).stem + ".json"
    out_path = json_dir / out_name
    with out_path.open("w") as f:
        json.dump(crfile.crdoc, f)


def parse_zip(zip_path: Path, out_root: Path) -> None:
    """
    Given one CREC-YYYY-MM-DD.zip, extract it under out_root/<year>/CREC-YYYY-MM-DD
    and write one JSON file per .htm into a json/ subdir.
    """
    logging.info("Processing %s", zip_path)

    _, day_dir, html_dir, json_dir = get_day_metadata(zip_path, out_root)

    extract_zip_if_needed(zip_path, day_dir)
    crdir = build_crdir(day_dir)

    json_dir.mkdir(exist_ok=True)
    for fname, parse_path in iter_html_files(html_dir):
        if should_skip_html(parse_path):
            logging.info("Skipping %s", parse_path)
            continue

        logging.info("Parsing %s", parse_path)
        crfile = parse_html_file(parse_path, crdir)
        if crfile is None:
            continue

        write_json_output(json_dir, fname, crfile)

    logging.info("Finished %s", zip_path)


def main():
    parser = argparse.ArgumentParser(
        description="Parse locally stored CREC-YYYY-MM-DD.zip files into JSON."
    )
    parser.add_argument(
        "start",
        help="First date to parse (YYYY-MM-DD)",
    )
    parser.add_argument(
        "end",
        help="Last date to parse (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--zip-root",
        default="../../../datastore/scrape/cr-daily",
        help="Directory containing CREC-YYYY-MM-DD.zip files",
    )
    parser.add_argument(
        "--out-root",
        default="../../../datastore/inference/daily",
        help="Directory where parsed days (and JSON) will be written",
    )
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    zip_root = (Path(__file__).resolve().parent / args.zip_root).resolve()
    out_root = (Path(__file__).resolve().parent / args.out_root).resolve()

    logging.info("ZIP root: %s", zip_root)
    logging.info("Output root: %s", out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for entry in sorted(zip_root.iterdir()):
        print(entry.name)
        if not (entry.is_file() and entry.name.startswith("CREC-") and entry.suffix == ".zip"):
            continue

        date_str = entry.stem.replace("CREC-", "")
        try:
            day = dt.date.fromisoformat(date_str)
        except ValueError:
            logging.warning("Skipping unexpected file name %s", entry.name)
            continue

        if start <= day <= end:
            parse_zip(entry, out_root)
        else:
            logging.debug("Skipping %s (outside date range)", entry.name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    main()