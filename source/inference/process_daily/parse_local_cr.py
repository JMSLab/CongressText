#!/usr/bin/env python
import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from zipfile import ZipFile

from congressionalrecord.govinfo.cr_parser import ParseCRDir, ParseCRFile


def parse_zip(zip_path: Path, out_root: Path) -> None:
    """
    Given one CREC-YYYY-MM-DD.zip, extract it under out_root/<year>/CREC-YYYY-MM-DD
    and write one JSON file per .htm into a json/ subdir.
    """
    logging.info("Processing %s", zip_path)

    # Example filename: CREC-2007-12-10.zip
    day_id = zip_path.stem           # "CREC-2007-12-10"
    parts = day_id.split("-")        # ["CREC", "2007", "12", "10"]
    year = parts[1]

    # Where this day's stuff will live
    day_dir = out_root / year / day_id
    html_dir = day_dir / "html"
    json_dir = day_dir / "json"

    # 1) Extract if not already extracted
    if not day_dir.exists():
        logging.info("Extracting %s to %s", zip_path, day_dir.parent)
        day_dir.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(day_dir.parent)
    else:
        logging.info("Directory %s already exists, skipping extract", day_dir)

    # 2) Make ParseCRDir object for this day (needs mods.xml + html/)
    crdir = ParseCRDir(str(day_dir))

    # 3) Iterate HTML files and parse each
    json_dir.mkdir(exist_ok=True)
    for fname in sorted(os.listdir(html_dir)):
        if not fname.lower().endswith(".htm"):
            continue

        parse_path = html_dir / fname

        # Match the repo's own skip logic: skip page D, front matter, and Pgnull files :contentReference[oaicite:2]{index=2}
        pstr = str(parse_path)
        if any(s in pstr for s in ("-PgD", "FrontMatter", "-Pgnull")):
            logging.info("Skipping %s", parse_path)
            continue

        logging.info("Parsing %s", parse_path)
        try:
            crfile = ParseCRFile(str(parse_path), crdir)
        except Exception as e:
            logging.exception("Failed to parse %s: %s", parse_path, e)
            # Optionally keep a log of failed files per day
            continue

        out_name = Path(fname).stem + ".json"
        out_path = json_dir / out_name
        with out_path.open("w") as f:
            json.dump(crfile.crdoc, f)

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

    # Loop over all CREC-*.zip in the ZIP root and filter by date
    for entry in sorted(zip_root.iterdir()):
        print(entry.name)
        if not (entry.is_file() and entry.name.startswith("CREC-") and entry.suffix == ".zip"):
            continue

        # Extract YYYY-MM-DD from CREC-YYYY-MM-DD.zip
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
