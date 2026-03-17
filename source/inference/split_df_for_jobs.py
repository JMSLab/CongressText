import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class PathsConfig:
    inference_dir: str = "datastore/inference"
    image_dir: str = "datastore/scrape/cr-bound"

    @property
    def docs_path(self) -> str:
        return os.path.join(self.inference_dir, "docs.csv")

    @property
    def sections_path(self) -> str:
        return os.path.join(self.inference_dir, "sections.csv")

    @property
    def speeches_path(self) -> str:
        return os.path.join(self.inference_dir, "speeches.csv")

    @property
    def speakers_path(self) -> str:
        return os.path.join(self.inference_dir, "speakers.csv")




PDF_PATTERN = re.compile(r"GPO-CRECB-(\d{4})-pt(\d{1,2})(-v\d+)?\.pdf")


def list_pdf_filenames(image_dir: str) -> List[str]:
    """List files in image_dir."""
    return os.listdir(image_dir)


def parse_doc_filename(filename: str) -> Tuple[str, int, int, int]:
    """
    Parse a filename into (filename, year, part, version_number).

    version_number is:
      - parsed from '-vN' if present
      - else 100 (matches original script's default)
    Raises ValueError if filename doesn't match the pattern.
    """
    match = PDF_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    year_str, part_str, version = match.groups()
    version_number = int(version.replace("-v", "")) if version else 100
    return filename, int(year_str), int(part_str), version_number


def discover_documents(image_dir: str) -> List[Tuple[str, int, int, int]]:
    """Return parsed doc tuples for files in directory that match the expected pattern."""
    parsed: List[Tuple[str, int, int, int]] = []
    for fn in list_pdf_filenames(image_dir):
        try:
            parsed.append(parse_doc_filename(fn))
        except ValueError:
            continue
    return parsed


def select_latest_version_per_year_part(
    parsed_docs: List[Tuple[str, int, int, int]]
) -> List[str]:
    """
    Select one doc per (year, part), keeping the latest version.
    """
    parsed_docs.sort(key=lambda x: (x[1], x[2], -x[3]))

    final_docs: Dict[Tuple[int, int], Tuple[str, int, int, int]] = {}
    for doc, year, part, version_number in parsed_docs:
        key = (year, part)
        if key not in final_docs or final_docs[key][3] > version_number:
            final_docs[key] = (doc, year, part, version_number)

    return [info[0] for info in final_docs.values()]



def build_docs_df(filenames: List[str]) -> pd.DataFrame:
    """Initialize docs progress dataframe."""
    return pd.DataFrame(
        {
            "title": filenames,
            "complete": [0] * len(filenames),
            "section_id": [0] * len(filenames),
            "speech_id": [0] * len(filenames),
            "paragraph_id": [0] * len(filenames),
        }
    )


def build_sections_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["year", "part", "part_page", "date", "volume_page", "section_name", "section_id"]
    )


def build_speeches_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["section_id", "speech_order", "speaker_name", "speaker_id", "speech_id"]
    )


def build_speakers_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["speaker_id", "speaker_name"])


def initialize_progress_dataframes(image_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construct docs_df from discovered files and initialize the other empty dfs.
    """
    parsed = discover_documents(image_dir)
    selected = select_latest_version_per_year_part(parsed)

    docs_df = build_docs_df(selected)
    sections_df = build_sections_df()
    speeches_df = build_speeches_df()
    speakers_df = build_speakers_df()
    return docs_df, sections_df, speeches_df, speakers_df




def ensure_inference_dir_exists(inference_dir: str) -> None:
    os.makedirs(inference_dir, exist_ok=True)


def write_initial_csvs_if_missing(cfg: PathsConfig) -> None:
    """
    Create docs/sections/speeches/speakers CSVs if docs.csv doesn't exist.
    """
    ensure_inference_dir_exists(cfg.inference_dir)

    if not os.path.isfile(cfg.docs_path):
        docs_df, sections_df, speeches_df, speakers_df = initialize_progress_dataframes(cfg.image_dir)
        docs_df.to_csv(cfg.docs_path, index=False)
        sections_df.to_csv(cfg.sections_path, index=False)
        speeches_df.to_csv(cfg.speeches_path, index=False)
        speakers_df.to_csv(cfg.speakers_path, index=False)




def load_docs_df(cfg: PathsConfig) -> pd.DataFrame:
    return pd.read_csv(cfg.docs_path)


def filter_incomplete_docs(docs_df: pd.DataFrame) -> pd.DataFrame:
    return docs_df[docs_df["complete"] == 0]


def split_into_chunks(df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
    """Split a DataFrame into N chunks (same as np.array_split)."""
    return list(np.array_split(df, num_chunks))


def write_chunks(chunks: List[pd.DataFrame], inference_dir: str, prefix: str = "chunk_") -> None:
    for i, chunk in enumerate(chunks):
        chunk.to_csv(os.path.join(inference_dir, f"{prefix}{i}.csv"), index=False)




def main(num_chunks: int = 10) -> None:
    cfg = PathsConfig()

    # 1) Ensure progress CSVs exist
    write_initial_csvs_if_missing(cfg)

    # 2) Load docs and chunk incomplete
    docs_df = load_docs_df(cfg)
    files_to_process = filter_incomplete_docs(docs_df)

    chunks = split_into_chunks(files_to_process, num_chunks=num_chunks)
    write_chunks(chunks, inference_dir=cfg.inference_dir, prefix="chunk_")  


if __name__ == "__main__":
    main(num_chunks=10)
    
