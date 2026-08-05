#!/usr/bin/env python3
"""
Plot the legislators who gave the most speeches in a year, by chamber.

Inference part files are cumulative worker snapshots: a file named for one
YEAR/PART can contain rows carried forward from earlier documents.  The loader
therefore reads parts in numeric order, deduplicates on each family's stable
key while keeping the latest snapshot copy, and selects rows using row-level
IDs/metadata rather than trusting the filename alone.

For speeches, the year is selected from speech_id after validating that
section_id has the same year and part.  Identified speakers are *not* filtered
by the year prefix of speaker_id, because a current-year speech may legitimately
reuse an ID first assigned in an earlier year.  Instead, identified rows are
selected relationally from the speaker IDs referenced by the target speeches.

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CHAMBERS = ("senate", "house")
PROCEDURAL_POSITIONS = frozenset({"2", "3", "4", "5"})
DEFAULT_MIN_KEY_COVERAGE = 0.05

KEY_COLUMNS = {
    "sections": "section_id",
    "speeches": "speech_id",
    "speakers": "speaker_id",
    "paragraphs": "paragraph_id",
    "identified_speakers": "speaker_id",
}

REQUIRED_COLUMNS = {
    "sections": ("section_id", "year"),
    "speeches": ("speech_id", "section_id", "speaker_id"),
    "speakers": ("speaker_id",),
    "paragraphs": ("paragraph_id", "speech_id"),
    "identified_speakers": (
        "speaker_id",
        "position",
        "icpsr",
        "chamber",
        "lastname",
        "firstname",
        "state_abbrev",
    ),
}

IDENTIFIED_JOIN_COLUMNS = (
    "speaker_id",
    "position",
    "icpsr",
    "chamber",
    "lastname",
    "firstname",
    "state_abbrev",
)

ID_PATTERN = r"^(?P<year>\d{4})_(?P<part>\d+)_(?P<counter>\d+)$"


class DataValidationError(RuntimeError):
    """Raised when an inference snapshot violates its expected schema."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=Path("datastore") / "inference",
    )
    parser.add_argument("--year", type=int, default=1895)
    parser.add_argument("--top-n", type=positive_int, default=15)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output") / "analysis",
    )
    parser.add_argument(
        "--min-match-rate",
        "--min-key-coverage",
        dest="min_key_coverage",
        type=probability,
        default=DEFAULT_MIN_KEY_COVERAGE,
        help=(
            "Minimum share of target-year speeches whose speaker_id must occur "
            "in current-year identified_speakers snapshots before plotting "
            f"(default: {DEFAULT_MIN_KEY_COVERAGE})."
        ),
    )
    return parser.parse_args(argv)


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def extract_id_components(series: pd.Series) -> pd.DataFrame:
    return normalize_string(series).str.extract(ID_PATTERN)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )


def part_files(inference_dir: Path | str, family: str, year: int) -> list[Path]:
    """Return exact top-level FAMILY_YEAR_ptPART.csv files in numeric part order."""
    if family not in KEY_COLUMNS:
        raise ValueError(f"Unknown inference family: {family}")

    inference_dir = Path(inference_dir)
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference directory does not exist: {inference_dir}")

    filename_re = re.compile(
        rf"^{re.escape(family)}_{year}_pt(?P<part>\d+)\.csv$"
    )
    matches: list[tuple[int, Path]] = []

    for path in inference_dir.iterdir():
        if not path.is_file():
            continue
        match = filename_re.fullmatch(path.name)
        if match:
            matches.append((int(match.group("part")), path))

    if not matches:
        raise FileNotFoundError(
            f"No files matching {inference_dir / f'{family}_{year}_pt*.csv'}"
        )

    matches.sort(key=lambda item: item[0])
    return [path for _, path in matches]


FrameSelector = Callable[[pd.DataFrame, Path], pd.DataFrame]


def accumulate_snapshots(
    inference_dir: Path | str,
    family: str,
    year: int,
    selector: FrameSelector | None = None,
) -> pd.DataFrame:
    """
    Read cumulative snapshots incrementally and keep the latest row per key.

    Incremental accumulation is equivalent to concatenating all files in numeric
    part order and calling drop_duplicates(key, keep="last"), but it avoids
    holding the full repeated snapshot stack in memory at once.

    A selector may reduce each frame before accumulation when selection depends
    only on stable row information, such as speech_id or membership in a fixed
    set of speaker IDs.
    """
    files = part_files(inference_dir, family, year)
    key = KEY_COLUMNS[family]
    required = REQUIRED_COLUMNS[family]

    accumulated: pd.DataFrame | None = None
    fallback_columns: list[str] = []
    rows_read = 0
    rows_selected = 0

    for path in files:
        frame = pd.read_csv(path, dtype="string", low_memory=False)
        require_columns(frame, required, path.name)
        fallback_columns = list(frame.columns)
        rows_read += len(frame)

        frame[key] = normalize_string(frame[key])
        invalid_key = frame[key].isna() | frame[key].eq("")
        if invalid_key.any():
            raise DataValidationError(
                f"{path.name}: {int(invalid_key.sum()):,} row(s) have missing {key}"
            )

        if selector is not None:
            frame = selector(frame, path).copy()
        else:
            frame = frame.copy()

        rows_selected += len(frame)
        if frame.empty:
            continue

        frame = frame.drop_duplicates(subset=key, keep="last")
        if accumulated is None:
            accumulated = frame
        else:
            accumulated = pd.concat(
                [accumulated, frame],
                ignore_index=True,
                sort=False,
            ).drop_duplicates(subset=key, keep="last")

    if accumulated is None:
        accumulated = pd.DataFrame(columns=fallback_columns)

    accumulated = accumulated.reset_index(drop=True)
    print(
        f"{family}: {len(files)} part file(s), {rows_read:,} rows read, "
        f"{rows_selected:,} relevant snapshot rows, "
        f"{len(accumulated):,} after dropping duplicate {key}"
    )
    return accumulated


def select_speech_rows(frame: pd.DataFrame, _path: Path, year: int) -> pd.DataFrame:
    """Select target-like speech IDs before cumulative deduplication."""
    speech_id = normalize_string(frame["speech_id"])
    target_like = speech_id.str.startswith(f"{year}_", na=False)
    selected = frame.loc[target_like].copy()
    if not selected.empty:
        selected["speech_id"] = normalize_string(selected["speech_id"])
        selected["section_id"] = normalize_string(selected["section_id"])
        selected["speaker_id"] = normalize_string(selected["speaker_id"])
    return selected


def validate_speeches_for_year(speeches: pd.DataFrame, year: int) -> pd.DataFrame:
    """Validate canonical, deduplicated speech/section ID consistency."""
    if speeches.empty:
        return speeches

    speech_parts = extract_id_components(speeches["speech_id"])
    section_parts = extract_id_components(speeches["section_id"])

    missing_section = speeches["section_id"].isna() | speeches["section_id"].eq("")
    malformed_speech = speech_parts["year"].isna()
    malformed_section = ~missing_section & section_parts["year"].isna()
    wrong_speech_year = (
        speech_parts["year"].notna()
        & ~speech_parts["year"].eq(str(year))
    )
    wrong_section_year = (
        section_parts["year"].notna()
        & ~section_parts["year"].eq(speech_parts["year"])
    )
    wrong_section_part = (
        section_parts["part"].notna()
        & speech_parts["part"].notna()
        & ~section_parts["part"].eq(speech_parts["part"])
    )

    failures = {
        "missing_section": int(missing_section.sum()),
        "malformed_speech": int(malformed_speech.sum()),
        "malformed_section": int(malformed_section.sum()),
        "wrong_speech_year": int(wrong_speech_year.sum()),
        "section_year_mismatch": int(wrong_section_year.sum()),
        "section_part_mismatch": int(wrong_section_part.sum()),
    }
    active = [f"{name}={count:,}" for name, count in failures.items() if count]
    if active:
        raise DataValidationError(
            "Canonical own-year speech ID validation failed "
            f"({'; '.join(active)})"
        )

    return speeches.reset_index(drop=True)

def select_section_rows(frame: pd.DataFrame, _path: Path, year: int) -> pd.DataFrame:
    """Select sections using sections.year and cross-check section_id year."""
    explicit_year = pd.to_numeric(frame["year"], errors="coerce")
    section_id = normalize_string(frame["section_id"])
    id_parts = extract_id_components(section_id)

    metadata_target = explicit_year.eq(year)
    id_target = id_parts["year"].eq(str(year))
    candidate = metadata_target | id_target

    malformed = candidate & id_parts["year"].isna()
    disagreement = candidate & ~metadata_target.eq(id_target)
    if malformed.any() or disagreement.any():
        raise DataValidationError(
            "sections: row-level year and section_id disagree for the requested "
            f"year (malformed={int(malformed.sum()):,}, "
            f"disagreement={int(disagreement.sum()):,})"
        )

    return frame.loc[metadata_target].copy()


def select_paragraph_rows(frame: pd.DataFrame, _path: Path, year: int) -> pd.DataFrame:
    """Select target-like paragraph speech IDs before deduplication."""
    speech_id = normalize_string(frame["speech_id"])
    target_like = speech_id.str.startswith(f"{year}_", na=False)
    selected = frame.loc[target_like].copy()
    if not selected.empty:
        selected["speech_id"] = normalize_string(selected["speech_id"])
    return selected


def validate_paragraphs_for_year(paragraphs: pd.DataFrame, year: int) -> pd.DataFrame:
    if paragraphs.empty:
        return paragraphs
    parsed = extract_id_components(paragraphs["speech_id"])
    malformed = parsed["year"].isna()
    wrong_year = parsed["year"].notna() & ~parsed["year"].eq(str(year))
    if malformed.any() or wrong_year.any():
        raise DataValidationError(
            "Canonical paragraph rows contain malformed or wrong-year speech_id "
            f"values (malformed={int(malformed.sum()):,}, "
            f"wrong_year={int(wrong_year.sum()):,})"
        )
    return paragraphs.reset_index(drop=True)

def load_year_family(
    inference_dir: Path | str,
    family: str,
    year: int,
) -> pd.DataFrame:
    """
    Return canonical target-year rows for a cumulative filename family.

    Speaker families are deliberately returned without speaker_id year-prefix
    filtering.  Select those rows relationally from target speeches.
    """
    if family == "speeches":
        deduplicated = accumulate_snapshots(
            inference_dir,
            family,
            year,
            selector=lambda frame, path: select_speech_rows(frame, path, year),
        )
        return validate_speeches_for_year(deduplicated, year)

    if family == "paragraphs":
        deduplicated = accumulate_snapshots(
            inference_dir,
            family,
            year,
            selector=lambda frame, path: select_paragraph_rows(frame, path, year),
        )
        return validate_paragraphs_for_year(deduplicated, year)

    if family == "sections":
        # Selection uses a mutable metadata column, so first deduplicate the full
        # family and only then select/cross-check the requested year.
        deduplicated = accumulate_snapshots(inference_dir, family, year)
        return select_section_rows(
            deduplicated,
            Path(f"deduplicated {family} family"),
            year,
        ).reset_index(drop=True)

    if family in {"speakers", "identified_speakers"}:
        return accumulate_snapshots(inference_dir, family, year)

    raise ValueError(f"Unknown inference family: {family}")


def load_identified_for_speeches(
    inference_dir: Path | str,
    year: int,
    speeches: pd.DataFrame,
) -> pd.DataFrame:
    """Load only identified-speaker rows referenced by target-year speeches."""
    needed_ids = frozenset(
        normalize_string(speeches["speaker_id"])
        .dropna()
        .loc[lambda values: values.ne("")]
        .tolist()
    )

    def select_needed(frame: pd.DataFrame, _path: Path) -> pd.DataFrame:
        speaker_id = normalize_string(frame["speaker_id"])
        return frame.loc[speaker_id.isin(needed_ids)].copy()

    identified = accumulate_snapshots(
        inference_dir,
        "identified_speakers",
        year,
        selector=select_needed,
    )
    print(
        f"identified_speakers: {len(needed_ids):,} unique speaker ID(s) "
        f"referenced; {len(identified):,} identified row(s) retained"
    )
    return identified


def normalize_position(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def normalize_chamber(series: pd.Series) -> pd.Series:
    chamber = normalize_string(series).str.lower()
    return chamber.replace(
        {
            "h": "house",
            "house of representatives": "house",
            "representative": "house",
            "s": "senate",
            "senator": "senate",
        }
    )


def build_speech_person_table(
    speeches: pd.DataFrame,
    identified: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Join speeches to identifications and count unique speeches per member."""
    require_columns(
        identified,
        REQUIRED_COLUMNS["identified_speakers"],
        "identified_speakers",
    )

    speech_ids = normalize_string(speeches["speaker_id"])
    identified_ids = frozenset(normalize_string(identified["speaker_id"]).dropna())
    key_matched = speech_ids.isin(identified_ids)
    key_coverage = float(key_matched.mean()) if len(speeches) else 0.0

    merged = speeches.merge(
        identified[list(IDENTIFIED_JOIN_COLUMNS)],
        on="speaker_id",
        how="left",
        validate="many_to_one",
    )
    merged["position"] = normalize_position(merged["position"])
    merged["chamber"] = normalize_chamber(merged["chamber"])

    procedural = merged["position"].isin(PROCEDURAL_POSITIONS)
    substantive = merged.loc[~procedural].copy()
    recognized = substantive["chamber"].isin(CHAMBERS)
    matched_member = substantive["icpsr"].notna() & recognized
    member_match_rate = (
        float(matched_member.mean()) if len(substantive) else 0.0
    )

    unknown_chambers = sorted(
        substantive.loc[
            substantive["chamber"].notna() & ~recognized,
            "chamber",
        ].unique().tolist()
    )
    if unknown_chambers:
        print(
            "WARNING: excluding unrecognized chamber value(s): "
            + ", ".join(map(str, unknown_chambers))
        )

    matched = substantive.loc[matched_member].copy()
    counts = (
        matched.groupby(["chamber", "icpsr"], dropna=False)
        .agg(
            n_speeches=("speech_id", "nunique"),
            lastname=("lastname", "first"),
            firstname=("firstname", "first"),
            state_abbrev=("state_abbrev", "first"),
        )
        .reset_index()
    )

    stats: dict[str, float | int] = {
        "speaker_key_coverage": key_coverage,
        "member_match_rate": member_match_rate,
        "speech_rows": len(speeches),
        "procedural_rows": int(procedural.sum()),
        "substantive_rows": len(substantive),
        "matched_substantive_rows": int(matched_member.sum()),
        "identified_rows": len(identified),
        "missing_speaker_ids": int((~key_matched).sum()),
    }

    print(f"speaker_id key coverage: {key_coverage:.4f}")
    print(
        "identified-legislator rate among non-procedural/unknown speeches: "
        f"{member_match_rate:.4f}"
    )
    print(f"{len(counts):,} chamber-legislator row(s) available for ranking")
    return counts, stats


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def format_label(row: pd.Series) -> str:
    lastname = clean_text(row.get("lastname"))
    firstname = clean_text(row.get("firstname"))
    state = clean_text(row.get("state_abbrev"))

    if lastname or firstname:
        name = ", ".join(
            part for part in (lastname.title(), firstname.title()) if part
        )
    else:
        name = f"ICPSR {row['icpsr']}"

    return f"{name} ({state})" if state else name


def top_speakers_by_chamber(
    counts: pd.DataFrame,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}

    for chamber in CHAMBERS:
        panel = counts.loc[counts["chamber"].eq(chamber)].copy()
        panel.sort_values(
            ["n_speeches", "lastname", "firstname", "icpsr"],
            ascending=[False, True, True, True],
            na_position="last",
            inplace=True,
        )
        panel = panel.head(top_n).copy()
        panel["label"] = panel.apply(format_label, axis=1)
        panels[chamber] = panel
        print(f"{chamber}: {len(panel)} legislator(s) selected")

    return panels


def make_dotplot(
    panels: Mapping[str, pd.DataFrame],
    year: int,
    out_path: Path | str,
) -> None:
    figure, axes = plt.subplots(1, len(CHAMBERS), figsize=(13, 7), squeeze=False)

    for axis, chamber in zip(axes.ravel(), CHAMBERS):
        panel = panels[chamber].iloc[::-1]
        axis.set_title(f"{chamber.capitalize()} ({len(panel)} shown)")

        if panel.empty:
            axis.text(
                0.5,
                0.5,
                "No identified legislators",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            continue

        positions = list(range(len(panel)))
        axis.hlines(
            y=positions,
            xmin=0,
            xmax=panel["n_speeches"],
            color="0.8",
            linewidth=1,
            zorder=1,
        )
        axis.scatter(
            panel["n_speeches"],
            positions,
            s=45,
            color="#1f4e79",
            zorder=2,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(panel["label"], fontsize=9)
        axis.set_xlabel("Speeches")
        axis.set_xlim(left=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="0.9", linewidth=0.8)
        axis.set_axisbelow(True)

    figure.suptitle(f"Legislators giving the most speeches, {year}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"Wrote {out_path}")


def make_placeholder(diagnostics: Sequence[str], out_path: Path | str) -> None:
    figure, axis = plt.subplots(figsize=(13, 7))
    axis.axis("off")
    axis.text(
        0.5,
        0.95,
        "PLACEHOLDER - FIGURE NOT PRODUCED",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#7f1d1d",
    )
    axis.text(
        0.02,
        0.82,
        "\n".join(diagnostics),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"Wrote placeholder {out_path}")


def write_counts(
    panels: Mapping[str, pd.DataFrame],
    out_path: Path | str,
) -> None:
    columns = [
        "chamber",
        "icpsr",
        "lastname",
        "firstname",
        "state_abbrev",
        "n_speeches",
    ]
    nonempty = [
        panels[chamber]
        for chamber in CHAMBERS
        if chamber in panels and not panels[chamber].empty
    ]
    if nonempty:
        table = pd.concat(nonempty, ignore_index=True, sort=False)[columns]
    else:
        table = pd.DataFrame(columns=columns)

    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(table)} rows)")


def coverage_diagnostics(
    year: int,
    stats: Mapping[str, float | int],
    min_key_coverage: float,
) -> list[str]:
    return [
        f"Year requested: {year}",
        f"Validated own-year speech rows: {stats['speech_rows']:,}",
        f"Relevant identified-speaker rows: {stats['identified_rows']:,}",
        f"speaker_id key coverage: {stats['speaker_key_coverage']:.4f} "
        f"(need >= {min_key_coverage:.4f})",
        f"identified legislator rate: {stats['member_match_rate']:.4f}",
        f"Speech rows lacking an identified speaker key: "
        f"{stats['missing_speaker_ids']:,}",
        "",
        "Cumulative copies were deduplicated and wrong-year speech rows were",
        "excluded using speech_id. Earlier-year prefixes in speaker_id are valid",
        "and were retained relationally.",
        "",
        "Low speaker-key coverage indicates missing, stale, or incomplete",
        "identified_speakers_YEAR_ptPART.csv outputs. Re-run speaker",
        "disambiguation for this year if necessary; this diagnostic alone does",
        "not call for re-running layout detection or OCR.",
    ]


def failure_diagnostics(year: int, error: Exception) -> list[str]:
    return [
        f"Year requested: {year}",
        "",
        "Input validation failed:",
        str(error),
        "",
        "No ranking was produced; the counts CSV is intentionally empty.",
    ]


def main(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "top_speakers.png"
    counts_path = args.out_dir / "top_speakers.csv"

    try:
        speeches = load_year_family(args.inference_dir, "speeches", args.year)

        if speeches.empty:
            stats: dict[str, float | int] = {
                "speaker_key_coverage": 0.0,
                "member_match_rate": 0.0,
                "speech_rows": 0,
                "identified_rows": 0,
                "missing_speaker_ids": 0,
            }
            diagnostics = coverage_diagnostics(
                args.year,
                stats,
                args.min_key_coverage,
            )
            for line in diagnostics:
                print(line)
            make_placeholder(diagnostics, figure_path)
            write_counts({}, counts_path)
            return 0

        identified = load_identified_for_speeches(
            args.inference_dir,
            args.year,
            speeches,
        )
        counts, stats = build_speech_person_table(speeches, identified)

        if (
            float(stats["speaker_key_coverage"]) < args.min_key_coverage
            or counts.empty
        ):
            diagnostics = coverage_diagnostics(
                args.year,
                stats,
                args.min_key_coverage,
            )
            for line in diagnostics:
                print(line)
            make_placeholder(diagnostics, figure_path)
            write_counts({}, counts_path)
            return 0

        panels = top_speakers_by_chamber(counts, args.top_n)
        make_dotplot(panels, args.year, figure_path)
        write_counts(panels, counts_path)
        return 0

    except (FileNotFoundError, DataValidationError, pd.errors.ParserError) as error:
        diagnostics = failure_diagnostics(args.year, error)
        for line in diagnostics:
            print(line, file=sys.stderr)
        make_placeholder(diagnostics, figure_path)
        write_counts({}, counts_path)
        return 1


if __name__ == "__main__":
    sys.exit(main(parse_args()))
