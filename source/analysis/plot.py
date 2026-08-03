"""Plot the legislators who gave the most speeches in a year, by chamber."""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


CHAMBERS = ['senate', 'house']

# Speaker, Vice-President, President and Presiding Officer
PROCEDURAL_POSITIONS = ['2', '3', '4', '5']

# Below this share of speeches joining to an identified speaker, treat the speaker
# keys as broken rather than merely incomplete.
MIN_MATCH_RATE = 0.05

KEY_COLUMNS = {
    'sections': 'section_id',
    'speeches': 'speech_id',
    'speakers': 'speaker_id',
    'paragraphs': 'paragraph_id',
    'identified_speakers': 'speaker_id',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-dir', default=os.path.join('datastore', 'inference'))
    parser.add_argument('--year', type=int, default=1895)
    parser.add_argument('--top-n', type=int, default=15)
    parser.add_argument('--out-dir', default=os.path.join('output', 'analysis'))
    return parser.parse_args(argv)


def part_files(inference_dir, family, year):
    pattern = os.path.join(inference_dir, f"{family}_{year}_pt*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    def part_number(path):
        return int(re.search(r"_pt(\d+)\.csv$", os.path.basename(path)).group(1))

    return sorted(files, key=part_number)


def load_year_family(inference_dir, family, year):
    key = KEY_COLUMNS[family]
    files = part_files(inference_dir, family, year)

    frames = [pd.read_csv(path, dtype=str, low_memory=False) for path in files]
    stacked = pd.concat(frames, ignore_index=True, sort=False)
    deduplicated = stacked.drop_duplicates(subset=key, keep='last')

    print(f"{family}: {len(files)} part files, "
          f"{len(stacked)} rows read, {len(deduplicated)} after dropping duplicate {key}")

    return deduplicated


def select_year(speeches, sections, year):
    dated = speeches.merge(sections[['section_id', 'year']], on='section_id', how='left')
    selected = dated[dated['year'] == str(year)]

    print(f"speeches: {len(selected)} of {len(dated)} rows are from {year}")

    return selected


def compute_match_rate(speeches, identified):
    if len(speeches) == 0:
        return 0.0

    return speeches['speaker_id'].isin(set(identified['speaker_id'])).mean()


def build_speech_person_table(speeches, identified):
    columns = ['speaker_id', 'position', 'icpsr', 'chamber',
               'lastname', 'firstname', 'state_abbrev']
    merged = speeches.merge(identified[columns], on='speaker_id', how='left')

    merged = merged[~merged['position'].isin(PROCEDURAL_POSITIONS)]
    merged = merged[merged['icpsr'].notna() & merged['chamber'].notna()]
    merged['chamber'] = merged['chamber'].str.lower()

    counts = (merged.groupby('icpsr')
                    .agg(n_speeches=('speech_id', 'nunique'),
                         lastname=('lastname', 'first'),
                         firstname=('firstname', 'first'),
                         state_abbrev=('state_abbrev', 'first'),
                         chamber=('chamber', 'first'))
                    .reset_index())

    print(f"{len(counts)} legislators after excluding procedural speakers")

    return counts


def format_label(row):
    lastname = str(row['lastname']).title()
    firstname = str(row['firstname']).title()
    state = row['state_abbrev']

    if pd.isna(state) or not str(state).strip():
        return f"{lastname}, {firstname}"

    return f"{lastname}, {firstname} ({state})"


def top_speakers_by_chamber(counts, top_n):
    panels = {}

    for chamber in CHAMBERS:
        panel = counts[counts['chamber'] == chamber]
        panel = panel.sort_values(['n_speeches', 'lastname', 'firstname'],
                                 ascending=[False, True, True])
        panel = panel.head(top_n).copy()
        panel['label'] = panel.apply(format_label, axis=1)
        panels[chamber] = panel

        print(f"{chamber}: {len(panel)} legislators plotted")

    return panels


def make_dotplot(panels, year, out_path):
    figure, axes = plt.subplots(1, len(CHAMBERS), figsize=(13, 7))

    for axis, chamber in zip(axes, CHAMBERS):
        panel = panels[chamber].iloc[::-1]
        positions = range(len(panel))

        axis.hlines(y=positions, xmin=0, xmax=panel['n_speeches'],
                    color='0.8', linewidth=1, zorder=1)
        axis.scatter(panel['n_speeches'], positions,
                     s=45, color='#1f4e79', zorder=2)

        axis.set_yticks(list(positions))
        axis.set_yticklabels(panel['label'], fontsize=9)
        axis.set_xlabel('Speeches')
        axis.set_title(f"{chamber.capitalize()} ({len(panel)} shown)")
        axis.set_xlim(left=0)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', color='0.9', linewidth=0.8)
        axis.set_axisbelow(True)

    figure.suptitle(f"Legislators giving the most speeches, {year}", fontsize=13)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)

    print(f"Wrote {out_path}")


def make_placeholder(diagnostics, out_path):
    figure, axis = plt.subplots(figsize=(13, 7))
    axis.axis('off')

    axis.text(0.5, 0.95, 'PLACEHOLDER - FIGURE NOT PRODUCED',
              ha='center', va='top', fontsize=15, fontweight='bold', color='#7f1d1d')
    axis.text(0.02, 0.82, '\n'.join(diagnostics),
              ha='left', va='top', fontsize=10, family='monospace')

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)

    print(f"Wrote placeholder {out_path}")


def write_counts(panels, out_path):
    columns = ['chamber', 'icpsr', 'lastname', 'firstname', 'state_abbrev', 'n_speeches']

    if panels:
        table = pd.concat([panels[chamber] for chamber in CHAMBERS], ignore_index=True)
        table = table[columns]
    else:
        table = pd.DataFrame(columns=columns)

    table.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(table)} rows)")


def describe_id_range(frame, label):
    numbers = (frame['speaker_id'].dropna().astype(str)
                                 .str.extract(r"_(\d+)$", expand=False)
                                 .dropna().astype(int))

    if numbers.empty:
        return f"  {label}: no speaker_id values"

    return f"  {label}: n={len(numbers)}, speaker_id suffix {numbers.min()}-{numbers.max()}"


def diagnose(year, speeches, identified, match_rate, empty_chambers):
    return [
        f"Year requested: {year}",
        f"Speeches in {year}: {len(speeches)}",
        f"Identified speakers available: {len(identified)}",
        f"speaker_id match rate: {match_rate:.4f} (need >= {MIN_MATCH_RATE})",
        f"Chambers with no speakers: {', '.join(empty_chambers)}",
        "",
        "speaker_id is a positional surrogate key, assigned by row order when",
        "speaker_disambiguation runs. Regenerating the upstream sections/speeches",
        "files renumbers it, so identified_speakers_*.csv produced by an earlier",
        "run no longer joins to speeches_*.csv.",
        "",
        describe_id_range(speeches, 'speeches'),
        describe_id_range(identified, 'identified_speakers'),
        "",
        "To fix: re-run source/inference/speaker_disambiguation against the current",
        "sections/speeches files, then re-run this script. Years known to be",
        "affected: 1873, 1901, 1902, 1903, 1904.",
    ]


def main(args):
    sections = load_year_family(args.inference_dir, 'sections', args.year)
    speeches = load_year_family(args.inference_dir, 'speeches', args.year)
    identified = load_year_family(args.inference_dir, 'identified_speakers', args.year)

    speeches = select_year(speeches, sections, args.year)

    match_rate = compute_match_rate(speeches, identified)
    print(f"speaker_id match rate: {match_rate:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    figure_path = os.path.join(args.out_dir, 'top_speakers.png')
    counts_path = os.path.join(args.out_dir, 'top_speakers.csv')

    panels = {}
    if match_rate >= MIN_MATCH_RATE:
        panels = top_speakers_by_chamber(build_speech_person_table(speeches, identified),
                                         args.top_n)

    empty_chambers = [chamber for chamber in CHAMBERS if len(panels.get(chamber, [])) == 0]

    # Every declared target must be written, or scons treats the build as failed.
    if empty_chambers:
        diagnostics = diagnose(args.year, speeches, identified, match_rate, empty_chambers)
        for line in diagnostics:
            print(line)
        make_placeholder(diagnostics, figure_path)
        write_counts({}, counts_path)
        return 0

    make_dotplot(panels, args.year, figure_path)
    write_counts(panels, counts_path)
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
