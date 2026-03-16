import os
import re
import sys
from math import floor

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz


STATE_ABBREV_PREFIXES = {
    'Alab': 'AL', 'Alas': 'AK', 'Ariz': 'AZ', 'Arka': 'AR', 'Cali': 'CA',
    'Colo': 'CO', 'Conn': 'CT', 'Dela': 'DE', 'Flor': 'FL', 'Geor': 'GA',
    'Hawa': 'HI', 'Idah': 'ID', 'Illi': 'IL', 'Indi': 'IN', 'Iowa': 'IA',
    'Kans': 'KS', 'Kent': 'KY', 'Loui': 'LA', 'Main': 'ME', 'Mary': 'MD',
    'Mass': 'MA', 'Mich': 'MI', 'Minn': 'MN', 'Missi': 'MS', 'Misso': 'MO',
    'Mont': 'MT', 'Nebr': 'NE', 'Neva': 'NV', 'New H': 'NH', 'New J': 'NJ',
    'New M': 'NM', 'New Y': 'NY', 'North C': 'NC', 'North D': 'ND', 'Ohio': 'OH',
    'Okla': 'OK', 'Oreg': 'OR', 'Penn': 'PA', 'Rhod': 'RI', 'South C': 'SC',
    'South D': 'SD', 'Tenn': 'TN', 'Texa': 'TX', 'Utah': 'UT', 'Verm': 'VT',
    'Virg': 'VA', 'Wash': 'WA', 'West': 'WV', 'Wisc': 'WI', 'Wyom': 'WY'
}

LEGISLATOR_COLUMNS = [
    'lastname', 'firstname', 'nickname', 'chamber', 'congress',
    'icpsr', 'district_code', 'state_abbrev', 'gender'
]


def normalize_name_columns(df):
    for column in ['last_name', 'first_name', 'nickname']:
        normalized = df[column].str.lower()
        normalized = normalized.str.replace('’', "'", regex=False)
        df[column.replace('_name', 'name') if column != 'nickname' else column] = normalized

    df = df.rename(columns={'last_name': 'last_name', 'first_name': 'first_name'})
    df['lastname'] = df['last_name'].str.lower().str.replace('’', "'", regex=False)
    df['firstname'] = df['first_name'].str.lower().str.replace('’', "'", regex=False)
    df['nickname'] = df['nickname'].str.lower().str.replace('’', "'", regex=False)
    return df


def load_legislators(inference_dir):
    legislators = pd.read_csv(inference_dir + "/congress_legislators.csv")
    legislators = normalize_name_columns(legislators)
    return legislators[LEGISLATOR_COLUMNS]


def extract_year(filename):
    match = re.search(r'\d{4}', filename)
    if match:
        return int(match.group())
    print("No four-digit sequence found.")
    return 0


def extract_part(filename):
    match = re.search(r'pt(\d+)', filename)
    if match:
        return int(match.group(1))
    print("No part found found.")
    return 0


def doc_to_yearpart(filename):
    return extract_year(filename), extract_part(filename)


def congress_from_year(year):
    return floor((year + 1) / 2) - 894


def load_dta(filename):
    speech_dta = pd.read_csv(filename)
    year_match = re.search(r'(\d{4})', filename)

    if not year_match:
        raise ValueError(f"Unable to extract year from filename: {filename}")

    year = int(year_match.group(1))
    congress = congress_from_year(year)
    return speech_dta, congress


def fuzzy_contains(text, target, threshold):
    if pd.isna(text):
        return False

    text = str(text)
    target = str(target)

    words = text.split()
    for i in range(len(words)):
        for j in range(i, len(words)):
            substr = ' '.join(words[i:j + 1])
            if fuzz.ratio(substr.lower(), target.lower()) >= (100 - threshold * 10):
                return True
    return False


def extract_speaker(text):
    if pd.isna(text):
        return None

    text = str(text)
    patterns = [
        r'(?:By\s+)?M[rs]\.?\s+([A-Z]+(?:\s+[A-Z]+)*)',
        r'(?:By\s+)?[rRfF]\s+M[rs]\.?\s+([A-Z]+(?:\s+[A-Z]+)*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_state(text):
    if pd.isna(text):
        return None

    s = str(text)
    match = re.search(
        r'\bof\s+(?:the\s+)?([A-Za-z.\-\s]+?)(?=[,:;.()\[\]]|$)',
        s,
        flags=re.IGNORECASE
    )
    if not match:
        return None

    raw = match.group(1).strip()
    raw = re.sub(r'\s+', ' ', raw)
    return raw


def assign_positions(speech_dta, levenshtein_threshold):
    speech_dta['position'] = 0
    speech_dta['gender'] = ''

    speech_dta.loc[
        speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Speaker', levenshtein_threshold)),
        'position'
    ] = 2

    speech_dta.loc[
        speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Vice-Pres', levenshtein_threshold)),
        'position'
    ] = 3

    speech_dta.loc[
        (
            speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'President', levenshtein_threshold))
        ) & (
            ~speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Vice-Pres', levenshtein_threshold))
        ) & (
            speech_dta['position'] == 0
        ),
        'position'
    ] = 4

    speech_dta.loc[
        speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Presiding Officer', levenshtein_threshold)),
        'position'
    ] = 5

    speech_dta.loc[
        speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Mr.', levenshtein_threshold)),
        'gender'
    ] = 'M'
    speech_dta.loc[
        speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Mrs.', levenshtein_threshold)),
        'gender'
    ] = 'F'
    speech_dta.loc[
        speech_dta['speaker_name'].apply(lambda x: fuzzy_contains(x, 'Ms.', levenshtein_threshold)),
        'gender'
    ] = 'F'

    return speech_dta


def assign_speakers_and_states(speech_dta):
    speech_dta.loc[speech_dta['position'] == 0, 'speaker'] = (
        speech_dta.loc[speech_dta['position'] == 0, 'speaker_name'].apply(extract_speaker)
    )
    speech_dta.loc[speech_dta['speaker'].notna(), 'position'] = 1

    speech_dta.loc[speech_dta['position'] == 1, 'state'] = (
        speech_dta.loc[speech_dta['position'] == 1, 'speaker_name'].apply(extract_state)
    )

    speech_dta['speaker'] = speech_dta['speaker'].str.lower()
    return speech_dta


def clean_speakers(speech_dta, levenshtein_threshold=2):
    speech_dta = assign_positions(speech_dta, levenshtein_threshold)
    speech_dta = assign_speakers_and_states(speech_dta)
    return speech_dta


def match_state(state_str):
    if pd.isnull(state_str):
        return ''

    state_str = state_str.capitalize()
    for prefix, abbrev in STATE_ABBREV_PREFIXES.items():
        if state_str.startswith(prefix):
            return abbrev
    return ''


def prepare_legislators_for_congress(legislators, congress):
    legislators = legislators[legislators['congress'] == congress].reset_index(drop=True)
    legislators = legislators.drop_duplicates(subset='icpsr', keep='first')
    return legislators


def attach_state_abbrev(speech_dta):
    speech_dta['state_abbrev'] = speech_dta['state'].apply(match_state)
    return speech_dta


def exact_match_row(row, legislators):
    matches = legislators[legislators['lastname'] == row['speaker']]

    if len(matches) == 1:
        return matches.iloc[0].to_dict()

    if len(matches) > 1:
        state_matches = matches[matches['state_abbrev'] == row['state_abbrev']]
        if len(state_matches) == 1:
            return state_matches.iloc[0].to_dict()
        if len(state_matches) > 0:
            gender_matches = state_matches[state_matches['gender'] == row['gender']]
            if len(gender_matches) == 1:
                return gender_matches.iloc[0].to_dict()
            return state_matches.iloc[np.random.randint(len(state_matches))].to_dict()

    return {}


def join_exact_matches(speech_dta, exact_matches):
    exact_match_df = pd.DataFrame(exact_matches.tolist(), index=speech_dta.index)

    if exact_match_df.empty:
        exact_match_df = pd.DataFrame(index=speech_dta.index)

    if 'state_abbrev' in exact_match_df.columns:
        exact_match_df = exact_match_df.rename(columns={'state_abbrev': 'state_abbrev_match'})

    for column in exact_match_df.columns:
        exact_match_df[column] = exact_match_df[column].replace('', np.nan)

    speech_dta = speech_dta.join(exact_match_df, rsuffix='_match')

    if 'icpsr' not in speech_dta.columns:
        speech_dta['icpsr'] = ''

    speech_dta['state_abbrev'] = speech_dta.get(
        'state_abbrev',
        pd.Series(index=speech_dta.index, dtype=object)
    )
    speech_dta['state_abbrev'] = speech_dta['state_abbrev'].replace('', np.nan)

    if 'state_abbrev_match' in speech_dta.columns:
        mask = speech_dta['state_abbrev'].isna()
        speech_dta.loc[mask, 'state_abbrev'] = speech_dta.loc[mask, 'state_abbrev_match']
        speech_dta.drop(columns=['state_abbrev_match'], inplace=True, errors='ignore')

    return speech_dta


def fuzzy_legislator_match(row, legislators, threshold):
    best_match = None
    best_score = 0

    speaker = str(row['speaker']) if pd.notna(row['speaker']) else ''

    if row['state_abbrev']:
        candidate_legislators = legislators[legislators['state_abbrev'] == row['state_abbrev']]
    else:
        candidate_legislators = legislators

    if speaker:
        for _, legislator_row in candidate_legislators.iterrows():
            score = fuzz.ratio(row['speaker'], legislator_row['lastname'])
            if score > best_score and score >= (100 - threshold * 10):
                best_score = score
                best_match = legislator_row

    return best_match


def fuzzy_match_wrapper(row, legislators, threshold):
    match = fuzzy_legislator_match(row, legislators, threshold)
    if match is not None:
        return pd.Series(match)
    return pd.Series([None] * len(legislators.columns), index=legislators.columns)


def run_baseline_fuzzy_matching(speech_dta, legislators):
    unmatched = speech_dta[speech_dta['icpsr'] == '']
    matched_1 = unmatched.apply(lambda row: fuzzy_match_wrapper(row, legislators, 1), axis=1)
    speech_dta.update(matched_1)

    still_unmatched = speech_dta[speech_dta['icpsr'] == '']
    matched_2 = still_unmatched.apply(lambda row: fuzzy_match_wrapper(row, legislators, 2), axis=1)
    speech_dta.update(matched_2)

    print(f"Matches after fuzzy matching: {(speech_dta['icpsr'] != '').sum()} out of {len(speech_dta)} total rows")
    return speech_dta


def merge_speakers_with_legislators(speech_dta, legislators):
    speech_dta['icpsr'] = pd.to_numeric(speech_dta['icpsr'], errors='coerce').astype('Int64')
    legislators['icpsr'] = pd.to_numeric(legislators['icpsr'], errors='coerce').astype('Int64')

    merged_dta = speech_dta.merge(legislators, on='icpsr', how='left', suffixes=('', '_legislators'))

    print(f"Rows in speech_dta: {len(speech_dta)}")
    print(f"Rows in merged_dta: {len(merged_dta)}")
    print(f"Matched rows: {merged_dta['icpsr'].notna().sum()}")

    return merged_dta


def get_speakers(legislators, speech_dta, congress, algorithm):
    np.random.seed(1)

    speech_dta = speech_dta[speech_dta['position'] != 0].reset_index(drop=True)
    legislators = prepare_legislators_for_congress(legislators, congress)
    speech_dta = attach_state_abbrev(speech_dta)

    exact_matches = speech_dta.apply(lambda row: exact_match_row(row, legislators), axis=1)
    speech_dta = join_exact_matches(speech_dta, exact_matches)

    print(f"Exact matches found: {(exact_matches.apply(bool)).sum()} out of {len(speech_dta)} total rows")

    if algorithm == 'baseline':
        speech_dta = run_baseline_fuzzy_matching(speech_dta, legislators)
    elif algorithm == 'linktransformer':
        pass

    return merge_speakers_with_legislators(speech_dta, legislators)


def load_chunk_file(chunk_file):
    if os.path.isfile(chunk_file):
        return pd.read_csv(chunk_file)
    raise RuntimeError("List of files to process does not exist. Run identify_speakers/split_df_for_jobs.py")


def build_input_path(inference_dir, title):
    return os.path.join(inference_dir, title)


def build_output_path(file_path, year, part):
    return os.path.join(
        os.path.dirname(file_path),
        f"identified_speakers_{year}_pt{part}.csv"
    )


def process_document_row(row, legislators, inference_dir):
    year, part = doc_to_yearpart(row['title'])
    file_path = build_input_path(inference_dir, row["title"])
    print(file_path)

    speech_dta, congress = load_dta(file_path)
    speech_dta = clean_speakers(speech_dta)
    merged_dta = get_speakers(legislators, speech_dta, congress, algorithm='baseline')

    output_path = build_output_path(file_path, year, part)
    merged_dta.to_csv(output_path, index=False)


def main(chunk_file, inference_dir):
    docs_df = load_chunk_file(chunk_file)
    legislators = load_legislators(inference_dir)

    for index, row in docs_df.iterrows():
        if row['complete'] == 1:
            process_document_row(row, legislators, inference_dir)
            docs_df.loc[index, 'complete'] = 2
            docs_df.to_csv(chunk_file, index=False)


if __name__ == "__main__":
    chunk_file = sys.argv[1]
    inference_dir = sys.argv[2]
    main(chunk_file, inference_dir)