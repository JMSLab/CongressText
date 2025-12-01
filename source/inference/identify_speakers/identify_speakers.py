import os
import pandas as pd
import pdb
import re
from fuzzywuzzy import fuzz
from math import floor 
import numpy as np
import sys


# always block merge on year
# test a few different methods, potentially combine

inference_dir = 'datastore/inference'

# returns ground truth data
def load_legislators():
    legislators = pd.read_csv(inference_dir + "/congress_legislators.csv")

    # text to lower case
    legislators['lastname'] = legislators['last_name'].str.lower()
    legislators['firstname'] = legislators['first_name'].str.lower()
    legislators['nickname'] = legislators['nickname'].str.lower()

    # replace '’' with "'"
    legislators['lastname'] = legislators['lastname'].str.replace('’', "'", regex=False)
    legislators['firstname'] = legislators['firstname'].str.replace('’', "'", regex=False)
    legislators['nickname'] = legislators['nickname'].str.replace('’', "'", regex=False)

    columns_to_keep = ['lastname', 'firstname', 'nickname', 'chamber', 'congress', 'icpsr', 'district_code', 'state_abbrev', 'gender']
    legislators = legislators[columns_to_keep]

    return legislators


def doc_to_yearpart(filename):
    """Get year and part number from filename  """
    year = 0
    match = re.search(r'\d{4}', filename)
    if match:
        year = int(match.group())
    else:
        print("No four-digit sequence found.") 


    part = 0
    match = re.search(r'pt(\d+)', filename)
    if match:
        part = int(match.group(1))
    else:
        print("No part found found.") 

    return year, part


# get Congress speech data
def load_dta(filename):
    speech_dta = pd.read_csv(filename)

    # extract year, while filename formats are like 'speakers_1882_pt1.csv'
    year_match = re.search(r'(\d{4})', filename)
    if year_match:
        year = int(year_match.group(1))
    else:
        raise ValueError(f"Unable to extract year from filename: {filename}")

    # calculate congress number
    congress = floor((year+1)/2) - 894

    return speech_dta, congress


# clean speakers data
def clean_speakers(speech_dta, levenshtein_threshold=2):
    # ID special positions:
    # generate a column `position` with values 2-5 based on whether they are:
    # speaker, vice-president, president pro tempore, presiding officer
    speech_dta['position'] = 0
    # add gender column
    speech_dta['gender'] = ''

    def fuzzy_match(text, target, threshold):
        # convert input to string, handling NaN and other types
        if pd.isna(text):
            return False
        text = str(text)
        target = str(target)

        words = text.split()
        for i in range(len(words)):
            for j in range(i, len(words)):
                substr = ' '.join(words[i:j+1])
                if fuzz.ratio(substr.lower(), target.lower()) >= (100 - threshold * 10):
                    return True
        return False

    # first, anything with speaker is labeled speaker
    speech_dta.loc[speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Speaker', levenshtein_threshold)), 'position'] = 2

    # then, anything with vice-president
    speech_dta.loc[speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Vice-Pres', levenshtein_threshold)), 'position'] = 3

    # then, anything else containing president but not vice-president
    speech_dta.loc[(speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'President', levenshtein_threshold))) & 
                   (~speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Vice-Pres', levenshtein_threshold))) &
                   (speech_dta['position'] == 0), 'position'] = 4

    # then, anything that contains presiding 
    speech_dta.loc[speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Presiding Officer', levenshtein_threshold)), 'position'] = 5

    # gender is "M" if contains substring "Mr." or "mr."
    # gender is "M" if contains substring "Mrs." or "mrs." or "Ms." or "ms."
    speech_dta.loc[speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Mr.', levenshtein_threshold)), 'gender'] = 'M'
    speech_dta.loc[speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Mrs.', levenshtein_threshold)), 'gender'] = 'F'
    speech_dta.loc[speech_dta['speaker_name'].apply(lambda x: fuzzy_match(x, 'Ms.', levenshtein_threshold)), 'gender'] = 'F'

    # get list of actual speakers. strip punctuation
    def extract_speaker(text):
        # convert input to string, handling NaN and other types
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

    speech_dta.loc[speech_dta['position'] == 0, 'speaker'] = speech_dta.loc[speech_dta['position'] == 0, 'speaker_name'].apply(extract_speaker)
    speech_dta.loc[speech_dta['speaker'].notna(), 'position'] = 1

    # for those which position == 1 , extract new column state, where possible
    def extract_state(text):
        if pd.isna(text):
            return None
        s = str(text)

        # capture the chunk after "of ..." up to punctuation/end
        m = re.search(r'\bof\s+(?:the\s+)?([A-Za-z.\-\s]+?)(?=[,:;.()\[\]]|$)', s, flags=re.IGNORECASE)
        if not m:
            return None
        raw = m.group(1).strip()

        # normalize spacing/punctuation
        raw = re.sub(r'\s+', ' ', raw)
        return raw


    speech_dta.loc[speech_dta['position'] == 1, 'state'] = speech_dta.loc[speech_dta['position'] == 1, 'speaker_name'].apply(extract_state)

    # text `speaker` to lower case
    speech_dta['speaker'] = speech_dta['speaker'].str.lower()

    return speech_dta


# get speakers associated with each speech
def get_speakers(legislators,speech_dta,congress,algorithm):
    np.random.seed(1)

    # step 0: drop anything with position = 0
    # TODO: also match positions 2-5 using manual dictionary (skip for now)
    speech_dta = speech_dta[speech_dta['position'] != 0].reset_index(drop=True)

    # step 1: block match on session of Congress
    # only keep rows in `legislators` that exactly match `congress`
    legislators = legislators[legislators['congress'] == congress].reset_index(drop=True)
    legislators = legislators.drop_duplicates(subset='icpsr', keep='first')

    # step 2: match to state
    # legislators `state_abbrev` is the two letter abbrevation
    # speech_dta `state` contains the state (if exists), but often is truncated
        # for example, "New Yo" instead of "New York" or "Cal" instead of "California"
        # add state_abbrev as a column to speech_dta with the closest match
        # or empty if no matches

    # minimal set required
    state_abbrev = {'Alab': 'AL', 'Alas': 'AK', 'Ariz': 'AZ', 'Arka': 'AR', 'Cali': 'CA',
                    'Colo': 'CO', 'Conn': 'CT', 'Dela': 'DE', 'Flor': 'FL', 'Geor': 'GA',
                    'Hawa': 'HI', 'Idah': 'ID', 'Illi': 'IL', 'Indi': 'IN', 'Iowa': 'IA',
                    'Kans': 'KS', 'Kent': 'KY', 'Loui': 'LA', 'Main': 'ME', 'Mary': 'MD',
                    'Mass': 'MA', 'Mich': 'MI', 'Minn': 'MN', 'Missi': 'MS', 'Misso': 'MO',
                    'Mont': 'MT', 'Nebr': 'NE', 'Neva': 'NV', 'New H': 'NH', 'New J': 'NJ',
                    'New M': 'NM', 'New Y': 'NY', 'North C': 'NC', 'North D': 'ND', 'Ohio': 'OH',
                    'Okla': 'OK', 'Oreg': 'OR', 'Penn': 'PA', 'Rhod': 'RI', 'South C': 'SC',
                    'South D': 'SD', 'Tenn': 'TN', 'Texa': 'TX', 'Utah': 'UT', 'Verm': 'VT',
                    'Virg': 'VA', 'Wash': 'WA', 'West': 'WV', 'Wisc': 'WI', 'Wyom': 'WY'}

    def match_state(state_str):
        if pd.isnull(state_str):
            return ''
        state_str = state_str.capitalize()
        for prefix, abbrev in state_abbrev.items():
            if state_str.startswith(prefix):
                return abbrev
        return ''

    speech_dta['state_abbrev'] = speech_dta['state'].apply(match_state)


    # sum([e is None for e in exact_matches])

    # step 3: get exact matches
    # for each row in speech_dta, see if there is exactly 1 row in legislators `lastname` 
    # that matches this row's `speaker`
    # if no matches, skip. 
    # if multiple matches, see if exactly 1 match when also including state_abbrev
        # use this match if exactly 1
        # if still multiple, or no state match, pick random to match
        # TODO: decide if this can be improved

    def exact_match(row):
        matches = legislators[legislators['lastname'] == row['speaker']]
        if len(matches) == 1:
            return matches.iloc[0].to_dict()
        elif len(matches) > 1:
            state_matches = matches[matches['state_abbrev'] == row['state_abbrev']]
            if len(state_matches) == 1:
                return state_matches.iloc[0].to_dict()
            elif len(state_matches) > 0:
                # match on gender
                gender_matches = state_matches[state_matches['gender'] == row['gender']]
                if len(gender_matches) == 1:
                    return gender_matches.iloc[0].to_dict()
                else:
                    return state_matches.iloc[np.random.randint(len(state_matches))].to_dict()
                
        return {}  # return empty dict if no match

    exact_matches = speech_dta.apply(exact_match, axis=1)

   # Convert the results to a DataFrame
    exact_match_df = pd.DataFrame(exact_matches.tolist(), index=speech_dta.index)

    # Ensure the DF has the columns we expect, even if there were zero matches
    if exact_match_df.empty:
        exact_match_df = pd.DataFrame(index=speech_dta.index)

    # Rename overlap and normalize empties to NaN
    if 'state_abbrev' in exact_match_df.columns:
        exact_match_df = exact_match_df.rename(columns={'state_abbrev': 'state_abbrev_match'})
    for c in exact_match_df.columns:
        exact_match_df[c] = exact_match_df[c].replace('', np.nan)

    # Join the results
    speech_dta = speech_dta.join(exact_match_df, rsuffix='_match')

    # Make sure 'icpsr' exists for the "unmatched" filter used later
    if 'icpsr' not in speech_dta.columns:
        speech_dta['icpsr'] = ''

    # Coalesce state_abbrev with state_abbrev_match (treat '' as missing)
    speech_dta['state_abbrev'] = speech_dta.get('state_abbrev', pd.Series(index=speech_dta.index, dtype=object))
    speech_dta['state_abbrev'] = speech_dta['state_abbrev'].replace('', np.nan)
    if 'state_abbrev_match' in speech_dta.columns:
        mask = speech_dta['state_abbrev'].isna()
        speech_dta.loc[mask, 'state_abbrev'] = speech_dta.loc[mask, 'state_abbrev_match']
        speech_dta.drop(columns=['state_abbrev_match'], inplace=True, errors='ignore')


    print(f"Exact matches found: {(exact_matches.apply(bool)).sum()} out of {len(speech_dta)} total rows")



    # step 4: do remaining matches
    # allow for different algorithms to be used
    # baseline algorithm: based on fuzzy string matches, see if any exact match
        # start with levenshtein_threshold of 1, move to 2 if necessary
    # alternative algorithm: call LinkTransformer (function linktransformer_match())

    def fuzzy_match(row, threshold):
        best_match = None
        best_score = 0

        # Convert row['speaker'] to string, handling NaN
        speaker = str(row['speaker']) if pd.notna(row['speaker']) else ''
        
        # Filter legislators based on state if available
        if row['state_abbrev']:
            state_filtered_legislators = legislators[legislators['state_abbrev'] == row['state_abbrev']]
        else:
            state_filtered_legislators = legislators
        
        if speaker:
            for _, v_row in state_filtered_legislators.iterrows():
                score = fuzz.ratio(row['speaker'], v_row['lastname'])
                if score > best_score and score >= (100 - threshold * 10):
                    best_score = score
                    best_match = v_row
            
        if best_match is not None:
            return best_match
        return None

    if algorithm == 'baseline':
        unmatched = speech_dta[speech_dta['icpsr'] == '']

        def fuzzy_match_wrapper(row, threshold):
            match = fuzzy_match(row, threshold)
            if match is not None:
                return pd.Series(match)
            return pd.Series([None] * len(legislators.columns), index=legislators.columns)

        # First round of matching with threshold 1
        matched_1 = unmatched.apply(lambda row: fuzzy_match_wrapper(row, 1), axis=1)
        
        # Update speech_dta with matched results
        speech_dta.update(matched_1)

        # Second round of matching with threshold 2
        still_unmatched = speech_dta[speech_dta['icpsr'] == '']
        matched_2 = still_unmatched.apply(lambda row: fuzzy_match_wrapper(row, 2), axis=1)
        
        # Update speech_dta with matched results
        speech_dta.update(matched_2)

        print(f"Matches after fuzzy matching: {(speech_dta['icpsr'] != '').sum()} out of {len(speech_dta)} total rows")
        
    elif algorithm == 'linktransformer':
        # Implement LinkTransformer matching here
        # figure out score threshold
        pass


    # step 5: conduct merge of legislators with speech_dta

    # ensure 'icpsr' is of the same type in both DataFrames
    speech_dta['icpsr'] = pd.to_numeric(speech_dta['icpsr'], errors='coerce').astype('Int64')
    legislators['icpsr'] = pd.to_numeric(legislators['icpsr'], errors='coerce').astype('Int64')

    # merge
    merged_dta = speech_dta.merge(legislators, on='icpsr', how='left', suffixes=('', '_legislators'))

    # merge statistics
    print(f"Rows in speech_dta: {len(speech_dta)}")
    print(f"Rows in merged_dta: {len(merged_dta)}")
    print(f"Matched rows: {merged_dta['icpsr'].notna().sum()}")

    return merged_dta



# example filename: filename = 'speakers_1882_pt1.csv'
def main(chunk_file):

    docs_df = None

    # if exists, load data
    if os.path.isfile(chunk_file):
        docs_df = pd.read_csv(chunk_file)
    # else, raise error
    else:
        raise RuntimeError("List of files to process does not exist. Run identify_speakers/split_df_for_jobs.py")

    # get legislator data
    legislators = load_legislators()

    # iterate through incomplete files
    for index, row in docs_df.iterrows():
        if row['complete'] == 1:
            year, part = doc_to_yearpart(row['title'])
            file_path = f"{inference_dir}/speakers_{year}_pt{part}.csv"
            print(file_path)

            speech_dta, congress = load_dta(file_path)

            speech_dta = clean_speakers(speech_dta)
            merged_dta = get_speakers(legislators,speech_dta, congress,algorithm='baseline')


            speakers_new_path = f"{inference_dir}/identified_speakers_{year}_pt{part}.csv"
            merged_dta.to_csv(speakers_new_path, index=False)

            # read docs_df and update to track progress
            docs_df.loc[index, 'complete'] = 2
            docs_df.to_csv(chunk_file, index=False)        



## TODO: actual test-train set for different algorithms

if __name__ == "__main__":
    chunk_file = sys.argv[1]
    main(chunk_file)







