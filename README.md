## CongressText

This repository is built from [`JMSLab/Template`](https://github.com/JMSLab/Template/tree/d39df7414aad471b3df214f6684191496b2a90fd) and by default shares its dependencies and prerequisites.

The datastore is [`CongressText`](https://drive.google.com/drive/u/1/folders/0ALDmSBTLfB78Uk9PVA).

Many build steps are excluded from the `scons` DAG because they require external scraping or heavy execution on a cluster.

### Workflow summary

This repository produces text data containing the speeches contained in the Congressional Record. 

The steps are outlined below.

#### Detect page layouts and OCR text for the (historical) bound Congressional Record
1. Scrape the Congress.gov website for PDF scans of the [bound](./source/scrape/scrape_bound.py) and [daily](./source/scrape/scrape_daily.py) Congressional Record
2. [Manually label](./source/label/README.md) the layouts for a random sample of PDFs using LabelStudio
3. [Fine-tune](./source/training/tools/train_net.py) a RCNN model to automatically detect layouts using LayoutParser
4. [Batch](./source/inference/historical/split_df_for_jobs.py) data for separate jobs.
5. In [one script](./source/inference/historical/submit_jobs.sh), detect layouts for all PDFs using this model, OCR to obtain text data for all PDFs, and clean text data into final speech data. 

#### Identify and disambiguate speakers in the bound Congressional Record
1. [Batch](./source/inference/speaker_disambiguation/split_df_for_jobs.py) data for separate jobs.
2. [Identify](./source/inference/speaker_disambiguation/submit_jobs.sh) speakers in the bound Congressional record.

#### Convert (modern) daily Congressional Record to the historical schema
1. [Batch](./source/inference/speaker_disambiguation/build_and_chunk_daily_docs.py) data for separate jobs.
2. [Parse](./source/inference/daily/submit_jobs_parse_daily.sh) daily Congressional record, using the [congressional-record](https://github.com/unitedstates/congressional-record/tree/main) repository.
3. [Convert](./source/inference/daily/submit_jobs_daily_to_historical_schema.sh) from daily to historical schema.

### Output structure

Key processed files for the (historical) bound Congressional Record are in `datastore/inference`:
* `sections_YYYY_ptP.csv`: `section_id` (key), `year`, `part_page` (foreign key), metadata
* `speeches_YYYY_ptP.csv`: `speech_id` (key), `section_id` (foreign key), `speaker_id` (foreign key)
* `paragraphs_YYYY_ptP.csv`: `paragraph_id` (key), `speech_id` (foreign key), `paragraph_order`, `paragraph_text`
* `speakers_YYYY_ptP.csv`: `speaker_id` (key), `speaker_name`
* `identified_speakers_YYYY_ptP.csv`: `speaker_id` (key), `icpsr` (foreign key), metadata

Files for the (modern) daily Congressional Record, converted to the historical schema, are in `datastore/inference/daily_harmonized/YYYY` and follow the same schema, with `P` a date rather than an integer (e.g. `sections_2022_pt20220104.csv`):
* `sections_YYYY_ptP.csv`: `section_id` (key), `year`, `part_page` (foreign key), metadata
* `speeches_YYYY_ptP.csv`: `speech_id` (key), `section_id` (foreign key), `speaker_id` (foreign key)
* `paragraphs_YYYY_ptP.csv`: `paragraph_id` (key), `speech_id` (foreign key), `paragraph_order`, `paragraph_text`
* `speakers_YYYY_ptP.csv`: `speaker_id` (key), `speaker_name`
* `identified_speakers_YYYY_ptP.csv`: `speaker_id` (key), `icpsr` (foreign key), metadata

`YYYY` spans 1873-1998 for the bound record and 1994-2025 for the daily record.

Note that `speaker_id` identifies a speaker *name as it appears in the text*, while `icpsr` identifies an actual legislator; several `speaker_id` may map to one `icpsr`.

#### Reading these files

Within a job, each file accumulates every document processed so far, and `P` names the last document processed rather than the file's contents. A file may therefore contain rows from several years, and a year's parts overlap heavily. To read a year, concatenate all of its parts, drop duplicates on the key column, and select the year via `sections.year` rather than the filename. See `load_year_family` in [`source/analysis/plot.py`](./source/analysis/plot.py) for an example.

To illustrate the contents of the files, [this script](./source/analysis/plot.py) plots the legislators who gave the most speeches, separately by chamber, for the year 1895.


