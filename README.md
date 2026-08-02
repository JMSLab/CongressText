## CongressText

This repository is built from [`JMSLab/Template`](https://github.com/JMSLab/Template/tree/d39df7414aad471b3df214f6684191496b2a90fd) and by default shares its dependencies and prerequisites.

The datastore is [`CongressText`](https://drive.google.com/drive/u/1/folders/0ALDmSBTLfB78Uk9PVA).

### Workflow summary

This repository produces text data containing the speeches contained in the Congressional Record. To accomplish this, we use the following procedure:
1. Detect page layouts and OCR text for the (historical) bound Congressional Record
2. Identify speakers in the bound Congressional Record
3. Convert (modern) daily Congressional Record to the historical schema

#### Detect page layouts and OCR text
1. Scrape the Congress.gov website for PDF scans of the [bound](./source/scrape/scrape_bound.py) and [daily](./source/scrape/scrape_daily.py) Congressional Record
2. [Manually label](./source/label/README.md) the layouts for a random sample of PDFs using LabelStudio
3. [Fine-tune](./source/training/tools/train_net.py) a RCNN model to automatically detect layouts using LayoutParser
4. [Batch](./source/inference/historical/split_df_for_jobs.py) data for separate jobs.
5. In [one script](./source/inference/historical/submit_jobs.sh), detect layouts for all PDFs using this model, OCR to obtain text data for all PDFs, and clean text data into final speech data. 

#### Identify and disambiguate speakers
1. [Batch](./source/inference/speaker_disambiguation/split_df_for_jobs.py) data for separate jobs.
2. [Identify](./source/inference/speaker_disambiguation/submit_jobs.sh) speakers in the bound Congressional record.

#### Convert daily Congressional Record to the historical schema
1. [Batch](./source/inference/speaker_disambiguation/build_and_chunk_daily_docs.py) data for separate jobs.
2. [Parse](./source/inference/daily/submit_jobs_parse_daily.sh) daily Congressional record, using the [congressional-record](https://github.com/unitedstates/congressional-record/tree/main) repository.
3. [Convert](./source/inference/daily/submit_jobs_daily_to_historical_schema.sh) from daily to historical schema.


### Output structure

Key processed files are in `datastore/output/inference`:
* `sections_YYYY_ptP.csv`: `section_id` (key), `year`, `part_page` (foreign key), metadata
* `speeches_YYYY_ptP.csv`: `speech_id` (key), `section_id` (foreign key), `speaker_id` (foreign key)
* `paragraphs_YYYY_pyP.csv`: `paragraph_id` (key), `speech_id` (foreign key), `paragraph_order`, `paragraph_text`
* `identified_speakers_YYYY_ptP.csv`: `speaker_id` (key), `icpsr` (foreign key), metadata

