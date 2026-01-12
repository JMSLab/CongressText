import os
import re
import sys
import cv2
import numpy as np
import pandas as pd
import pymupdf
import pytesseract
import layoutparser as lp

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pdf2image import convert_from_path
from tqdm import tqdm

# from efficient_ocr import EffOCR  # alternative engine, currently unused


# ----------------------------
# Environment / engine setup
# ----------------------------

def setup_tesseract(home_dir: Optional[str] = None) -> None:
    """Configure tesseract binary path and tessdata env vars"""
    if home_dir is None:
        home_dir = os.getenv("HOME", "")

    pytesseract.pytesseract.tesseract_cmd = home_dir + "/tesseract/tesseract"
    os.environ["TESSDATA_PREFIX"] = home_dir + "/tesseract/tessdata"


def build_layout_model(
    config_path: str,
    model_path: str,
    score_thresh: float = 0.8,
) -> lp.Detectron2LayoutModel:
    """Instantiate Detectron2 model"""
    return lp.Detectron2LayoutModel(
        config_path=config_path,
        model_path=model_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
    )


def build_ocr_agent(languages: str = "eng") -> lp.TesseractAgent:
    """Instantiate OCR agent (tesseract, as in original)"""
    return lp.TesseractAgent(languages=languages)


# ----------------------------
# Data / config containers
# ----------------------------

@dataclass(frozen=True)
class InferenceConfig:
    image_dir: str = "datastore/scrape/cr-bound"
    output_dir: str = "output/inference"
    inference_dir: str = "datastore/inference"

    model_config_path: str = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/config.yaml"
    model_weights_path: str = "datastore/training/outputs/fast_rcnn_R_50_FPN_3x_batch3_manual/model_final.pth"
    model_score_thresh: float = 0.8

    engine_type: str = "tesseract"
    chunk_size: int = 100  # part_page in size 100 chunks


# layout detector + OCR
@dataclass
class InferenceEngines:
    model: lp.Detectron2LayoutModel
    ocr_agent: object  # lp.TesseractAgent or other


@dataclass
class InferenceState:
    docs_df: pd.DataFrame
    sections_df: pd.DataFrame
    speeches_df: pd.DataFrame
    speakers_df: pd.DataFrame

    speaker_vals: np.ndarray  # speakers_df['speaker_name'].values (plus any new names appended)


@dataclass
class InferenceContext:
    cfg: InferenceConfig
    engines: InferenceEngines
    state: InferenceState


# ----------------------------
# Helpers (kept compatible)
# ----------------------------

def new_paragraph_df() -> pd.DataFrame:
    """Construct df for paragraph level text. One per document."""
    return pd.DataFrame(columns=["speech_id", "paragraph_text", "paragraph_order", "paragraph_id"])


def pdf_to_cv2_images(pdf_path: str, first_page: int, last_page: int) -> List[np.ndarray]:
    """Convert a PDF file to a list of images for cv2."""
    pil_images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page)

    # validation (original behavior)
    for image in pil_images:
        img_array = np.array(image)
        if img_array is None or not isinstance(img_array, np.ndarray):
            raise ValueError("Conversion to numpy array failed.")
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError("Image is not in the expected format (H x W x 3).")

    # RGB -> BGR for OpenCV
    return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in pil_images]


def doc_to_yearpart(filename: str) -> Tuple[int, int]:
    """Get year and part number from filename."""
    year = 0
    match = re.search(r"\d{4}", filename)
    if match:
        year = int(match.group())
    else:
        print("No four-digit sequence found.")

    part = 0
    match = re.search(r"pt(\d+)", filename)
    if match:
        part = int(match.group(1))
    else:
        print("No part found found.")

    return year, part


def infer_img2txt(engine_type: str, engine: object, image: np.ndarray):
    """Goes from image to text using an OCR engine."""
    text = None
    if engine_type == "tesseract":
        text = engine.detect(image)
    elif engine_type == "effocr":
        text = engine.detect(image)  # engine.infer(image)
    return text


def get_pdf_page_count(pdf_path: str) -> int:
    """Get page count without loading file."""
    doc = pymupdf.open(pdf_path)
    return doc.page_count


def convert_pdf_in_chunks(pdf_path: str, chunk_size: int = 100) -> Iterable[List[np.ndarray]]:
    """Handle PDF chunking for memory limits."""
    total_pages = get_pdf_page_count(pdf_path)
    chunks = [(i, min(i + chunk_size, total_pages)) for i in range(0, total_pages, chunk_size)]
    for start, end in chunks:
        yield pdf_to_cv2_images(pdf_path, first_page=start + 1, last_page=end)


# ----------------------------
# Layout processing
# ----------------------------

def should_process_page(layout: lp.Layout) -> bool:
    """If any skip block (type==2) exists, skip the whole page."""
    return len(layout) > 0 and (not any(item.type == 2 for item in layout))


def split_layout_by_type(layout: lp.Layout) -> Dict[str, lp.Layout]:
    """Split detected blocks into titles/sections/speeches/speakers."""
    titles = lp.Layout([b for b in layout if b.type == 0])
    titles.sort(key=lambda b: b.coordinates[1], inplace=True)

    sections = lp.Layout([b for b in layout if b.type == 1])
    speeches = lp.Layout([b for b in layout if b.type == 4])
    speakers = lp.Layout([b for b in layout if b.type == 3])

    return {"titles": titles, "sections": sections, "speeches": speeches, "speakers": speakers}


def split_blocks_within_title(
    titles: lp.Layout,
    sections: lp.Layout,
    speeches: lp.Layout,
    speakers: lp.Layout,
    title_id: int,
    title,
    page_height: int,
) -> Tuple[List, List, List, lp.Layout, lp.Layout, lp.Layout, lp.Layout]:
    """
    For a given title region, take all blocks above the next title boundary (upper_y),
    then remove them from the remaining lists.
    """
    upper_y = page_height

    # check: only do title boundary logic if titles exist and title isn't last
    if len(titles) > 0 and title != titles[-1]:
        upper_y = titles[title_id + 1].coordinates[1] + 5  # 5 px grace

    section_Tblocks = [b for b in sections if b.coordinates[1] <= upper_y]
    speech_Tblocks = [b for b in speeches if b.coordinates[1] <= upper_y]
    speaker_Tblocks = [b for b in speakers if b.coordinates[1] <= upper_y]

    # remove from original lists (same semantics)
    sections = lp.Layout([b for b in sections if b.coordinates[1] > upper_y])
    speeches = lp.Layout([b for b in speeches if b.coordinates[1] > upper_y])
    speakers = lp.Layout([b for b in speakers if b.coordinates[1] > upper_y])

    all_Tblocks = lp.Layout(section_Tblocks + speech_Tblocks + speaker_Tblocks)
    return section_Tblocks, speech_Tblocks, speaker_Tblocks, all_Tblocks, sections, speeches, speakers


def order_blocks_by_columns(
    all_Tblocks: lp.Layout,
    image: np.ndarray,
    year: int,
) -> lp.Layout:
    """Order blocks by column then y."""
    height, width = image.shape[:2]

    # year >= 1941: 3 columns
    if year >= 1941:
        left_threshold = 0.35 * width
        right_threshold = 0.62 * width

        left_interval = lp.Interval(0, left_threshold, axis="x").put_on_canvas(image)
        middle_interval = lp.Interval(left_threshold, right_threshold, axis="x").put_on_canvas(image)
        right_interval = lp.Interval(right_threshold, width, axis="x").put_on_canvas(image)

        left_blocks = all_Tblocks.filter_by(left_interval, center=True)
        middle_blocks = all_Tblocks.filter_by(middle_interval, center=True)
        right_blocks = all_Tblocks.filter_by(right_interval, center=True)

        left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
        middle_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
        right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

        return left_blocks + middle_blocks + right_blocks

    # year < 1941: 2 columns
    left_interval = lp.Interval(0, width / 2 * 1.05, axis="x").put_on_canvas(image)
    left_blocks = all_Tblocks.filter_by(left_interval, center=True)
    left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    right_blocks = lp.Layout([b for b in all_Tblocks if b not in left_blocks])
    right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    return left_blocks + right_blocks


def order_blocks_with_speaker_fix(
    all_Tblocks: lp.Layout,
    section_Tblocks: List,
    speech_Tblocks: List,
    speaker_Tblocks: List,
) -> List:
    """
    Classify speaker before speech if intersecting.
    Returns a Python list of blocks in reading order.
    """
    ordered_blocks = []
    inserted_speakers = set()

    for block in all_Tblocks:
        if block in section_Tblocks:
            ordered_blocks.append(block)

        elif block in speech_Tblocks:
            best_speaker = None
            max_intersection_area = 0

            for speaker_block in speaker_Tblocks:
                if speaker_block.id not in inserted_speakers:
                    intersection = block.intersect(speaker_block)
                    intersection_area = intersection.area

                    if (
                        intersection_area > 0.7 * speaker_block.area
                        and intersection_area > max_intersection_area
                    ):
                        best_speaker = speaker_block
                        max_intersection_area = intersection_area

            if best_speaker:
                ordered_blocks.append(best_speaker)
                inserted_speakers.add(best_speaker.id)

            ordered_blocks.append(block)

        elif block in speaker_Tblocks:
            if block.id not in inserted_speakers:
                ordered_blocks.append(block)
                inserted_speakers.add(block.id)

    return ordered_blocks


def ocr_block_image(ctx: InferenceContext, block, page_image: np.ndarray):
    """Crop + pad + OCR."""
    segment_image = (
        block.pad(left=1, right=1, top=1, bottom=1)
            .crop_image(page_image)
    )
    return infer_img2txt(ctx.cfg.engine_type, ctx.engines.ocr_agent, segment_image)


# ----------------------------
# IO / progress tracking
# ----------------------------

def build_master_paths(cfg: InferenceConfig) -> Dict[str, str]:
    """Dictionary of file paths."""
    return {
        "docsdf_path": cfg.inference_dir + "/docs.csv",
        "sectionsdf_path": cfg.inference_dir + "/sections.csv",
        "speechesdf_path": cfg.inference_dir + "/speeches.csv",
        "speakersdf_path": cfg.inference_dir + "/speakers.csv",
        "docsdf_log": cfg.output_dir + "/docs.log",
        "sectionsdf_log": cfg.output_dir + "/sections.log",
        "speechesdf_log": cfg.output_dir + "/speeches.log",
        "speakersdf_log": cfg.output_dir + "/speakers.log",
    }


def load_progress_dataframes(chunk_file: str, cfg: InferenceConfig) -> InferenceState:
    """Load chunk_file + base dfs."""
    paths = build_master_paths(cfg)

    if not os.path.isfile(chunk_file):
        raise RuntimeError("List of files to process does not exist. Run split_df_for_jobs.py")

    docs_df = pd.read_csv(chunk_file)
    sections_df = pd.read_csv(paths["sectionsdf_path"])
    speeches_df = pd.read_csv(paths["speechesdf_path"])
    speakers_df = pd.read_csv(paths["speakersdf_path"])

    speaker_vals = speakers_df["speaker_name"].values
    return InferenceState(
        docs_df=docs_df,
        sections_df=sections_df,
        speeches_df=speeches_df,
        speakers_df=speakers_df,
        speaker_vals=speaker_vals,
    )


def mark_doc_complete_and_save_master(
    cfg: InferenceConfig,
    chunk_file: str,
    docs_df: pd.DataFrame,
    docs_master: pd.DataFrame,
) -> None:
    """Persist doc completion flags."""
    docsdf_path = cfg.inference_dir + "/docs.csv"
    docs_df.to_csv(chunk_file, index=False)
    docs_master.to_csv(docsdf_path, index=False)


def save_per_doc_outputs(
    cfg: InferenceConfig,
    year: int,
    part: int,
    paragraphs_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    speeches_df: pd.DataFrame,
    speakers_df: pd.DataFrame,
) -> None:
    """Save outputs."""
    paragraphsdf_path = cfg.inference_dir + f"/paragraphs_{year}_pt{part}.csv"

    section_new_path = f"{cfg.inference_dir}/sections_{year}_pt{part}.csv"
    speech_new_path = f"{cfg.inference_dir}/speeches_{year}_pt{part}.csv"
    speakers_new_path = f"{cfg.inference_dir}/speakers_{year}_pt{part}.csv"

    sections_df.to_csv(section_new_path, index=False)
    speeches_df.to_csv(speech_new_path, index=False)
    speakers_df.to_csv(speakers_new_path, index=False)
    paragraphs_df.to_csv(paragraphsdf_path, index=False)


# ----------------------------
# Document processing
# ----------------------------

@dataclass
class DocumentCounters:
    section_id: int = 0
    speech_id: int = 0
    paragraph_id: int = 0
    speaker_id: int = 0
    speech_order: int = 1
    paragraph_order: int = 1


def process_ordered_blocks(
    ctx: InferenceContext,
    ordered_blocks: List,
    section_Tblocks: List,
    speech_Tblocks: List,
    speaker_Tblocks: List,
    image: np.ndarray,
    year: int,
    part: int,
    part_page: int,
    date,
    counters: DocumentCounters,
    accum: Dict[str, List[dict]],
) -> None:
    """Per-block extraction + row creation logic."""
    for block in ordered_blocks:
        # SECTION
        if block in section_Tblocks:
            section_name = ocr_block_image(ctx, block, image)

            section_id_text = "_".join(map(str, [year, part, counters.section_id]))
            new_row = {
                "year": year,
                "part": part,
                "part_page": part_page,
                "date": date,
                "volume_page": "",
                "section_name": section_name,
                "section_id": section_id_text,
            }
            accum["sections_new"].append(new_row)

            counters.section_id += 1
            counters.speaker_id = 0
            counters.speech_order = 1

        # SPEECH (paragraph text)
        elif block in speech_Tblocks:
            paragraph_text = ocr_block_image(ctx, block, image)

            speech_id_text = "_".join(map(str, [year, part, counters.speech_id]))
            paragraph_id_text = "_".join(map(str, [year, part, counters.paragraph_id]))
            new_row = {
                "speech_id": speech_id_text,
                "paragraph_text": paragraph_text,
                "paragraph_order": counters.paragraph_order,
                "paragraph_id": paragraph_id_text,
            }
            accum["paragraphs_new"].append(new_row)

            counters.paragraph_id += 1
            counters.paragraph_order += 1

        # SPEAKER (and create speech row)
        elif block in speaker_Tblocks:
            speaker_name = ocr_block_image(ctx, block, image)

            speaker_id_text = ""
            if speaker_name in ctx.state.speaker_vals:
                try:
                    speaker_id_text = (
                        ctx.state.speakers_df[ctx.state.speakers_df["speaker_name"] == speaker_name][
                            "speaker_id"
                        ].iloc[0]
                    )
                except IndexError:
                    speaker_id_text = next(
                        (item["speaker_id"] for item in accum["speakers_new"] if item["speaker_name"] == speaker_name),
                        None,
                    )
            else:
                counters.speaker_id = max(counters.speaker_id, len(ctx.state.speaker_vals)) + 1
                speaker_id_text = "_".join(map(str, [year, part, counters.speaker_id]))

                accum["speakers_new"].append({"speaker_id": speaker_id_text, "speaker_name": speaker_name})
                ctx.state.speaker_vals = np.append(ctx.state.speaker_vals, speaker_name)

            # add new speech data (kept identical, including section_id usage)
            section_id_text = "_".join(map(str, [year, part, counters.section_id]))
            speech_id_text = "_".join(map(str, [year, part, counters.speech_id]))

            accum["speeches_new"].append(
                {
                    "section_id": section_id_text,
                    "speech_order": counters.speech_order,
                    "speaker_name": speaker_name,
                    "speaker_id": speaker_id_text,
                    "speech_id": speech_id_text,
                }
            )

            counters.speech_id += 1
            counters.speech_order += 1
            counters.paragraph_order = 1

        else:
            print("Block not categorized")


def process_document(ctx: InferenceContext, chunk_file: str, doc_index: int, doc_row: pd.Series) -> None:
    """Process a single document row (row['complete']==0), updating ctx.state in-place."""
    cfg = ctx.cfg

    file_path = os.path.join(cfg.image_dir, doc_row["title"])
    print(file_path)
    year, part = doc_to_yearpart(doc_row["title"])

    # set up for new document
    paragraphs_df = new_paragraph_df()

    # local counters and accumulators
    counters = DocumentCounters()
    accum = {
        "sections_new": [],
        "paragraphs_new": [],
        "speakers_new": [],
        "speeches_new": [],
    }

    # convert PDF pages to images in chunks
    chunk_num = 0
    for chunk in convert_pdf_in_chunks(file_path, chunk_size=cfg.chunk_size):
        print("--- Chunk loaded to PNG")

        for page_id, image in enumerate(tqdm(chunk)):
            layout = ctx.engines.model.detect(image)

            # skip logic
            if not should_process_page(layout):
                continue

            height, width = image.shape[:2]
            layout = lp.Layout([b.set(id=idx) for idx, b in enumerate(layout)])

            parts = split_layout_by_type(layout)
            titles = parts["titles"]
            sections = parts["sections"]
            speeches = parts["speeches"]
            speakers = parts["speakers"]

            part_page = chunk_num * cfg.chunk_size + page_id + 1
            date = titles[0] if titles else ""

            # split vertically by title block (exact loop structure)
            for title_id, title in enumerate(titles if titles else [""]):
                (
                    section_Tblocks,
                    speech_Tblocks,
                    speaker_Tblocks,
                    all_Tblocks,
                    sections,
                    speeches,
                    speakers,
                ) = split_blocks_within_title(
                    titles=titles,
                    sections=sections,
                    speeches=speeches,
                    speakers=speakers,
                    title_id=title_id,
                    title=title,
                    page_height=height,
                )

                all_Tblocks = order_blocks_by_columns(all_Tblocks, image=image, year=year)
                ordered_blocks = order_blocks_with_speaker_fix(
                    all_Tblocks=all_Tblocks,
                    section_Tblocks=section_Tblocks,
                    speech_Tblocks=speech_Tblocks,
                    speaker_Tblocks=speaker_Tblocks,
                )

                process_ordered_blocks(
                    ctx=ctx,
                    ordered_blocks=ordered_blocks,
                    section_Tblocks=section_Tblocks,
                    speech_Tblocks=speech_Tblocks,
                    speaker_Tblocks=speaker_Tblocks,
                    image=image,
                    year=year,
                    part=part,
                    part_page=part_page,
                    date=date,
                    counters=counters,
                    accum=accum,
                )

        chunk_num += 1

    # build new dfs and merge into global dfs
    sections_new_df = pd.DataFrame(accum["sections_new"])
    paragraphs_new_df = pd.DataFrame(accum["paragraphs_new"])
    speakers_new_df = pd.DataFrame(accum["speakers_new"])
    speeches_new_df = pd.DataFrame(accum["speeches_new"])

    ctx.state.sections_df = pd.concat([ctx.state.sections_df, sections_new_df], ignore_index=True)
    paragraphs_df = pd.concat([paragraphs_df, paragraphs_new_df], ignore_index=True)
    ctx.state.speakers_df = pd.concat([ctx.state.speakers_df, speakers_new_df], ignore_index=True)
    ctx.state.speeches_df = pd.concat([ctx.state.speeches_df, speeches_new_df], ignore_index=True)

    # update docs_df and docs master (same as original)
    docsdf_path = cfg.inference_dir + "/docs.csv"
    docs_master = pd.read_csv(docsdf_path)
    docs_master.loc[docs_master["title"] == doc_row["title"], "complete"] = 1
    ctx.state.docs_df.loc[doc_index, "complete"] = 1

    # save outputs (same filenames/locations as original)
    mark_doc_complete_and_save_master(cfg, chunk_file, ctx.state.docs_df, docs_master)
    save_per_doc_outputs(
        cfg=cfg,
        year=year,
        part=part,
        paragraphs_df=paragraphs_df,
        sections_df=ctx.state.sections_df,
        speeches_df=ctx.state.speeches_df,
        speakers_df=ctx.state.speakers_df,
    )


# ----------------------------
# Main entrypoint
# ----------------------------

def build_context(cfg: Optional[InferenceConfig], chunk_file: str) -> InferenceContext:
    if cfg is None:
        cfg = InferenceConfig()

    setup_tesseract()

    engines = InferenceEngines(
        model=build_layout_model(cfg.model_config_path, cfg.model_weights_path, cfg.model_score_thresh),
        ocr_agent=build_ocr_agent(languages="eng"),
    )

    state = load_progress_dataframes(chunk_file, cfg)
    return InferenceContext(cfg=cfg, engines=engines, state=state)


def main(chunk_file: str) -> None:
    ctx = build_context(cfg=None, chunk_file=chunk_file)

    # iterate through incomplete files
    for index, row in ctx.state.docs_df.iterrows():
        if row["complete"] == 0:
            process_document(ctx, chunk_file=chunk_file, doc_index=index, doc_row=row)


if __name__ == "__main__":
    chunk_file = sys.argv[1]
    main(chunk_file)
