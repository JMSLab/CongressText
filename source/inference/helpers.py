# ----------------------------
# Helper functions for infer_layouts_and_text.py
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

