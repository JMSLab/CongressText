import os
import random
import PyPDF2
from pdf2image import convert_from_path

def select_random_pdfs(folder_path, num_pdfs=100):
    """Select random PDFs from a folder, weighted by file size."""
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    weights = [os.path.getsize(os.path.join(folder_path, f)) for f in all_files]
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    selected_files = random.choices(all_files, k=num_pdfs, weights=normalized_weights)
    return selected_files

def extract_random_page(source_folder, dest_folder, filename):
    """Extracts a random page from the PDF and saves it to the destination folder."""
    with open(os.path.join(source_folder, filename), 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        page_num = random.randint(0, total_pages-1)
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[page_num])

        output_filename = f"extracted_{page_num}_{filename}"
        with open(os.path.join(dest_folder, output_filename), 'wb') as output_pdf_file:
            writer.write(output_pdf_file)

def pdf_to_png(folder_path):
	"""Converts PDF to png in folder"""
	all_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

	for pdf in all_files:
		images = convert_from_path(os.path.join(folder_path,pdf))
		base_name, _ = os.path.splitext(pdf)
		output_filename = f"{base_name}.png"
		images[0].save(os.path.join(folder_path, output_filename), 'PNG')


def main(source_folder, dest_folder, num_pdfs=100):
	if not bool(os.listdir(dest_folder)):
		selected_pdfs = select_random_pdfs(source_folder, num_pdfs)
		for pdf in selected_pdfs:
			extract_random_page(source_folder, dest_folder, pdf)

	pdf_to_png(dest_folder)

if __name__ == "__main__":
    SOURCE_FOLDER = "../../../../../holyscratch01/jshapiro_lab/Lab/CongressText/datastore/scrape/cr-bound"
    DEST_FOLDER = "datastore/scrape/cr-label/batch3"
    os.makedirs(DEST_FOLDER, exist_ok=True)  # Ensuring destination directory exists
    main(SOURCE_FOLDER, DEST_FOLDER)
