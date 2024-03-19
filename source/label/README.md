### Manual layout labeling
To contribute more annotations for further model fine-tuning, follow these steps:
1. Download Label Studio and create a new project for labeling [here](https://labelstud.io/guide/quick_start). When initializing the project, use the config `datastore/scrape/cr-label.label_config.xml`
2. Select a sample of PDFs and convert them to PNG for labeling using `source/label/select_for_labeling.py` 
3. Label images in Label Studio. Keep bounding boxes as small as possible while covering all relevant text.
4. Once done labeling, export the output. 
5. Use Python package `label-studio-converter` to convert the output into COCO format. Name the file `output.manual.batch3.json` and move it under `datastore/scrape/cr-label`
