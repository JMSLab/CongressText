## CongressText

This repository is built from [`JMSLab/Template`](https://github.com/JMSLab/Template/tree/d39df7414aad471b3df214f6684191496b2a90fd) and by default shares its dependencies and prerequisites.

The datastore is [`CongressText`](https://drive.google.com/drive/u/1/folders/0ALDmSBTLfB78Uk9PVA).


### Workflow summary
This repository produces text data containing the speeches contained in the Congressional Record. To accomplish this, we use the following procedure:
1. Scrape the Congress.gov website for PDF scans of the Congressional Record
2. Manually label the layouts for a random sample of PDFs using LabelStudio
3. Fine-tune a mask RCNN model to automatically detect layouts using LayoutParser
4. Detect layouts for all PDFs using this model
5. OCR to obtain text data for all PDFs
6. Clean text data into final speech data

### Manual layout labeling
All steps in this procedure can be automated except for labeling layouts. To contribute more annotations for further model fine-tuning, follow these steps:
1. Download Label Studio and create a new project for labeling [here](https://labelstud.io/guide/quick_start). When initializing the project, use the config `datastore/scrape/cr-label.label_config.xml`
2. Select a sample of PDFs and convert them to PNG for labeling using `source/label/select_for_labeling.py` 
3. Label images in Label Studio. Keep bounding boxes as small as possible while covering all relevant text.
4. Once done labeling, export the output. 
5. Use Python package `label-studio-converter` to convert the output into COCO format. Name the file `output.manual.batch3.json` and move it under `datastore/scrape/cr-label`

At this stage, "Step 3. Fine-tune a mask RCNN model..." can be executed without modification.

The full documentation for our procedure is located [here](https://github.com/JMSLab/CongressText/issues/6).


