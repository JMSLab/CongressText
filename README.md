## CongressText

This repository is built from [`JMSLab/Template`](https://github.com/JMSLab/Template/tree/d39df7414aad471b3df214f6684191496b2a90fd) and by default shares its dependencies and prerequisites.

The datastore is [`CongressText`](https://drive.google.com/drive/u/1/folders/0ALDmSBTLfB78Uk9PVA).

### Workflow summary
This repository produces text data containing the speeches contained in the Congressional Record. To accomplish this, we use the following procedure:
1. Scrape the Congress.gov website for PDF scans of the Congressional Record
2. [Manually label](./source/label/README.md) the layouts for a random sample of PDFs using LabelStudio
3. Fine-tune a mask RCNN model to automatically detect layouts using LayoutParser
4. Detect layouts for all PDFs using this model
5. OCR to obtain text data for all PDFs
6. Clean text data into final speech data

