# Training

Fine-tunes a RCNN model to detect page layouts in scans of the bound Congressional
Record, using [LayoutParser](https://github.com/Layout-Parser/layout-parser) and
[Detectron2](https://github.com/facebookresearch/detectron2).

## Prerequisites not covered by `requirements.txt`

Two dependencies cannot be installed by `pip install -r source/lib/requirements.txt`
and must be installed separately.

**Detectron2** is not distributed on PyPI. It is built from source against your
CUDA and PyTorch versions, so the correct command depends on the machine. See the
[installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html);
on a cluster node with a GPU this is typically:

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Tesseract** is an OCR engine, not a Python package. `requirements.txt` declares
`pytesseract`, which is only a wrapper around the `tesseract` executable and will
fail at runtime if the executable is absent. Install the engine through your system
package manager (`apt-get install tesseract-ocr`, `brew install tesseract`, or the
[Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)).
