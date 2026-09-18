# OmniBrain – PDF Ingestion Module

## Overview

OmniBrain is an Agentic Multi-Modal RAG system designed to work with complex documents containing text, images, charts, and other multimodal information.

This module implements the **PDF ingestion pipeline** for Week 1 of the OmniBrain project.

The module performs three main tasks:

1. PDF text extraction
2. Text chunking
3. Image extraction

The extracted data is prepared for the next stages of the OmniBrain pipeline, including text/image embedding and Qdrant vector database integration.

---

## Project Structure

```text
OmniBrain/
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── text_chunker.py
│   ├── image_extractor.py
│   └── pipeline.py
│
├── data/
│   ├── input/
│   │   └── sample.pdf
│   │
│   └── output/
│       ├── text/
│       ├── chunks/
│       │   └── chunks.json
│       └── images/
│           └── images.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

### 1. PDF Text Extraction

The PDF parser extracts text from every page of the input PDF.

The extracted text is stored page-wise in the `data/output/text/` directory.

Example:

```text
page_001.txt
page_002.txt
page_003.txt
```

Each text file corresponds to one PDF page.

---

### 2. Text Chunking

The extracted text is divided into smaller overlapping chunks.

The current configuration is:

```text
Chunk size: 1000 characters
Overlap: 200 characters
```

Each chunk maintains its PDF page information.

Example:

```json
{
    "chunk_id": 1,
    "page": 5,
    "text": "Extracted text from the PDF..."
}
```

All chunks are stored in:

```text
data/output/chunks/chunks.json
```

---

### 3. Image Extraction

Images embedded in the PDF are extracted and saved separately.

Images are stored in:

```text
data/output/images/
```

Example:

```text
page_001_img_001.png
page_003_img_001.jpeg
```

Metadata about extracted images is stored in:

```text
data/output/images/images.json
```

---

## Technologies Used

* Python
* PyMuPDF
* JSON
* VS Code
* Git/GitHub

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd OmniBrain
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

---

### 3. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Input PDF

Place the PDF that needs to be processed inside:

```text
data/input/
```

For example:

```text
data/input/sample.pdf
```

The PDF should be a valid, non-empty PDF file.

---

## Running the Pipeline

Open the terminal inside the `ingestion` directory:

```bash
cd ingestion
```

Run:

```bash
python pipeline.py
```

---

## Expected Output

After successful execution, the terminal should display:

```text
Starting OmniBrain PDF ingestion...

[1/3] Extracting text...
Extracted text from 6 pages.

[2/3] Creating text chunks...
Created 18 chunks.

[3/3] Extracting images...
Extracted 6 images.

--------------------------------
OmniBrain ingestion completed!
--------------------------------
```

The exact number of pages, chunks, and images depends on the input PDF.

---

## Output Structure

After running the pipeline:

```text
data/
└── output/
    ├── text/
    │   ├── page_001.txt
    │   ├── page_002.txt
    │   └── ...
    │
    ├── chunks/
    │   └── chunks.json
    │
    └── images/
        ├── page_001_img_001.png
        ├── page_002_img_001.png
        └── images.json
```

---

## Pipeline

```text
                Input PDF
                    │
                    ▼
             PDF Text Parser
                    │
                    ▼
              Text Extraction
                    │
                    ▼
              Text Chunking
                    │
                    ▼
               chunks.json


                Input PDF
                    │
                    ▼
             Image Extraction
                    │
                    ▼
             Extracted Images
                    │
                    ▼
              images.json
```

---

## Current Test Result

The ingestion pipeline was successfully tested with a PDF containing:

* 6 pages
* 18 text chunks
* 6 extracted images

The pipeline completed successfully without errors.

---

## Integration with OmniBrain

The output of this module will be used by the next stages of the OmniBrain system.

```text
PDF
 │
 ▼
PDF Ingestion
 │
 ├── Text
 │    └── Text Chunks
 │
 └── Images
      │
      ▼
 Embedding Generation
      │
      ▼
 Qdrant Vector Database
```

The extracted chunks and images are therefore prepared as input for the embedding and Qdrant integration modules.

---

## Git Workflow

Create a feature branch for the ingestion module:

```bash
git checkout -b feature/pdf-ingestion
```

Add the changes:

```bash
git add ingestion requirements.txt README.md
```

Commit:

```bash
git commit -m "Add PDF ingestion pipeline"
```

Push the branch:

```bash
git push -u origin feature/pdf-ingestion
```

The branch can then be reviewed and merged into the main project branch.

---

## Future Improvements

Possible improvements to this module include:

* Support for scanned PDFs using OCR
* Better semantic text chunking
* Table extraction
* Improved image metadata
* Support for multiple PDF files
* Metadata linking between text chunks and images
* Integration with the Qdrant embedding pipeline

---
