# VidScout

**VidScout** is a computer-vision and multimodal AI application for searching video footage with natural-language descriptions.

Instead of relying on filenames, manual metadata, or fixed keywords, VidScout analyzes the visual content of a video, divides it into editorial shots, creates semantic representations of those shots using CLIP, and allows a user to search the footage with queries such as:

```text
a man kicking a soccer ball
```

or:

```text
a woman holding a cup
```

VidScout returns the most relevant matching shots along with representative thumbnails, timestamps, and cosine-similarity scores.

---

## Features

* Custom histogram-based shot-boundary detection
* Duration-aware representative-frame sampling
* CLIP-based multimodal image/text embeddings
* Natural-language semantic video search
* Shot-level result aggregation
* Cosine-similarity ranking
* Calibrated relevance filtering
* "No strong matches found" behavior for unrelated queries
* Small positive/negative retrieval evaluation suite
* Streamlit user interface with thumbnails and shot metadata

---

## How It Works

VidScout processes footage through the following pipeline:

```text
Video
  |
  v
OpenCV Video Decoding
  |
  v
Shot-Boundary Detection
  |
  v
Editorial Shots
  |
  v
Duration-Aware Representative Frames
  |
  v
CLIP Image Encoder
  |
  v
Normalized Image Embeddings
  |
  v
Saved Search Index
```

When the user submits a search:

```text
Natural-Language Query
  |
  v
CLIP Text Encoder
  |
  v
Normalized Text Embedding
  |
  v
Cosine Similarity
  |
  v
Frame-Level Ranking
  |
  v
Shot-Level Aggregation
  |
  v
Relevance Filtering
  |
  v
Matching Shots + Thumbnails
```

CLIP is used as a pretrained vision-language model. VidScout builds the video-processing, shot segmentation, frame-selection, indexing, retrieval, aggregation, evaluation, and user-interface systems around the model.

---

## Shot-Boundary Detection

VidScout includes a custom first-pass hard-cut detector implemented with OpenCV.

For consecutive video frames, the detector:

1. Resizes the frame for more efficient processing.
2. Converts the image from BGR to HSV.
3. Calculates a normalized two-dimensional hue/saturation histogram.
4. Compares consecutive histograms using Bhattacharyya distance.
5. Declares a probable editorial cut when the visual difference exceeds a configurable threshold.

This allows VidScout to reason about video in terms of editorial **shots** rather than treating the entire video as a sequence of arbitrary fixed-interval images.

---

## Representative-Frame Sampling

A single frame may not adequately describe a long shot.

For example, one continuous shot could contain:

```text
soccer ball alone
      |
person enters frame
      |
person kicks ball
      |
ball leaves frame
```

To preserve more of the visual information within longer shots, VidScout adjusts the number of representative frames according to shot duration.

The current sampling strategy is approximately:

```text
Shot <= 4 seconds
    -> 1 representative frame

Shot 4-8 seconds
    -> 2 representative frames

Shot > 8 seconds
    -> multiple frames distributed across the shot
```

Multiple frame embeddings may therefore belong to the same shot.

During search, VidScout keeps the strongest matching representative frame for each shot so that users receive meaningful shot-level results instead of multiple nearly identical frame-level results.

---

## Semantic Search

VidScout uses the pretrained CLIP vision-language model to encode both images and text into a shared embedding space.

A representative video frame becomes an image embedding:

```text
Representative Frame
        |
        v
      CLIP
        |
        v
512-dimensional embedding
```

A user query follows a parallel path:

```text
"a man playing soccer"
        |
        v
      CLIP
        |
        v
512-dimensional embedding
```

The embeddings are L2-normalized and compared using cosine similarity.

Higher similarity indicates greater semantic alignment between a query and a representative video frame.

These similarity scores are **not confidence percentages or accuracy values**. They are retrieval scores used to rank the indexed footage.

---

## Evaluation and Relevance Filtering

VidScout includes a small evaluation script containing both:

* positive queries describing visual content known to exist in the indexed test footage
* negative queries describing content known not to exist

An initial 12-query evaluation consisted of:

```text
6 positive queries
6 negative queries
```

Results on the initial soccer-footage benchmark:

```text
Positive similarity range: 0.2735 - 0.3042
Mean positive similarity:  0.2855

Negative similarity range: 0.1500 - 0.1937
Mean negative similarity:  0.1721

Observed positive/negative separation: 0.0798
Calibrated relevance threshold:        0.2336
Calibration accuracy:                  100.0%
```

The 100% figure refers only to separation of this small initial calibration set. It should **not** be interpreted as 100% general retrieval accuracy.

The calibrated threshold is currently used to suppress weak results:

```text
similarity >= threshold
    -> return as a plausible match

similarity < threshold
    -> suppress result
```

If no indexed shot passes the threshold, VidScout reports:

```text
No strong matches found.
```

The current threshold is therefore an experimentally calibrated development value, not a universal CLIP confidence threshold.

---

# Getting Started

## 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd vidscout
```

---

## 2. Create a Python Virtual Environment

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

VidScout uses packages including:

* PyTorch
* Hugging Face Transformers
* OpenCV
* NumPy
* Pillow
* Streamlit

The CLIP model is downloaded automatically the first time it is required and is subsequently cached locally.

---

## 5. Add Test Footage

Create or use the existing:

```text
sample_media/
```

directory and place a video inside it.

For example:

```text
sample_media/test_video.mp4
```

The `sample_media` directory should remain excluded from Git so that large or copyrighted video files are not committed to the repository.

---

# Optional: Test Shot Detection Independently

Before building the semantic-search index, the custom shot detector can be tested on its own:

```bash
python detect_shots.py sample_media/test_video.mp4
```

Example output:

```text
Cut detected at 00:00:03.086
Cut detected at 00:00:05.756
Cut detected at 00:00:07.716

Shot detection complete.
FPS: 23.976
Detected shots: 4
```

The detector also supports a configurable histogram-distance threshold:

```bash
python detect_shots.py sample_media/test_video.mp4 --threshold 0.45
```

This step is useful for debugging, but it is **not required** before indexing because `index_video.py` runs shot detection automatically.

---

# 6. Index a Video

To make footage searchable, run:

```bash
python index_video.py sample_media/test_video.mp4
```

VidScout will:

1. Detect editorial shots.
2. Determine how many representative frames each shot requires.
3. Extract representative JPEG images.
4. Load CLIP.
5. Generate normalized image embeddings.
6. Save the embedding matrix and metadata.

Example output:

```text
Detecting shots...
Detected 24 shots.

Extracting representative frames...
Extracted 50 representative frames.

Generating CLIP image embeddings...
Loading CLIP on: cpu

VidScout shot-based index created.
Detected shots: 24
Representative frames: 50
Embedding matrix shape: (50, 512)

Metadata: data/index/metadata.json
Embeddings: data/index/embeddings.npy
```

Generated representative images are stored under:

```text
data/frames/
```

The semantic-search index is stored under:

```text
data/index/
```

---

# 7. Search from the Command Line

After indexing, searches can be performed directly from the terminal:

```bash
python search_video.py "a man kicking a soccer ball"
```

Example result:

```text
Search results for: "a man kicking a soccer ball"
======================================================================

Relevance threshold: 0.2336

1. Shot 001
   00:00:00.000 -> 00:00:13.960
   score=0.2735

   Best frame:
   00:00:05.560
   data/frames/shot_000_sample_01.jpg
```

An unrelated search may instead produce:

```text
Search results for: "a volcano erupting"
======================================================================

No strong matches found.
Best available similarity: 0.1771
Relevance threshold: 0.2336
```

The maximum number of returned results can also be configured:

```bash
python search_video.py "a person outdoors" --top-k 3
```

---

# Optional: Run the Evaluation Suite

After indexing the current test footage, run:

```bash
python evaluate.py
```

The evaluation script tests a small collection of positive and negative queries and reports their similarity distributions along with the relevance threshold that best separates them.

This is primarily a development and calibration tool rather than part of the end-user workflow.

---

# 8. Launch the Streamlit Application

Once a video has been indexed, launch the graphical interface with:

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

Streamlit starts a local development server and should automatically open VidScout in the default web browser.

The application normally runs at an address similar to:

```text
http://localhost:8501
```

The interface provides:

* natural-language video search
* representative thumbnails
* shot numbers
* shot in/out timestamps
* best matching frame timestamps
* cosine-similarity scores
* adjustable relevance threshold
* adjustable maximum result count
* no-match behavior for weak queries

The CLIP model is cached by Streamlit after loading so it does not need to be recreated for every interaction.

---

## Stopping the Application

To stop the Streamlit server, return to the terminal where it is running and press:

```text
Ctrl+C
```

---

# Typical Workflow

After the repository has been configured, the normal VidScout workflow is:

```bash
# Activate environment
source .venv/bin/activate

# Index new footage
python index_video.py sample_media/test_video.mp4

# Optional command-line search
python search_video.py "a person standing outside"

# Optional evaluation
python evaluate.py

# Launch graphical application
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

---

# Project Structure

```text
vidscout/
|
|-- app.py
|-- detect_shots.py
|-- evaluate.py
|-- index_video.py
|-- search_video.py
|-- requirements.txt
|-- README.md
|
|-- src/
|   |-- __init__.py
|   |-- embeddings.py
|   |-- shot_detection.py
|   `-- video.py
|
|-- data/
|   |-- frames/
|   `-- index/
|       |-- embeddings.npy
|       `-- metadata.json
|
`-- sample_media/
```

### `app.py`

Streamlit-based graphical interface for semantic video search.

### `detect_shots.py`

Standalone command-line interface for testing the custom shot-boundary detector.

### `evaluate.py`

Runs positive and negative semantic queries to evaluate retrieval behavior and calibrate relevance filtering.

### `index_video.py`

Runs the complete indexing pipeline: shot detection, representative-frame extraction, CLIP inference, and index generation.

### `search_video.py`

Performs command-line natural-language retrieval against an existing VidScout index.

### `src/shot_detection.py`

Implements custom histogram-based shot-boundary detection.

### `src/video.py`

Handles duration-aware representative-frame selection and extraction.

### `src/embeddings.py`

Loads CLIP and generates normalized image and text embeddings.

---

# Technologies and Concepts

VidScout demonstrates practical use of:

* Python
* OpenCV
* PyTorch
* Hugging Face Transformers
* CLIP
* Streamlit
* Computer vision
* Video processing
* Feature extraction
* HSV color histograms
* Bhattacharyya distance
* Temporal video segmentation
* Multimodal deep learning
* Pretrained model inference
* Embeddings
* Vector representations
* L2 normalization
* Cosine similarity
* Semantic search
* Information retrieval
* Relevance ranking
* Threshold calibration
* Model/system evaluation
* ML application architecture

---

# Current Limitations

VidScout is currently a portfolio-scale prototype rather than a production video-search platform.

Current limitations include:

* The shot detector is optimized primarily for hard cuts and may be less reliable on dissolves, fades, flashes, whip pans, or unusually high-motion footage.
* The relevance threshold was calibrated using a small initial evaluation set and requires testing on substantially more diverse footage.
* CLIP performs whole-image semantic representation rather than dedicated object detection, so very small objects may not always dominate retrieval behavior.
* Indexed footage must currently be prepared from the command line before launching the Streamlit application.
* Only visual information is indexed; audio, dialogue, transcripts, captions, and metadata are not currently part of retrieval.
* Source-video timecode and professional NLE interchange are not currently implemented.

---

# Potential Future Work

Possible extensions include:

* Larger retrieval evaluation datasets
* Comparison against dedicated scene-detection libraries
* Improved fade and dissolve detection
* Adaptive relevance calibration
* Drag-and-drop video indexing
* Video preview and seek-to-result functionality
* Persistent multi-video libraries
* Vector-database indexing for large media collections
* Visual similarity search from a selected frame
* Speech transcription and dialogue search
* Automatic visual metadata generation
* Professional source-timecode handling
* NLE or media-asset-management integration

---

# Motivation

VidScout was created as an exploration of how modern computer vision and multimodal AI could improve video and post-production workflows.

Large video projects frequently require editors and other post-production professionals to locate particular visual moments within extensive amounts of footage. Traditional workflows often depend on filenames, manually entered metadata, transcripts, or human memory.

VidScout explores a different approach: making the **visual meaning of the footage itself searchable**.

Rather than training a new foundation model from scratch, the project focuses on a common AI-engineering challenge: selecting an appropriate pretrained model and designing a complete, evaluated software system around it.
