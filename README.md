# VidScout v0

VidScout is an AI-powered semantic video search prototype.

This first vertical slice:

1. samples frames from a video,
2. converts each frame into a CLIP image embedding,
3. converts a natural-language query into a CLIP text embedding,
4. ranks frames by cosine similarity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Index a video

Place a short test video somewhere accessible, then run:

```bash
python index_video.py /path/to/video.mp4
```

For denser sampling:

```bash
python index_video.py /path/to/video.mp4 --interval 2
```

## Search

```bash
python search_video.py "a person standing outside"
```

Try several visually concrete queries.

## Current limitation

v0 uses fixed-interval frame sampling. The next milestone replaces this with
shot-boundary detection and representative frames per shot, then adds a
Streamlit interface.
