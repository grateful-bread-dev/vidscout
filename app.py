import json
from pathlib import Path

import numpy as np
import streamlit as st

from src.embeddings import ClipSearchEngine


DEFAULT_RELEVANCE_THRESHOLD = 0.2336
DEFAULT_INDEX_DIR = "data/index"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="VidScout",
    page_icon="🎬",
    layout="wide",
)


# ---------------------------------------------------------
# Data/model loading
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_index(index_dir: str):
    """
    Load the saved VidScout embedding matrix and metadata.
    """
    index_path = Path(index_dir)

    embeddings_path = index_path / "embeddings.npy"
    metadata_path = index_path / "metadata.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Could not find {embeddings_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}"
        )

    embeddings = np.load(embeddings_path)

    with open(
        metadata_path,
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return embeddings, metadata


@st.cache_resource(
    show_spinner="Loading CLIP model..."
)
def load_search_engine(
    model_name: str,
) -> ClipSearchEngine:
    """
    Load CLIP once and reuse it across Streamlit reruns.
    """
    return ClipSearchEngine(
        model_name=model_name
    )


# ---------------------------------------------------------
# Search logic
# ---------------------------------------------------------

def aggregate_best_frame_per_shot(
    scores: np.ndarray,
    frames: list[dict],
) -> list[dict]:
    """
    Keep the highest-scoring representative frame
    for each shot.
    """

    best_by_shot = {}

    for frame_index, score in enumerate(scores):
        frame = frames[frame_index]

        shot_index = int(
            frame["shot_index"]
        )

        current_best = best_by_shot.get(
            shot_index
        )

        if (
            current_best is None
            or score > current_best["score"]
        ):
            best_by_shot[shot_index] = {
                "shot_index": shot_index,
                "score": float(score),
                "frame": frame,
            }

    return sorted(
        best_by_shot.values(),
        key=lambda result: result["score"],
        reverse=True,
    )


def search_video(
    query: str,
    embeddings: np.ndarray,
    metadata: dict,
    engine: ClipSearchEngine,
    threshold: float,
    top_k: int,
):
    """
    Perform semantic search and return relevant
    shot-level results.
    """

    query_embedding = engine.embed_text(
        query
    )

    # All embeddings are L2-normalized,
    # so dot product == cosine similarity.
    scores = embeddings @ query_embedding

    ranked_shots = (
        aggregate_best_frame_per_shot(
            scores,
            metadata["frames"],
        )
    )

    relevant_results = [
        result
        for result in ranked_shots
        if result["score"] >= threshold
    ]

    return (
        relevant_results[:top_k],
        ranked_shots,
    )


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

st.title("VidScout")

st.subheader(
    "Semantic video search powered by "
    "computer vision and multimodal embeddings"
)

st.write(
    "Search indexed footage using natural-language "
    "descriptions. VidScout segments video into shots, "
    "samples representative frames, and ranks shots "
    "using CLIP image-text similarity."
)


# ---------------------------------------------------------
# Load existing index
# ---------------------------------------------------------

try:
    embeddings, metadata = load_index(
        DEFAULT_INDEX_DIR
    )
except FileNotFoundError as error:
    st.error(str(error))

    st.info(
        "Create a VidScout index with "
        "`python index_video.py <video>` "
        "before launching the app."
    )

    st.stop()


engine = load_search_engine(
    metadata["model"]
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

video_metadata = metadata.get(
    "video",
    {}
)

st.sidebar.header("VidScout")

st.sidebar.metric(
    "Detected shots",
    video_metadata.get(
        "shot_count",
        len(metadata.get("shots", [])),
    ),
)

st.sidebar.metric(
    "Representative frames",
    video_metadata.get(
        "representative_frame_count",
        len(metadata.get("frames", [])),
    ),
)

threshold = st.sidebar.slider(
    "Relevance threshold",
    min_value=0.10,
    max_value=0.40,
    value=float(
        DEFAULT_RELEVANCE_THRESHOLD
    ),
    step=0.005,
)

top_k = st.sidebar.slider(
    "Maximum results",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

st.sidebar.caption(
    "Default relevance threshold: "
    "0.2336, calibrated using VidScout's "
    "initial positive/negative evaluation set."
)


# ---------------------------------------------------------
# Search form
# ---------------------------------------------------------

st.divider()

with st.form("semantic_search"):
    query = st.text_input(
        "Search the footage",
        placeholder=(
            'Try: "a man kicking a soccer ball"'
        ),
    )

    submitted = st.form_submit_button(
        "Search"
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

if submitted:
    query = query.strip()

    if not query:
        st.warning(
            "Enter a visual description to search."
        )

    else:
        with st.spinner(
            "Searching video..."
        ):
            results, ranked_shots = (
                search_video(
                    query=query,
                    embeddings=embeddings,
                    metadata=metadata,
                    engine=engine,
                    threshold=threshold,
                    top_k=top_k,
                )
            )

        st.subheader(
            f'Results for "{query}"'
        )

        if not results:
            best_score = (
                ranked_shots[0]["score"]
                if ranked_shots
                else 0.0
            )

            st.warning(
                "No strong matches found."
            )

            st.write(
                f"Best available similarity: "
                f"**{best_score:.4f}**"
            )

            st.write(
                f"Current relevance threshold: "
                f"**{threshold:.4f}**"
            )

        else:
            st.caption(
                f"{len(results)} relevant "
                f"{'result' if len(results) == 1 else 'results'} found"
            )

            columns = st.columns(3)

            for index, result in enumerate(
                results
            ):
                frame = result["frame"]
                shot_number = (
                    result["shot_index"] + 1
                )

                column = columns[
                    index % 3
                ]

                with column:
                    st.image(
                        frame["path"],
                        width="stretch",
                    )

                    st.markdown(
                        f"### Shot {shot_number:03d}"
                    )

                    st.write(
                        f"**In:** "
                        f"{frame['shot_start_timestamp']}"
                    )

                    st.write(
                        f"**Out:** "
                        f"{frame['shot_end_timestamp']}"
                    )

                    st.write(
                        f"**Best frame:** "
                        f"{frame['timestamp']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['score']:.4f}"
                    )