import collections
import os
from pathlib import Path
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer

from search.engine import Document, Engine, Result
from search.model import load_minilm_model
from search.utils import get_memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DATA_DIR = os.environ.get("DATA_DIR", "data/people_pm_minilm")

st.set_page_config(page_title="Search Engine", layout="wide")

st.markdown(
    """
<style>
.big-font {
    font-size:20px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache(allow_output_mutation=True)
def load_engine() -> Engine:
    engine = Engine(
        data_dir=Path(_DATA_DIR),
    )
    return engine


@st.cache(allow_output_mutation=True)
def load_model() -> SentenceTransformer:
    return load_minilm_model()


engine = load_engine()

model = load_model()

query = st.text_input(
    label='Query',
    value=''
)

limit = st.slider(
    label='Slider',
    min_value=0,
    max_value=100,
    step=1
)

with st.spinner("Querying index ..."):
    embedding = model.encode([query])[0]

    results = engine.search(
        embedding=embedding,
        limit=limit
    )

# Show the results.
# You can use st.markdown to render markdown.
# e.g. st.markdown("**text**") will add text in bold font.

def get_wikipedia_page(pageid: str) -> str:
    return f'https://es.wikipedia.org/?curid={pageid}'

def display(result: Result) -> None:
    st.markdown(f"### {result.doc.title} ([link]({get_wikipedia_page(result.doc.pageid)}))")
    st.write(result.sentence.text)
    st.markdown(f"(**score**: {result.score})")
    

[
    display(result)
    for result in results
]

st.markdown(f"**Mem Usage**: {get_memory_usage()}MB")
