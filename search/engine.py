import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from search.utils import get_memory_usage


class Document(BaseModel):
    pageid: str
    title: str
    sentences: List[str]


class Sentence(BaseModel):
    text: str
    doc_index: int


class Result(BaseModel):
    doc: Document
    sentence: Sentence
    score: float


class DataChunk:
    def __init__(
        self,
        doc: Path,
        title_embeddings: Path,
        text_embeddings: Path,
    ):
        documents = []

        with Path(doc).open('br') as doc_reader:
            documents = json.load(doc_reader)
        self.documents = [Document(**d) for d in documents]
        del documents

        with Path(title_embeddings).open('br') as reader:
            t_embeddings = np.load(reader)
        t_embeddings /= np.linalg.norm(
            t_embeddings, axis=-1, keepdims=True
        )

        with Path(text_embeddings).open('br') as reader:
            self.s_embeddings = np.load(reader)
        self.s_embeddings /= np.linalg.norm(
            self.s_embeddings, axis=-1, keepdims=True
        )

        raise NotImplementedError("Populate sentences list!")

       
        self.sentences = []
        for doc_index, doc in enumerate(self.documents):
            for text in doc.sentences:
                raise NotImplementedError("Create sentence list!")
                # Add title embeddings to embeddings
                title = doc['title']
                self.embeddings
                # Add a Sentence to list!
                self.sentences.append(
                )

    def search(self, embedding: np.ndarray, limit: int) -> List[Result]:
        raise NotImplementedError("Scoring")
        # Normalize embedding
        # Compute dot product between embedding and self.embeddings
        # Get indexes of the highest scoring sentences.
        indexes = ...
        return [
            Result(
                doc=self.documents[self.sentences[i].doc_index],
                sentence=self.sentences[i],
                score=scores[i],
            )
            for i in indexes[:limit]
            if not self.documents[self.sentences[i].doc_index].title.startswith(
                (
                    "Usuario:",
                    "Usuaria:",
                    "Usuario discusión:",
                    "Usuaria discusión:",
                )
            )
        ]


class Engine:
    def __init__(self, data_dir: Path, limit: Optional[int] = None):

        doc_paths = sorted(data_dir.glob("doc_*.json"))

        if not doc_paths:
            raise ValueError(data_dir)

        print(datetime.now(), "loading data chunks...")

        self.chunks = []
        for nr, doc_path in enumerate(doc_paths):

            if limit is not None and nr >= limit:
                break

            # fmt: off
            index = doc_path.stem[doc_path.stem.find("_") + 1:]
            # fmt: on
            print(
                f"{datetime.now()} {index} mem usage: {get_memory_usage()} MB"
            )
            self.chunks.append(
                DataChunk(
                    doc=doc_path,
                    title_embeddings=doc_path.with_name(
                        f"title_embedding_{index}.npy"
                    ),
                    text_embeddings=doc_path.with_name(
                        f"sentence_embedding_{index}.npy"
                    ),
                )
            )

        print(datetime.now(), "done.")

    def search(self, embedding: np.ndarray, limit: int) -> List[Result]:
        results = []
        for chunk in self.chunks:
            results.extend(chunk.search(embedding, limit))
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:limit]
