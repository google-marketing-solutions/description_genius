# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for Retrieval Augmented Generation (RAG)."""

import dataclasses
import hashlib
import os
from typing import Any, Iterable, Optional, Sequence

from langchain_community.example_selectors import NGramOverlapExampleSelector
from langchain_core.documents import Document
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
_CHROMA_METADATA = {"hnsw:space": "cosine"}


class CustomSimilarityExampleSelector(BaseExampleSelector):
  """Langchain ExampleSelector for selecting most or least similar examples."""

  def __init__(
      self,
      examples: list[dict[Any, Any]],
      ex_prompt: PromptTemplate,
      k: int,
      selection_criteria: Optional[str] = "max",
  ):
    self.examples = examples
    self.example_prompt = ex_prompt
    self.selection_criteria = selection_criteria
    self.k = k

  @property
  def example_count(self):
    """Gets the number of examples available."""
    return len(self.examples)

  def add_example(self, example: dict[str, str]) -> None:
    """Add new example to store for a key."""
    self.examples.append(example)

  def select_examples(self, input_variables: dict[str,
                                                  str]) -> list[dict[str, str]]:
    """Select which examples to use based on the inputs."""
    ng_example_selector = NGramOverlapExampleSelector(
        examples=self.examples,
        example_prompt=self.example_prompt,
        threshold=-1,  # Select all examples but reorder them.
    )

    ordered_examples = ng_example_selector.select_examples(input_variables)

    if self.k > self.example_count:
      self.k = self.example_count

    if self.selection_criteria == "min":
      sublist = ordered_examples[self.k * -1:]
      sublist.reverse()  # Order list by descending n-gram overlap.
      return sublist

    return ordered_examples[:self.k]


@dataclasses.dataclass
class CollectionDocument:
  """A Document from a ChromaDB Collection.

  Attributes:
    doc_id: A unique identifier of the Document.
    content: Document content.
  """

  doc_id: str
  content: str


@st.cache_resource
def get_embedding_function():
  """Returns a cached HuggingFace embedding function."""
  from langchain_huggingface import HuggingFaceEmbeddings  # pylint: disable=g-import-not-at-top
  return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_db_client():
  """Returns a cached Chroma instance."""
  import chromadb  # pylint: disable=g-import-not-at-top
  return chromadb.Client()


def get_context_documents(
    file_list: Iterable[Document],
) -> list[CollectionDocument]:
  """Prepares the given list of files for ingestion in Chroma.

  Args:
    file_list (Iterable[Document]): List of files to generate CollectionDocument
      for.

  Returns:
    list[CollectionDocument]: List of CollectionDocument that can be
      ingested in Chroma.
  """
  result = []
  for file in file_list:
    md5_hash = hashlib.md5(file.page_content.encode()).hexdigest()
    result.append(
        CollectionDocument(doc_id=md5_hash, content=file.page_content),
    )
  return result


@st.cache_resource
def get_context_list_from_docs(
    context_docs: list[UploadedFile], prompt_features_list: Sequence[str]
) -> list[str]:
  """Retrieves context from given documents."""
  # Step 1: Build a list of documents.
  docs = [
      Document(page_content=context_doc.read().decode("utf-8"))
      for context_doc in context_docs
  ]
  if not docs:
    return []
  # Step 2: Build a vector database from the provided documents.
  chroma_client = get_chroma_db_client()
  embedding_function = get_embedding_function()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=800, chunk_overlap=400
  )
  chunks = text_splitter.split_documents(docs)
  context_chunks = get_context_documents(chunks)
  context_docs_db = chroma_client.get_or_create_collection(
      name="context_docs_db"
  )
  context_docs_db = clear_chroma_collection(context_docs_db)
  ids = [context_chunk.doc_id for context_chunk in context_chunks]
  content = [context_chunk.content for context_chunk in context_chunks]
  context_docs_db.upsert(ids=ids, documents=content)

  # Step 3: Build a langchain vectorstore with our vector db.
  from langchain_chroma import Chroma  # pylint: disable=g-import-not-at-top
  langchain_db = Chroma(
      client=chroma_client,
      collection_name="context_docs_db",
      embedding_function=embedding_function,
  )
  context_list = []
  # Step 4: Get relevant chunks using similarity search on product features.
  for pf in prompt_features_list:
    context_docs_list = [
        pf.page_content for pf in langchain_db.similarity_search(
            pf,
            k=4,
        )
    ]
    context_list.append(". ".join(context_docs_list))
  return context_list


def clear_chroma_collection(collection):
  """Cleans a given Chroma Collection."""
  if collection.count() > 0:
    collection.delete(collection.get(include=[])["ids"])
  return collection
