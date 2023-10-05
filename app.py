# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main app."""
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

import chromadb
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import VertexAI
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import utils

_AVAILABLE_MODELS = ["text-bison@001", "text-bison", "text-bison-32k"]

_CHROMA_METADATA = {"hnsw:space": "cosine"}


@dataclass
class CollectionDocument:
    """A Document from a ChromaDB Collection.

    Attributes:
        doc_id: A unique identifier of the Document.
        content: Document content.
    """

    doc_id: str
    content: str


def load_dataframe(csv_data) -> pd.DataFrame:
    """Loads a Pandas DataFrame from a given file object."""
    return pd.read_csv(csv_data)


def preprocess_dataframe(
    dataframe: pd.DataFrame, remove_html: bool = False
) -> pd.DataFrame:
    """Applies provided modifications to the DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        remove_html (bool, optional): If html should be removed.
            Defaults to False.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df_preprocessed = dataframe
    if remove_html:
        df_preprocessed = dataframe.applymap(utils.remove_html_tags)
    return df_preprocessed


def render_dataframe(dataframe: pd.DataFrame, title: str) -> None:
    """Renders the given DataFrame in a streamlit expander."""
    with st.expander(title):
        st.dataframe(dataframe)


def fetch_response(
    google_cloud_project_id: str,
    langchain_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    features: list[dict[str, Any]],
    llm_model: str,
    llm_temperature: float,
) -> list[dict[str, str]]:
    """Fetches generated text from Google Cloud Vertex AI.

    Applies Langchain's LLMChain on the provided list of inputs to generate
    text descriptions.

    Args:
        google_cloud_project_id (str): Goole cloud project to use for text
            generation.
        langchain_prompt_template (FewShotPromptTemplate): _description_
        features (list[dict[str, Any]]): _description_
        llm_model (str): Vertex AI model to use for text generation.
        llm_temperature (float): LLM setting to control how imaginative the
            model can be.

    Returns:
        list[dict[str, str]]: A list of generated texts.
    """
    llm = VertexAI(
        project=google_cloud_project_id,
        location="us-central1",
        model_name=llm_model,
        temperature=llm_temperature,
        max_output_tokens=1024,
        top_p=0.8,
        top_k=40,
    )

    chain = LLMChain(
        prompt=langchain_prompt_template,
        llm=llm,
        output_key="output_description",
        verbose=True,
    )
    return chain.apply(features)


def get_context_documents(file_list: Iterable[Document]) -> list[CollectionDocument]:
    """Prepares the given list of files for ingestion in Chroma.

    Args:
        file_list (Iterable[Document]): List of files to generate
            CollectionDocument for.

    Returns:
        list[CollectionDocument]: List of CollectionDocument that can be
            ingested in Chroma.
    """
    result = []
    for file in file_list:
        md5_hash = hashlib.md5(file.page_content.encode()).hexdigest()
        result.append(CollectionDocument(doc_id=md5_hash, content=file.page_content))
    return result


def clear_chroma_collection(collection: chromadb.Collection) -> chromadb.Collection:
    """Cleans a given Chroma Collection."""
    if collection.count() > 0:
        collection.delete(collection.get(include=[])["ids"])
    return collection


@st.cache_data
def convert_df(dataframe: pd.DataFrame) -> bytes:
    """Creates a utf-8 encoded CSV from a given DataFrame."""
    return dataframe.to_csv(index=False).encode("utf-8")


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

    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, str]]:
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
            sublist = ordered_examples[self.k * -1 :]
            sublist.reverse()  # Order list by descending n-gram overlap.
            return sublist

        return ordered_examples[: self.k]


chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# UI Implementation.
st.set_page_config(page_title="Description Genius", page_icon=":owl:", layout="wide")
st.title("Description Genius 🦉")
st.subheader("Data Upload", divider="rainbow")
input_data = st.file_uploader("Upload your CSV table", type=["csv"])
prompt_features_list = []
prompt_features = []  # A list of dictionaries needed for Langchain prompting.
if input_data:
    input_df = load_dataframe(input_data)
    remove_html_tags = st.checkbox("Remove html tags", value=True)
    processed_df = preprocess_dataframe(input_df, remove_html_tags)
    render_dataframe(processed_df, "Review input data")
    with st.expander("Advanced options"):
        input_columns = input_df.columns.to_list()
        columns_for_prompt = st.multiselect(
            "Columns to use in prompt",
            input_columns,
            input_columns,
            placeholder="Select columns...",
        )
        remove_empty_values = st.checkbox("Ignore empty or NaN values", value=True)
    prompt_df = processed_df[columns_for_prompt].copy()
    prompt_features_str = prompt_df.astype(str).apply(
        utils.row_to_custom_str, args=(remove_empty_values,), axis=1
    )
    prompt_features_list = prompt_features_str.to_list()
    prompt_features = [{"input_features": val} for val in prompt_features_list]
st.subheader("Model Configuration", divider="rainbow")
llm_settings_container = st.container()
llm_settings_col1, llm_settings_col2 = llm_settings_container.columns(2)

gcp_id = llm_settings_col1.text_input(
    "Google Cloud Project Id", placeholder="cloud-project-id"
)
llm_model_name = (
    llm_settings_col2.selectbox("LLM model", (model for model in _AVAILABLE_MODELS))
    or ""
)
temperature = llm_settings_col2.slider(
    label="Temperature", min_value=0.0, max_value=1.0, value=0.20
)
st.subheader("Prompt", divider="rainbow")
prompt_llm_role = st.text_input(
    "Provide a role to the LLM",
    disabled=not (gcp_id, input_data),
    placeholder="You are an expert digital marketer. Your job is to write creative ads based on the provided data.",
)
prompt_llm_guidelines = st.text_area(
    "Provide any guidelines that the LLM should consider",
    height=150,
    disabled=not (gcp_id, input_data),
    placeholder="###GUIDELINES###\n- Always be truthful and present factual information only.\n- Only use the provided features for generating the text.\n- Use an informal tone.\n- ...",
)

examples_df = pd.DataFrame(
    [
        {
            "input": "feature 1: value 1, feature 2: value 2, ..., feature n",
            "output": "[HEADLINE] Lorem ipsum dolor sit amet\n[PARAGRAPH 1] Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        },
        {
            "input": "feature 1: value 1, feature 2: value 2, ..., feature n",
            "output": "[HEADLINE] Lorem ipsum dolor sit amet\n[PARAGRAPH 1] Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        },
    ]
)
prompt_action = st.text_input(
    "Generation prompt (can be edited)",
    value="Generate text descriptions based on the given ###FEATURES### and ###GUIDELINES###.",
    disabled=not (gcp_id, input_data),
)
st.subheader("Few-Shot examples")
edited_df = st.data_editor(
    examples_df, num_rows="dynamic", use_container_width=True, hide_index=True
)
example_selection_criteria = st.radio(
    label="Example selection similarity",
    options=["min", "max"],
    captions=[
        "Select the example which is least similar to our input (prevents over-fitting)",
        "Select the example which is most similar to our input (prevents hallucination)",
    ],
)

prompt_additional_context = ""
additional_context = ""

_PROMPT_SUFFIX = """
Input features: {input_features}
Output description:
"""

with st.expander("Provide additional context (optional)"):
    context_docs = st.file_uploader(
        "Upload context documents", type=["txt"], accept_multiple_files=True
    )
    if context_docs:
        docs = [
            Document(page_content=context_doc.read().decode("utf-8"))
            for context_doc in context_docs
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=400
        )
        chunks = text_splitter.split_documents(docs)
        context_chunks = get_context_documents(chunks)
        context_docs_db = chroma_client.get_or_create_collection(name="context_docs_db")
        context_docs_db = clear_chroma_collection(context_docs_db)
        ids = [context_chunk.doc_id for context_chunk in context_chunks]
        content = [context_chunk.content for context_chunk in context_chunks]
        context_docs_db.upsert(ids=ids, documents=content)
        langchain_db = Chroma(
            client=chroma_client,
            collection_name="context_docs_db",
            embedding_function=embedding_function,
        )
        context_list = []
        for pf in prompt_features_list:
            context_docs_list = [
                pf.page_content
                for pf in langchain_db.similarity_search(
                    pf,
                    embedding_function=embedding_function,
                    k=2,
                )
            ]
            context_list.append(". ".join(context_docs_list))
        prompt_features = [
            {"input_features": f, "additional_context": c}
            for f, c in zip(prompt_features_list, context_list)
        ]
        prompt_additional_context = "\n###START - ADDITIONAL CONTEXT###:\n {additional_context}\n###END - ADDITIONAL CONTEXT###:\n"

prompt_input_variables = ["input_features"]
if prompt_additional_context:
    prompt_input_variables.append("additional_context")

prompt_prefix = f"{prompt_llm_role}\n{prompt_llm_guidelines}\n{prompt_additional_context}\n{prompt_action}"

# If examples have been provided, use a Few Shot Prompt.
if len(edited_df.index) > 0:
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input features: {input}\nOutput description: {output}",
    )
    example_selector = CustomSimilarityExampleSelector(
        examples=edited_df.to_dict("records"),
        ex_prompt=example_prompt,
        selection_criteria=example_selection_criteria,
        k=1,  # TODO: Make this a user-provided value.
    )

    prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prompt_prefix,
        suffix=_PROMPT_SUFFIX,
        input_variables=prompt_input_variables,
    )
# If no examples provided, use a standard prompt instead.
else:
    prompt_template = PromptTemplate.from_template(prompt_prefix)

generate_button = st.button(
    "Generate", disabled=not (input_data, gcp_id, prompt_template), type="primary"
)
if generate_button:
    with st.spinner("Running request..."):
        results = fetch_response(
            gcp_id, prompt_template, prompt_features, llm_model_name, temperature
        )
        if results:
            results_df = st.data_editor(
                pd.DataFrame.from_records(results),
                use_container_width=True,
                hide_index=True,
            )
            csv = convert_df(results_df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="text_descriptions.csv",
                mime="text/csv",
            )
        else:
            st.warning("No results were returned.", icon="⚠️")
