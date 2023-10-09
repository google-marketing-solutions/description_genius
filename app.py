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
from typing import Any, Iterable, Optional, Sequence, Union

import chromadb
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain, SequentialChain
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

_AVAILABLE_MODELS = ["text-bison-32k", "text-bison@001", "text-bison"]

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


def render_dataframe(dataframe: pd.DataFrame) -> None:
    """Renders the given DataFrame in a streamlit container."""
    with st.container():
        st.dataframe(dataframe)


def fetch_response(
    google_cloud_project_id: str,
    description_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    features: list[dict[str, Any]],
    llm_model: str,
    llm_temperature: float,
    scoring_prompt_template: Optional[
        Union[PromptTemplate, FewShotPromptTemplate]
    ] = None,
    has_additional_context: bool = False,
) -> list[dict[str, str]]:
    """Fetches generated text from Google Cloud Vertex AI.

    Applies Langchain's LLMChain on the provided list of inputs to generate
    text descriptions.

    Args:
        google_cloud_project_id (str): Google cloud project to use for text
            generation.
        description_prompt_template (FewShotPromptTemplate): Langchain Prompt
            template for the description.
        features (list[dict[str, Any]]): Product Attributes
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

    description_chain = LLMChain(
        prompt=description_prompt_template,
        llm=llm,
        output_key="output_description",
        verbose=True,
    )

    sequence_chains: Sequence = [description_chain]
    output_variables: Sequence = ["output_description"]
    input_variables: Sequence = ["input_features"]
    if has_additional_context:
        input_variables.append("additional_context")

    if scoring_prompt_template:
        scoring_llm = VertexAI(
            project=google_cloud_project_id,
            location="us-central1",
            model_name="text-bison@001",
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
        )
        scoring_chain = LLMChain(
            prompt=scoring_prompt_template,
            llm=scoring_llm,
            output_key="score",
            verbose=True,
        )
        sequence_chains.append(scoring_chain)
        output_variables.append("score")

    overall_chain = SequentialChain(
        chains=sequence_chains,
        input_variables=input_variables,
        output_variables=output_variables,
        verbose=True,
    )
    return overall_chain.apply(features)


@st.cache_data
def score_description(description: str) -> float:
    """Calculates a quality score for a given description.

    Args:
        description (str): _description_

    Returns:
        float: Quality score between -1 and 1.
    """
    raise NotImplementedError


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
with st.sidebar:
    st.subheader("Configuration")
    llm_settings_container = st.container()
    llm_settings_col1, llm_settings_col2 = llm_settings_container.columns(2)

    gcp_id = st.text_input("Google Cloud Project Id", placeholder="cloud-project-id")
    llm_model_name = (
        st.selectbox("LLM model", (model for model in _AVAILABLE_MODELS)) or ""
    )
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        format="%.1f",
    )

with st.expander("**Data Upload**", expanded=True):
    input_data = st.file_uploader("Upload your CSV table", type=["csv"])
    prompt_features_list = []
    prompt_features = []  # A list of dictionaries needed for Langchain prompting.
    if input_data:
        input_df = load_dataframe(input_data)
        remove_html_tags = st.checkbox("Remove html tags", value=True)
        processed_df = preprocess_dataframe(input_df, remove_html_tags)
        render_dataframe(processed_df)
        with st.container():
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

with st.form("generation_config"):
    st.write("**Prompt**")
    prompt_llm_role = st.text_input(
        "Provide a role to the LLM",
        placeholder="You are an expert digital marketer. Your job is to write creative ads based on the provided data.",
    )
    prompt_llm_guidelines = st.text_area(
        "Provide any guidelines that the LLM should consider",
        height=150,
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
        "Generation prompt",
        placeholder="Generate text descriptions based on the given ###FEATURES### and ###GUIDELINES###.",
    )
    st.write("**Few-Shot examples**")
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

    prompt_additional_context = None  # Variable so pylint: disable=C0103

    _PROMPT_SUFFIX = """
    Input features: {input_features}
    Output description:
    """

    with st.expander("Advanced"):
        additional_context_col, forbidden_words_col = st.columns(2)
        additional_context_col.write("**Additional Context**")
        context_docs = additional_context_col.file_uploader(
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
            context_docs_db = chroma_client.get_or_create_collection(
                name="context_docs_db"
            )
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
            prompt_additional_context = "\nNow I will provide some Additional Context for generating the descriptions: {additional_context}\n\n"

        forbidden_words_col.write("**Forbidden Words**")
        filter_words_str = None
        filter_words_str = forbidden_words_col.text_area(
            "List of forbidden words",
            height=100,
            placeholder="Enter a comma-separated list of words that should not occur in the results.",
        )
        enable_word_filtering = forbidden_words_col.toggle(
            "Enable forbidden word scanning"
        )

        st.write("**Scoring**")
        scoring_template = None  # Variable so pylint: disable=C0103
        scoring_prompt = st.text_area(
            "Scoring criteria for the LLM",
            height=200,
            value="""
                Here is the scoring Criteria:
                Criteria 1: Repeating sentences depict poor quality and should be
                scored low.
                Criteria 2: The text should strictly be about the provided product.
                Correct product type, number of items contained in the the product
                as well as product features such as color should be followed. A list
                of product attributes is provided to you for your scoring.
                Criteria 3: Hyperbolic text, over promising or guarantees.

                Assign a score of 1-5 to the product description, based on the above criteria:
                4-5 points: The product description is accurate, well-structured, unique, uses appropriate language, and has an informal tone.
                3-3.5 points: The product description meets most of the criteria, but may have some minor issues, such as a few repeating keywords or phrases, or a slightly too formal tone.
                2-2.5 points: The product description meets some of the criteria, but has some significant issues, such as inaccurate information, poor structure, or excessive hyperbole.
                1-1.5 points: The product description meets very few of the criteria and is of low quality.
            """,
        )
        enable_scoring = st.toggle("Enable scoring")

        SCORING_PROMPT_PREFIX = """
            You are a critic with an IQ of 140 and an expert in content creation who scores a generated product description based on the following criteria and points per criterion.

            Instructions for you:
            Read the product description carefully and compare it to these product attributes: Here are the Product Attributes: {input_features}
        """

        SCORING_PROMPT_SUFFIX = """
            Now, I will provide you the Product Description that you need to score:
            {output_description}

            Please provide a score, formatted as [SCORE] followed by your reasoning with examples. Let's work this out step by step to make sure we have the right answer.
        """

        scoring_template = PromptTemplate(
            input_variables=["output_description", "input_features"],
            template=f"{SCORING_PROMPT_PREFIX}\n{scoring_prompt}\n{SCORING_PROMPT_SUFFIX}",
        )

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
            k=2,  # TODO: Make this a user-provided value.
        )

        description_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prompt_prefix,
            suffix=_PROMPT_SUFFIX,
            input_variables=prompt_input_variables,
        )
    # If no examples provided, use a standard prompt instead.
    else:
        description_template = PromptTemplate.from_template(
            f"{prompt_prefix}\n{_PROMPT_SUFFIX}"
        )

    generate_button = st.form_submit_button(
        "Generate",
        disabled=not (gcp_id and input_data),
        type="primary",
    )

if generate_button:
    with st.spinner("Running request..."):
        results = fetch_response(
            gcp_id,
            description_template,
            prompt_features,
            llm_model_name,
            temperature,
            scoring_template,
            prompt_additional_context is not None,
        )
        if results:
            results_columns = ["output_description"]
            if enable_scoring:
                results_columns.append("score")
            results_df = pd.DataFrame.from_records(
                data=results, columns=results_columns
            )
            if enable_word_filtering:
                if filter_words_str:
                    filter_words = filter_words_str.split(",")
                    results_df["contains_forbidden_words"] = results_df[
                        "output_description"
                    ].str.contains("|".join(filter_words), na=False)

            results_view = st.data_editor(
                results_df,
                use_container_width=True,
                hide_index=True,
            )
            csv = convert_df(results_view)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="text_descriptions.csv",
                mime="text/csv",
            )
        else:
            st.warning("No results were returned.", icon="⚠️")
