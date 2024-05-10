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
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Union

import chromadb
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain, SequentialChain
from langchain.docstore.document import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
    VertexAI,
)

import utils

_AVAILABLE_MODELS = ["gemini-1.0-pro-001", "gemini-pro", "text-bison-32k", "text-bison"]

_CHROMA_METADATA = {"hnsw:space": "cosine"}

_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


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


def render_dataframe(
    dataframe: pd.DataFrame, column_config: Optional[dict] = None
) -> None:
    """Renders the given DataFrame in a streamlit container."""
    with st.container():
        st.dataframe(dataframe, column_config=column_config, hide_index=True)


def selectable_dataframe(
    dataframe: pd.DataFrame, column_config: Optional[dict] = None
) -> pd.DataFrame:
    """(Re-)renders a dataframe and returns the selected rows.

    This function provides a workaround for missing streamlit functionality to
    get user selected rows from a DataFrame / Data editior. It (re-)renders a
    given DataFrame on the UI with all the selected rows and returns those
    selected rows as a new DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Selected rows from the Data editor.
    """
    selections_df = dataframe.copy()
    selections_df.insert(0, "Select", False)

    results_view = st.data_editor(
        selections_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )

    selected_rows = results_view[results_view.Select]
    return selected_rows.drop("Select", axis=1)


def transform_input_features_to_multimodal(
    features: Sequence[dict[str, Any]],
    description_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    has_additional_context: bool = False,
):
    """Transforms input features and prompt template to multimodal prompts."""
    generation_prompts = []
    for feature in features:
        description_prompt = description_prompt_template.format(
            input_features=feature["input_features"],
            additional_context=(
                feature["additional_context"] if has_additional_context else ""
            ),
        )
        description_message = get_vertexai_message(
            message_content=description_prompt, message_type="text"
        )

        product_image = get_vertexai_message(
            message_content=feature["image_url"], message_type="image_url"
        )

        image_message = get_vertexai_message(
            message_content="""You are provided an image of this product. Extract all of
              the information from this image. Then use the provided text attributes and
              the additional information you have extracted from the image to write a
              product description.""",
            message_type="text",
        )

        generation_prompts.append(
            [HumanMessage(content=[description_message, image_message, product_image])]
        )
    return generation_prompts


def get_vertexai_message(message_content: str, message_type: str) -> dict[str, Any]:
    """Formats a message to be sent to a Vertex AI Multimodal endpoint."""
    if message_type == "text":
        return {"type": message_type, "text": message_content}
    elif message_type == "image_url":
        return {"type": message_type, "image_url": {"url": message_content}}
    else:
        raise ValueError(f"Invalid message type: {message_type}")


def fetch_response_multimodal(
    google_cloud_project_id: str,
    description_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    features: list[dict[str, Any]],
    llm_temperature: float,
    has_additional_context: bool = False,
    multimodal_model: str = "gemini-1.5-pro-preview-0409",
):
    """Fetches generated text from a Google Cloud Vertex AI multimodal model.

    Args:
        google_cloud_project_id (str): Google cloud project to use for text
            generation.
        description_prompt_template (Union[PromptTemplate, FewShotPromptTemplate]):
            Langchain Prompt template for the description.
        features (list[dict[str, Any]]): Product Attributes.
        llm_temperature (float): LLM setting to control how imaginative the
            model can be.
        has_additional_context (bool, optional): If any additional context should be included in
            prompt. Defaults to False.
        multimodal_model (str, optional): The Google Cloud Vertex AI multimodal model to
            use. Defaults to "gemini-1.5-pro-preview-0409".

    Returns:
        list[dict[str, str]]: A list of generated texts.
    """
    llm = ChatVertexAI(
        project=google_cloud_project_id,
        model_name=multimodal_model,
        temperature=llm_temperature,
        verbose=True,
        safety_settings=_SAFETY_SETTINGS,
    )

    generation_prompts = transform_input_features_to_multimodal(
        features=features,
        description_prompt_template=description_prompt_template,
        has_additional_context=has_additional_context,
    )

    output = llm.batch(inputs=generation_prompts)
    output_result = []
    for llm_response, product_feature in zip(output, features):
        output_result.append(
            {
                "input_features": product_feature["input_features"],
                "generated_description": llm_response.content,
            }
        )

    return output_result


def fetch_response(
    google_cloud_project_id: str,
    description_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    features: list[dict[str, Any]],
    llm_model: str,
    llm_temperature: float,
    has_additional_context: bool = False,
    has_image: bool = False,
) -> list[dict[str, str]]:
    """Fetches generated text from Google Cloud Vertex AI.

    Applies Langchain's LLMChain on the provided list of inputs to generate
    text descriptions.

    Args:
        google_cloud_project_id (str): Google cloud project to use for text
            generation.
        description_prompt_template (Union[PromptTemplate, FewShotPromptTemplate]):
            Langchain Prompt template for the description.
        features (list[dict[str, Any]]): Product Attributes
        llm_model (str): Vertex AI model to use for text generation.
        llm_temperature (float): LLM setting to control how imaginative the
            model can be.
        has_additional_context (bool): If any additional context should be included in
            prompt. Defaults to False.
        has_image (bool): If the product has an image that must be used for description.
            Defaults to False.


    Returns:
        list[dict[str, str]]: A list of generated texts.
    """
    if has_image:
        return fetch_response_multimodal(
            google_cloud_project_id,
            description_prompt_template,
            features,
            llm_temperature,
            has_additional_context,
        )
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
        output_key="generated_description",
        verbose=True,
    )

    sequence_chains: Sequence = [description_chain]
    output_variables: Sequence = ["generated_description"]
    input_variables: Sequence = ["input_features"]
    if has_additional_context:
        input_variables.append("additional_context")

    overall_chain = SequentialChain(
        chains=sequence_chains,
        input_variables=input_variables,
        output_variables=output_variables,
        verbose=True,
    )
    return overall_chain.apply(features)


def score_descriptions(
    google_cloud_project_id: str,
    llm_model: str,
    descriptions: Iterable[Mapping[str, str]],
    criteria: Iterable[Mapping[Hashable, Any]],
    total_points: float,
    passing_points: float,
) -> Sequence[Mapping[Hashable, str]]:
    scoring_llm = VertexAI(
        project=google_cloud_project_id,
        location="us-central1",
        model_name=llm_model,
        temperature=0.3,
        max_output_tokens=8192,
        top_k=40,
    )

    class Score(BaseModel):
        score: str = Field(
            description="single character Y if ALL criterion are met or N if some criteria are not met (without quotes or punctuation)"
        )
        reasoning: str = Field(description="Step by step reasoning about the criterion")

    parser = JsonOutputParser(pydantic_object=Score)

    prompt = PromptTemplate(
        template="""[BEGIN DATA]
        You are a critic that scores a given a submission based on the given criteria. Now, I will provide the submission, followed by the scoring Criterion.
  ***
  [Submission]:
        {description}
  ***
  [Criterion]:
        {criterion}
  [END DATA]
  Does the submission meet the criterion? First, reason about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Return a JSON object formatted as: {format_instructions}""",
        input_variables=["features", "description", "criterion"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    runnable = prompt | scoring_llm | parser
    description_evals = []
    for description in descriptions:
        current_score = total_points
        score_details = []
        for criterion in criteria:
            # Stop evaluation if description failed already.
            if current_score < passing_points:
                break
            criteria_points = criterion["Points"]
            try:
                eval_result = runnable.invoke(
                    {
                        "features": description["input_features"],
                        "description": description["generated_description"],
                        "criterion": criterion,
                    }
                )
                if eval_result["score"] != "Y":
                    current_score -= criteria_points
                score_details.append(
                    "Reasoning: "
                    + eval_result["reasoning"]
                    + "Passing: "
                    + eval_result["score"]
                )
            except (OutputParserException, KeyError) as e:
                score_details.append(
                    "Reasoning: An error occurred while evaluating the description. Error: "
                    + str(e)
                    + "Passing: N"
                )
        description_evals.append(
            {
                "input_features": description["input_features"],
                "generated_description": description["generated_description"],
                "evaluation_details": "\n\n".join(score_details),
                "passed": current_score >= passing_points,
                "score": current_score,
            }
        )
    return description_evals


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
        image_column = st.selectbox(
            "Image Column",
            input_df.columns.tolist(),
            index=None,
            placeholder="Select an image column if available.",
        )
        use_image_for_generation = st.checkbox(
            "Use image attributes for description",
            value=False,
            disabled=image_column is None,
            help="Extract additional product attributes from the given image. **Requires a multimodal model e.g. Gemini Pro Vision. May incur additional costs**.",
        )
        input_data_columns_config = {}
        if image_column is not None:
            input_data_columns_config[image_column] = st.column_config.ImageColumn()
        remove_html_tags = st.checkbox("Remove html tags", value=True)
        processed_df = preprocess_dataframe(input_df, remove_html_tags)
        render_dataframe(processed_df, input_data_columns_config)
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
        if image_column is not None and use_image_for_generation:
            image_links_list = processed_df[image_column].to_list()
            prompt_features = [
                {"input_features": val, "image_url": url}
                for val, url in zip(prompt_features_list, image_links_list)
            ]
        else:
            prompt_features = [{"input_features": val} for val in prompt_features_list]

with st.form("generation_config"):
    st.write("**Prompt**")
    prompt_llm_role = st.text_input(
        "Provide a role to the LLM",
        value="You are an expert digital marketer. Your job is to write creative ads based on the provided data.",
    )
    prompt_llm_guidelines = st.text_area(
        "Provide any guidelines that the LLM should consider",
        height=150,
        value="###GUIDELINES###\n- Always be truthful and present factual information only.\n- Only use the provided features for generating the text.\n- Structure your output in three sections, each with its headline. First section should give a general overview, second should discuss the materials and third should provide any usage or styling guidelines.\n- Use HTML formatting. Section headings should be <h2>. The text following should be in <p> blocks. Any keywords that could convince a user about purchasing our product should be highlighted with <strong>.",
    )

    examples_df = pd.DataFrame(
        [
            {
                "input": "feature 1: value 1, feature 2: value 2, ..., feature n",
                "output": "<h2> Lorem ipsum dolor sit amet</h2>\n<p> Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>",
            }
        ]
    )
    prompt_action = st.text_input(
        "Generation prompt",
        value="Generate text descriptions based on the given ###FEATURES### and ###GUIDELINES###.",
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
                        k=4,
                    )
                ]
                context_list.append(". ".join(context_docs_list))
            # Add additional context to the prompt_features dictionary.
            for feature, context in zip(prompt_features, context_list):
                feature["additional_context"] = context
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
        scoring_df = pd.DataFrame(columns=["Criterion", "Points"])
        scoring_criteria = st.data_editor(
            data=scoring_df,
            column_config={
                "Criterion": st.column_config.TextColumn(width="large"),
                "Points": st.column_config.NumberColumn(
                    "Points",
                    help="Assign a numeric value to signify the importance of a particular criterion in the total scoring e.g. **Criterion Points > Min. passing score** ensure the entire description fails the quality check if this criterion is not passed.",
                    width="small",
                    required=True,
                ),
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )
        scoring_prompt = ""
        scoring_total_points_col, passing_score_col = st.columns(2)
        scoring_total_points = scoring_total_points_col.number_input(
            "Total available points",
            0,
            1000,
            value=0,
            help="Highest score a description can achieve. All descriptions start with this score and lose points if they fail a criterion.",
        )
        passing_score = passing_score_col.number_input(
            "Minimum score required to pass",
            -1000,
            1000,
            value=0,
            help="Minimum score required for a description to pass a quality check.",
        )
        enable_scoring = st.toggle("Enable scoring")

        SCORING_PROMPT_PREFIX = """
            You are a critic with an IQ of 140 and an expert in content creation who scores a generated product description based on the following criteria and points per criterion.

            Instructions for you:
            Read the product description carefully and compare it to the given product attributes.
            Review the quality of the generated description based on the given scoring criteria and output your findings.
            Format it as `Quality Review: ____`. Then, provide a final score based on your quality review. Format it as `Final score is: |SCORE|`.
            Let's work this out step by step.
        """

        SCORING_PROMPT_SUFFIX = """
            Here are the product attributes and the generated product description.
            Do the quality review and provide your score.

            Product Attributes: {input_features}
            Product Description: {generated_description}
            Your quality review and final score:
        """

        scoring_template = PromptTemplate(
            input_variables=["generated_description", "input_features"],
            template=f"{SCORING_PROMPT_PREFIX}\n{scoring_prompt}\n{SCORING_PROMPT_SUFFIX}",
        )

    prompt_input_variables = ["input_features"]
    if prompt_additional_context:
        prompt_input_variables.append("additional_context")

    prompt_prefix = f"{prompt_llm_role}\n{prompt_llm_guidelines}\n{prompt_additional_context}\n{prompt_action}"

    # If examples have been provided, use a Few Shot Prompt.
    edited_df.dropna(how="all", inplace=True)
    if not edited_df.empty:
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

if "results" not in st.session_state:
    st.session_state.results = None

if generate_button:
    with st.spinner("Running request..."):
        results = fetch_response(
            gcp_id,
            description_template,
            prompt_features,
            llm_model_name,
            temperature,
            prompt_additional_context is not None,
            use_image_for_generation and image_column is not None,
        )
        if results:
            st.session_state.results = results
            if enable_scoring:
                results_evals = score_descriptions(
                    gcp_id,
                    llm_model_name,
                    st.session_state.results,
                    scoring_criteria.to_dict("records"),
                    scoring_total_points,
                    passing_score,
                )
                if results_evals:
                    st.session_state.results = results_evals
                else:
                    st.warning("An error occured when scoring descriptions.", icon="⚠️")
        else:
            st.warning("No results were returned.", icon="⚠️")

if st.session_state.results is not None:
    results_df = pd.DataFrame.from_records(data=st.session_state.results)
    if enable_word_filtering:
        if filter_words_str:
            filter_words = filter_words_str.split(",")
            results_df["contains_forbidden_words"] = results_df[
                "generated_description"
            ].str.contains("|".join(filter_words), na=False)
    output_df_column_config = {"Select": st.column_config.CheckboxColumn(required=True)}
    if image_column is not None:
        results_df.insert(loc=0, column=image_column, value=input_df[image_column])
        # results_df[image_column] = input_df[image_column]
        output_df_column_config[image_column] = st.column_config.ImageColumn()

    # Workaround to render the results DataFrame with selectable rows.
    # TODO(): Update this when built-in selection functionality becomes
    # available in Streamlit.
    selected_results = selectable_dataframe(results_df, output_df_column_config)
    selected_indices = selected_results.index
    regenerate_features = [prompt_features[i] for i in selected_indices]

    regenerate_button = st.button("Regenerate Selected")
    if regenerate_button:
        with st.spinner("Running request..."):
            regenerated_results = fetch_response(
                gcp_id,
                description_template,
                regenerate_features,
                llm_model_name,
                temperature,
                prompt_additional_context is not None,
                use_image_for_generation and image_column is not None,
            )
            if regenerated_results:
                if enable_scoring:
                    regenerated_results = score_descriptions(
                        gcp_id,
                        llm_model_name,
                        regenerated_results,
                        scoring_criteria.to_dict("records"),
                        scoring_total_points,
                        passing_score,
                    )
                for index, result in zip(selected_indices, regenerated_results):
                    st.session_state.results[index] = result
                st.rerun()
            else:
                st.warning("No results were returned.", icon="⚠️")

    csv = convert_df(results_df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="text_descriptions.csv",
        mime="text/csv",
    )
