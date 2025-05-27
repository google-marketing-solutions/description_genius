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

"""Functions to fetch and generate text from LLMs."""

from collections.abc import Hashable
from typing import Any, Dict, Generator, Iterable, List, Mapping, Sequence, Union

import langchain.chains
from langchain_core.exceptions import LangChainException
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold
from langchain_google_vertexai import HarmCategory
from langchain_google_vertexai import VertexAI
import pydantic
import streamlit as st

AVAILABLE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


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

    generation_prompts.append([
        HumanMessage(
            content=[description_message, image_message, product_image]
        )
    ])
  return generation_prompts


def get_vertexai_message(message_content: str,
                         message_type: str) -> dict[str, Any]:
  """Formats a message to be sent to a Vertex AI Multimodal endpoint."""
  if message_type == "text":
    return {"type": message_type, "text": message_content}
  elif message_type == "image_url":
    return {"type": message_type, "image_url": {"url": message_content}}
  else:
    raise ValueError(f"Invalid message type: {message_type}")


def fetch_response_multimodal(
    google_cloud_project_id: str,
    region: str,
    description_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    features: list[dict[str, Any]],
    llm_temperature: float,
    multimodal_model: str,
    has_additional_context: bool = False,
) -> Generator[dict[str, str], None, None]:
  """Fetches generated text from a Google Cloud Vertex AI multimodal model.

  Args:
      google_cloud_project_id (str): Google cloud project to use for text
        generation.
      region (str): Google cloud region to process the data in.
      description_prompt_template: Langchain Prompt template for the
        description.
      features (list[dict[str, Any]]): Product Attributes.
      llm_temperature (float): LLM setting to control how imaginative the model
        can be.
      multimodal_model (str, optional): The Google Cloud Vertex AI multimodal
        model to use.
      has_additional_context (bool, optional): If any additional context should
        be included in prompt. Defaults to False.

  Yields:
      Generator[dict[str, str], None, None]: A generator yielding generated
      texts.
  """
  llm = ChatVertexAI(
      project=google_cloud_project_id,
      location=region,
      model_name=multimodal_model,
      temperature=llm_temperature,
      verbose=True,
      safety_settings=_SAFETY_SETTINGS,
  )

  generation_prompts_for_llm = transform_input_features_to_multimodal(
      features=features,
      description_prompt_template=description_prompt_template,
      has_additional_context=has_additional_context,
  )

  for single_feature_prompts, product_feature in zip(
      generation_prompts_for_llm, features
  ):
    try:
      llm_response = llm.invoke(input=single_feature_prompts)
      yield {
          "input_features": product_feature["input_features"],
          "generated_description": llm_response.content,
      }
    except (
        pydantic.ValidationError,
        LangChainException,
        RuntimeError,
        AttributeError,
    ) as e:
      st.error(
          "Error generating (multimodal) description for feature"
          f" '{product_feature.get('input_features', 'Unknown')}': {e}"
      )
      yield {
          "input_features": product_feature.get(
              "input_features", "Error during generation"
          ),
          "generated_description": f"Error: {e}",
      }


def fetch_response(
    google_cloud_project_id: str,
    region: str,
    description_prompt_template: Union[PromptTemplate, FewShotPromptTemplate],
    features: list[dict[str, Any]],
    llm_model: str,
    llm_temperature: float,
    has_additional_context: bool = False,
    has_image: bool = False,
) -> Generator[dict[str, str], None, None]:
  """Fetches generated text from Google Cloud Vertex AI.

  Applies Langchain's LLMChain on the provided list of inputs to generate
  text descriptions.

  Args:
    google_cloud_project_id (str): Google cloud project to use for text
      generation.
    region (str): Google cloud region to process the data in.
    description_prompt_template: Langchain Prompt template for the description.
    features (list[dict[str, Any]]): Product Attributes
    llm_model (str): Vertex AI model to use for text generation.
    llm_temperature (float): LLM setting to control how imaginative the model
      can be.
    has_additional_context (bool): If any additional context should be
      included in prompt. Defaults to False.
    has_image (bool): If the product has an image that must be used for
      description. Defaults to False.

  Yields:
      Generator[dict[str, str], None, None]: A generator yielding generated
        texts.
  """
  if has_image:
    yield from fetch_response_multimodal(
        google_cloud_project_id,
        region,
        description_prompt_template,
        features,
        llm_temperature,
        llm_model,
        has_additional_context,
    )
    return
  llm = VertexAI(
      project=google_cloud_project_id,
      location=region,
      model_name=llm_model,
      temperature=llm_temperature,
      max_output_tokens=8192,
      top_p=0.8,
      top_k=40,
  )

  description_chain = langchain.chains.LLMChain(
      prompt=description_prompt_template,
      llm=llm,
      output_key="generated_description",
      verbose=True,
  )

  sequence_chains: Sequence[langchain.chains.LLMChain] = [description_chain]
  output_variables: Sequence[str] = ["generated_description"]
  input_variables: List[str] = ["input_features"]
  if has_additional_context:
    input_variables.append("additional_context")

  overall_chain = langchain.chains.SequentialChain(
      chains=sequence_chains,
      input_variables=input_variables,
      output_variables=output_variables,
      verbose=True,
  )

  for feature_set in features:
    try:
      result = overall_chain.invoke(feature_set)
      yield result
    except (
        pydantic.ValidationError,
        LangChainException,
        RuntimeError,
        AttributeError,
    ) as e:
      st.error(
          "Error generating description for feature"
          f" '{feature_set.get('input_features', 'Unknown')}': {e}"
      )
      # Construct a dictionary matching the expected output structure.
      error_result = {
          key: feature_set.get(key, f"Missing {key}")
          for key in input_variables
      }
      error_result["generated_description"] = f"Error: {e}"
      for out_var in output_variables:
        if (
            out_var not in error_result
        ):  # e.g. if generated_description was not the only output_var
          error_result[out_var] = f"Error generating {out_var}: {e}"
      yield error_result


def score_descriptions(
    google_cloud_project_id: str,
    region: str,
    llm_model: str,
    descriptions: Iterable[Mapping[str, str]],
    criteria: Iterable[Mapping[Hashable, Any]],
    total_points: float,
    passing_points: float,
) -> Sequence[Mapping[Hashable, str]]:
  """Assigns a score to the given description including pass/fail.

  Args:
      google_cloud_project_id (str): Google cloud project to use for text
        generation.
      region (str): Google cloud region to process the data in.
      llm_model (str): Vertex AI model to use for text generation.
      descriptions (Iterable[Mapping[str, str]]): Descriptions to score.
      criteria (Iterable[Mapping[Hashable, Any]]): Scoring criteria.
      total_points (float): Total assignable points.
      passing_points (float): Least score required to pass.

  Returns:
      Sequence[Mapping[Hashable, str]]: Description scores.
  """
  scoring_llm = VertexAI(
      project=google_cloud_project_id,
      location=region,
      model_name=llm_model,
      temperature=0.3,
      max_output_tokens=8192,
      top_k=40,
  )

  class Score(pydantic.BaseModel):
    score: str = pydantic.Field(
        description=(
            "single character Y if ALL criterion are met or N if some criteria"
            " are not met (without quotes or punctuation)"
        )
    )
    reasoning: str = pydantic.Field(
        description="Step by step reasoning about the criterion"
    )

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
      partial_variables={
          "format_instructions": parser.get_format_instructions()
      },
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
        eval_result = runnable.invoke({
            "features": description["input_features"],
            "description": description["generated_description"],
            "criterion": criterion,
        })
        if eval_result["score"] != "Y":
          current_score -= criteria_points
        score_details.append(
            "Reasoning: " + eval_result["reasoning"] + "Passing: "
            + eval_result["score"]
        )
      except (OutputParserException, KeyError) as e:
        score_details.append(
            "Reasoning: An error occurred while evaluating the description."
            " Error: " + str(e) + "Passing: N"
        )
    description_evals.append({
        "input_features": description["input_features"],
        "generated_description": description["generated_description"],
        "evaluation_details": "\n\n".join(score_details),
        "passed": current_score >= passing_points,
        "score": current_score,
    })
  return description_evals


class Translation(pydantic.BaseModel):
  """A single translation of a text."""

  language: str = pydantic.Field(
      description="The language of the translation, e.g., 'english', 'german'"
  )
  text: str = pydantic.Field(description="The translated text.")


class MultilingualContent(pydantic.BaseModel):
  """Content with an original description and multiple translations."""

  original_description: str = pydantic.Field(
      description="The original text that was translated."
  )
  translations: List[Translation] = pydantic.Field(
      description="A list of translations of the original text."
  )


def translate_texts_to_json_multiple_languages(
    google_cloud_project_id: str,
    region: str,
    llm_model: str,
    texts_to_translate: List[str],
    target_languages: List[str],
    translation_guidelines: str,
    temperature: float,
) -> Generator[Dict[str, str], None, None]:
  """Translates a list of texts into multiple target languages using an LLM.

  Args:
      google_cloud_project_id: The Google Cloud Project ID.
      region (str): Google Cloud region to process the data in.
      llm_model: The Vertex AI model name to use for translation.
      texts_to_translate: A list of text strings to translate.
      target_languages: A list of target language names (e.g., ["Spanish",
        "French"]).
      translation_guidelines: Specific guidelines for the translation task.
      temperature: The temperature setting for the LLM.

  Yields:
      Generator[dict[str, str], None, None]: A generator yielding translated
      texts.
      Example: {"original_text": "Hello", "Spanish": "Hola", "French":
      "Bonjour"}
  """
  llm = ChatVertexAI(
      project=google_cloud_project_id,
      location=region,
      model_name=llm_model,
      temperature=temperature,
      max_output_tokens=8192,
      max_retries=3,
      top_p=0.8,
      top_k=40,
      safety_settings=_SAFETY_SETTINGS,
      verbose=True,
  )

  structured_llm = llm.with_structured_output(MultilingualContent)

  prompt_template_str = """
    Translate the following text into these languages: {target_language_list_str}.
    Adhere to these translation guidelines:
    {guidelines_str}

    Text to translate:
    "{text_input}"
    """
  prompt = PromptTemplate(
      template=prompt_template_str,
      input_variables=[
          "text_input",
          "target_language_list_str",
          "guidelines_str",
      ],
  )
  runnable = prompt | structured_llm
  target_language_list_str = ", ".join(target_languages)

  for text in texts_to_translate:
    try:
      parsed_output = runnable.invoke({
          "text_input": text,
          "target_language_list_str": target_language_list_str,
          "guidelines_str": (
              translation_guidelines
              if translation_guidelines else "No specific guidelines provided."
          ),
      })
      translations = {
          translation.language.capitalize(): translation.text
          for translation in parsed_output.translations
      }
      translation_entry = {"Original Description": text, **translations}
      yield translation_entry
    except (
        pydantic.ValidationError,
        LangChainException,
        RuntimeError,
        AttributeError,
    ) as e:
      error_entry = {
          "Original Description": text,
      }
      error_message = f"Translation failed with error: {e}"
      # Add keys for target languages with error message
      for lang in target_languages:
        error_entry[lang.capitalize()] = error_message
      yield error_entry
