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

"""Unit tests for llm_services module."""
import unittest
import unittest.mock

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

import llm_services


class TestLlmServices(unittest.TestCase):

  def test_get_vertexai_message_text(self):
    content = "Hello"
    expected = {"type": "text", "text": content}
    self.assertEqual(
        llm_services.get_vertexai_message(content, "text"), expected
    )

  def test_get_vertexai_message_image_url(self):
    url = "http://example.com/image.png"
    expected = {"type": "image_url", "image_url": {"url": url}}
    self.assertEqual(
        llm_services.get_vertexai_message(url, "image_url"), expected
    )

  def test_get_vertexai_message_invalid_type(self):
    with self.assertRaises(ValueError):
      llm_services.get_vertexai_message("some_content", "invalid_type")

  @unittest.mock.patch("llm_services.ChatVertexAI")
  def test_fetch_response_multimodal(self, mock_chat_vertexai):
    mock_llm_instance = mock_chat_vertexai.return_value
    mock_llm_instance.invoke.return_value = AIMessage(
        content="Multimodal description 1"
    )

    gcp_id = "test-project"
    region = "us-central1"
    description_template = PromptTemplate.from_template(
        "Input: {input_features}"
    )
    features = [{
        "input_features": "feature: value",
        "image_url": "http://example.com/image.png"
    }]
    temperature = 0.5
    model_name = "gemini-1.5-pro-001"

    # fetch_response returns a generator when has_image is True
    results_generator = llm_services.fetch_response(
        gcp_id, region, description_template, features, model_name, temperature,
        has_image=True
    )
    # Convert generator to list to make it subscriptable
    results = list(results_generator)

    self.assertEqual(len(results), 1)
    self.assertEqual(
        results[0]["generated_description"], "Multimodal description 1"
    )
    mock_chat_vertexai.assert_called_once_with(
        project=gcp_id, location=region, model_name=model_name,
        temperature=temperature, verbose=True,
        safety_settings=llm_services._SAFETY_SETTINGS
    )
    mock_llm_instance.invoke.assert_called_once()


if __name__ == "__main__":
  unittest.main()
