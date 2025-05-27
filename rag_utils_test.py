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

import unittest

import langchain.docstore.document
import rag_utils


class TestRagUtils(unittest.TestCase):

  def test_get_context_document_creates_valid_document(self):
    doc_id = "5d41402abc4b2a76b9719d911017c592"
    doc_content = "hello"
    input_doc = langchain.docstore.document.Document(page_content=doc_content)
    expected_doc = rag_utils.CollectionDocument(doc_id, doc_content)
    context_doc = rag_utils.get_context_documents([input_doc])[0]
    self.assertEqual(context_doc, expected_doc)


if __name__ == "__main__":
  unittest.main()
