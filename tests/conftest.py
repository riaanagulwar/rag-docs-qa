"""Shared test setup.

app.embeddings creates a genai.Client at import time; patch it out globally
before any test module imports app.embeddings (directly, or via app.main /
eval.run_eval), so the whole suite runs offline -- no GEMINI_API_KEY, no
network access, no Postgres required. Individual tests still stub the
specific methods (embed_content, generate_content) they need.
"""
from unittest import mock

mock.patch("google.genai.Client").start()
