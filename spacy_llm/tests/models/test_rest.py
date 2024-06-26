# mypy: ignore-errors
import copy
import re
from typing import Iterable, Optional, Tuple

import pytest
import spacy
from spacy.tokens import Doc

from ...registry import registry
from ..compat import has_azure_openai_key, has_openai_key

PIPE_CFG = {
    "model": {
        "@llm_models": "spacy.OpenAI.v1",
    },
    "task": {"@llm_tasks": "spacy.TextCat.v1", "labels": "POSITIVE,NEGATIVE"},
}


@registry.llm_tasks("spacy.Count.v1")
class _CountTask:
    _PROMPT_TEMPLATE = "Count the number of characters in this string: '{text}'."

    def generate_prompts(
        self, docs: Iterable[Doc], context_length: Optional[int] = None
    ) -> Iterable[Tuple[Iterable[str], Iterable[Doc]]]:
        for doc in docs:
            yield [_CountTask._PROMPT_TEMPLATE.format(text=doc.text)], [doc]

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        # Grab the first shard per doc
        return [list(shards_for_doc)[0] for shards_for_doc in shards]

    @property
    def prompt_template(self) -> str:
        return _CountTask._PROMPT_TEMPLATE


def test_initialization():
    """Test initialization and simple run"""
    nlp = spacy.blank("en")
    cfg = copy.deepcopy(PIPE_CFG)
    cfg["model"] = {"@llm_models": "spacy.NoOp.v1"}
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_model_error_handling():
    """Test error handling for wrong model."""
    nlp = spacy.blank("en")
    with pytest.raises(ValueError, match="is not available"):
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.NoOp.v1"},
                "model": {"@llm_models": "spacy.OpenAI.v1", "name": "GPT-3.5-x"},
            },
        )


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_doc_length_error_handling():
    """Test error handling for excessive doc length."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            # Not using the NoOp task is necessary here, as the NoOp task sends a fixed-size prompt.
            "task": {"@llm_tasks": "spacy.Count.v1"},
            "model": {"config": {"model": "ada"}},
        },
    )
    # Call with an overly long document to elicit error.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Request to OpenAI API failed: This model's maximum context length is 16385 tokens. However, your messages "
            "resulted in 40018 tokens. Please reduce the length of the messages."
        ),
    ):
        nlp("this is a test " * 10000)


@pytest.mark.skipif(has_openai_key is False, reason="OpenAI API key not available")
@pytest.mark.external
def test_max_time_error_handling():
    """Test error handling for exceeding max. time."""
    nlp = spacy.blank("en")
    with pytest.raises(
        TimeoutError,
        match="Request time out. Check your network connection and the API's availability.",
    ):
        nlp.add_pipe(
            "llm",
            config={
                "task": {"@llm_tasks": "spacy.Count.v1"},
                "model": {
                    "config": {"model": "ada"},
                    "max_request_time": 0.001,
                    "max_tries": 1,
                    "interval": 0.001,
                },
            },
        )


@pytest.mark.skipif(
    has_azure_openai_key is False, reason="Azure OpenAI API key not available"
)
@pytest.mark.external
@pytest.mark.parametrize("deployment_name", ("gpt-35-turbo", "gpt-35-turbo-instruct"))
def test_azure_openai(deployment_name: str):
    """Test initialization and simple run for Azure OpenAI."""
    nlp = spacy.blank("en")
    _pipe_cfg = {
        "model": {
            "@llm_models": "spacy.Azure.v1",
            "base_url": "https://explosion.openai.azure.com/",
            "model_type": "completions",
            "deployment_name": deployment_name,
            "name": deployment_name.replace("35", "3.5"),
        },
        "task": {"@llm_tasks": "spacy.NoOp.v1"},
        "save_io": True,
    }

    cfg = copy.deepcopy(_pipe_cfg)
    nlp.add_pipe("llm", config=cfg)
    nlp("This is a test.")
