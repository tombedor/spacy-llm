from typing import Any, Dict, Optional

from confection import SimpleFrozenDict

from ....compat import Literal
from ....registry import registry
from .model import Endpoints, Local

_DEFAULT_TEMPERATURE = 0.0


@registry.llm_models("spacy.Local.v1")
def llama_instruct(
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=_DEFAULT_TEMPERATURE),
    name: str = "Meta-Llama-3-8B-Instruct-GGUF",
    strict: bool = Local.DEFAULT_STRICT,
    max_tries: int = Local.DEFAULT_MAX_TRIES,
    interval: float = Local.DEFAULT_INTERVAL,
    max_request_time: float = Local.DEFAULT_MAX_REQUEST_TIME,
    endpoint: Optional[str] = None,
    context_length: Optional[int] = None,
) -> Local:
    """Returns OpenAI instance for 'gpt-4' model using REST to prompt API.

    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    name (str): Model name to use. Can be any model name supported by the OpenAI API - e. g. 'gpt-4',
        "gpt-4-1106-preview", ....
    context_length (Optional[int]): Context length for this model. Only necessary for sharding and if no context length
        natively provided by spacy-llm.
    RETURNS (OpenAI): OpenAI instance for 'gpt-4' model.

    DOCS: https://spacy.io/api/large-language-models#models
    """
    return Local(
        name=name,
        endpoint=endpoint or Endpoints.CHAT.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )    

