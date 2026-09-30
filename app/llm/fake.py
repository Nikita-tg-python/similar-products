"""FakeLLM: no network, no keys, deterministic. Used by tests and by `LLM_PROVIDER=fake` (an
offline demo).

By default it answers the query-parsing prompt from a phrase book (the task's example and a few
more); unknown queries get "no filters, search the text as is". Tests can script exact replies
instead (e.g. broken JSON to exercise the retry).
"""

import json
from collections import deque
from collections.abc import Callable, Iterable

from app.llm.base import LLMClient, Message

# A scripted reply: text, or a function of the messages returning text.
Script = str | Callable[[list[Message]], str]

NO_FILTERS = {
    "exclude_allergens": [],
    "vegan_required": None,
    "nutriscore_max": None,
    "sugar_level": "any",
    "unsupported": [],
}

PHRASE_BOOK: dict[str, dict] = {
    "веганське печиво без горіхів з низьким вмістом цукру": {
        "semantic_text": "cookies biscuits",
        "exclude_allergens": ["nuts"],
        "vegan_required": True,
        "nutriscore_max": None,
        "sugar_level": "low",
    },
    "vegan cookies without nuts, low in sugar": {
        "semantic_text": "cookies biscuits",
        "exclude_allergens": ["nuts"],
        "vegan_required": True,
        "nutriscore_max": None,
        "sugar_level": "low",
    },
    "йогурт без лактози з нутрі-скором не гірше b": {
        "semantic_text": "yogurt",
        "exclude_allergens": ["milk"],
        "vegan_required": None,
        "nutriscore_max": "b",
        "sugar_level": "any",
    },
    "sausages without pork": {
        "semantic_text": "sausages",
        **NO_FILTERS,
        "unsupported": ["without pork"],
    },
    "шоколадна паста без горіхів і арахісу": {
        "semantic_text": "chocolate spread",
        "exclude_allergens": ["nuts", "peanuts"],
        "vegan_required": None,
        "nutriscore_max": None,
        "sugar_level": "any",
    },
}


def phrase_book_reply(messages: list[Message]) -> str:
    query = messages[-1].content.strip().lower()
    parsed = PHRASE_BOOK.get(query, {"semantic_text": messages[-1].content.strip(), **NO_FILTERS})
    return json.dumps(parsed, ensure_ascii=False)


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, replies: Iterable[Script] | None = None) -> None:
        self._replies = deque(replies) if replies is not None else None
        self.calls: list[list[Message]] = []  # every request, for assertions

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        self.calls.append(list(messages))
        if self._replies is None:
            return phrase_book_reply(messages)
        if not self._replies:
            raise AssertionError("FakeLLM: no scripted replies left")
        script = self._replies.popleft()
        return script(messages) if callable(script) else script
