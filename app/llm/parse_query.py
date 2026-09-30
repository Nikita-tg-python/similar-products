"""Free-text query -> ParsedQuery: one LLM call (plus one retry if its JSON does not validate).

The LLM only reads the user's intent. It never sees the catalog and never decides which products
match: the vector search and the database filters do, deterministically (app/api/search.py).
"""

import logging
import re

from opentelemetry import trace
from pydantic import ValidationError

from app.llm.base import LLMClient, LLMError, Message
from app.schemas import ALLERGEN_ALIASES, ALLERGENS, ParsedQuery
from app.telemetry import span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You read a shopper's food search query (any language) and return ONE JSON object
with exactly these keys:

- "semantic_text": ONLY the kind of product, as 2-4 English nouns: the product type plus its
  common synonyms, since product data names the same thing differently (e.g. "pasta noodles",
  "chocolate bar candy", "yogurt yoghurt"). Translate if needed. It must NOT contain any word about
  diets, allergens, nutrition or negations: no "vegan", "gluten-free", "without", "low sugar",
  "healthy". Those go in the fields below; in a text embedding "vegan" would rank anything called
  vegan first and "without nuts" would pull nut products closer.
- "exclude_allergens": allergens the product must not contain, only from: {", ".join(ALLERGENS)}.
  Also accepted: {", ".join(ALLERGEN_ALIASES)}. "nuts" means tree nuts; add "peanuts" only if the
  user mentions peanuts. Lactose-free / dairy-free -> "milk"; gluten-free -> "gluten". [] if none.
- "vegan_required": true if the user wants vegan products, otherwise null.
- "nutriscore_max": the worst acceptable Nutri-Score letter "a".."e" if the user states one
  ("Nutri-Score A or B" -> "b"), otherwise null. Do not infer it from words like "healthy".
- "sugar_level": "low" if the user asks for low, reduced or no sugar, otherwise "any".
- "unsupported": every other constraint the user stated that the fields above cannot express,
  in the user's words (e.g. "without pork", "organic", "under 2 euros"). [] if none. Never drop a
  constraint silently: the user is told these were not applied.

Never add a constraint the user did not state. Output the JSON object only.

Examples:
"gluten-free vegan pasta, Nutri-Score A" ->
{{"semantic_text": "pasta noodles", "exclude_allergens": ["gluten"], "vegan_required": true,
 "nutriscore_max": "a", "sugar_level": "any", "unsupported": []}}
"шоколад без молока та цукру" ->
{{"semantic_text": "chocolate bar candy", "exclude_allergens": ["milk"], "vegan_required": null,
 "nutriscore_max": null, "sugar_level": "low", "unsupported": []}}
"organic sparkling water" ->
{{"semantic_text": "sparkling mineral water", "exclude_allergens": [], "vegan_required": null,
 "nutriscore_max": null, "sugar_level": "any", "unsupported": ["organic"]}}"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _validate(raw: str) -> ParsedQuery:
    return ParsedQuery.model_validate_json(_FENCE_RE.sub("", raw.strip()))


async def parse_query(llm: LLMClient, query: str) -> ParsedQuery:
    with span("llm_parse", **{"llm.provider": llm.provider, "query.chars": len(query)}) as s:
        parsed = await _parse(llm, query, s)
        s.set_attribute("parsed.semantic_text", parsed.semantic_text)
        return parsed


async def _parse(llm: LLMClient, query: str, s: trace.Span) -> ParsedQuery:
    messages = [Message("system", SYSTEM_PROMPT), Message("user", query)]
    s.set_attribute("llm.attempts", 1)
    raw = await llm.complete(messages, json_output=True)
    try:
        return _validate(raw)
    except ValidationError as first:
        logger.warning("llm returned invalid JSON, retrying once: %s", first)
        messages += [
            Message("assistant", raw),
            Message(
                "user",
                f"That is not valid for the schema: {first.errors(include_url=False)}. "
                "Return only the corrected JSON object.",
            ),
        ]
    s.set_attribute("llm.attempts", 2)
    raw = await llm.complete(messages, json_output=True)
    try:
        return _validate(raw)
    except ValidationError as second:
        raise LLMError(
            f"{llm.provider} returned invalid JSON twice: {second.errors(include_url=False)}",
            "llm_bad_response",
            502,
        ) from second
