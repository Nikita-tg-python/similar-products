"""API shapes and the search filters shared by the API, the vector stores and the query parser."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NutriScore = Literal["a", "b", "c", "d", "e"]
Backend = Literal["qdrant", "pgvector"]

# The 14 EU allergens (Reg. (EU) 1169/2011, Annex II) as Open Food Facts tags.
ALLERGENS = {
    "gluten": "en:gluten",
    "crustaceans": "en:crustaceans",
    "eggs": "en:eggs",
    "fish": "en:fish",
    "peanuts": "en:peanuts",
    "soybeans": "en:soybeans",
    "milk": "en:milk",
    "nuts": "en:nuts",  # tree nuts; peanuts are a separate allergen, as in the regulation
    "celery": "en:celery",
    "mustard": "en:mustard",
    "sesame-seeds": "en:sesame-seeds",
    "sulphur-dioxide-and-sulphites": "en:sulphur-dioxide-and-sulphites",
    "lupin": "en:lupin",
    "molluscs": "en:molluscs",
}
ALLERGEN_ALIASES = {
    "nut": "nuts",
    "tree-nuts": "nuts",
    "peanut": "peanuts",
    "soy": "soybeans",
    "soya": "soybeans",
    "egg": "eggs",
    "dairy": "milk",
    "lactose": "milk",
    "wheat": "gluten",
    "sesame": "sesame-seeds",
    "sulphites": "sulphur-dioxide-and-sulphites",
    "sulfites": "sulphur-dioxide-and-sulphites",
    "shellfish": "crustaceans",
}


def allergen_tag(name: str) -> str:
    """'nuts', 'Tree nuts', 'en:nuts' -> 'en:nuts'; anything else is a ValueError, so a typo is a
    422 rather than a filter that silently lets everything through."""
    key = name.strip().lower().removeprefix("en:").replace(" ", "-").replace("_", "-")
    key = ALLERGEN_ALIASES.get(key, key)
    if key not in ALLERGENS:
        raise ValueError(f"unknown allergen {name!r}; known: {', '.join(sorted(ALLERGENS))}")
    return ALLERGENS[key]


class SearchFilters(BaseModel):
    """Applied inside the database (SQL WHERE / Qdrant payload filter), never by post-filtering.
    Unknown values are never a match: no Nutri-Score fails `nutriscore_max`, vegan status 'maybe'
    or 'unknown' fails `vegan_only`, unknown sugar fails `sugars_max`, and with `exclude_allergens`
    a product whose ingredients are unknown fails too: in Open Food Facts empty allergen tags on
    such a product mean "nobody entered them", not "free of" (KAN-64: hummus "without sesame")."""

    exclude_allergens: list[str] = Field(
        default=[],
        description="Drops products listing these in allergens OR traces, and products whose "
        "ingredients are unknown",
    )
    vegan_only: bool = False
    nutriscore_max: NutriScore | None = Field(
        default=None, description="Worst grade allowed: 'c' keeps a, b, c"
    )
    sugars_max: float | None = Field(
        default=None, ge=0, le=100, description="Max sugars, g/100 g; unknown sugar fails"
    )

    @field_validator("exclude_allergens", mode="before")
    @classmethod
    def _tags(cls, value: list[str] | str) -> list[str]:
        items = value.split(",") if isinstance(value, str) else value
        return sorted({allergen_tag(v) for v in items for v in v.split(",") if v.strip()})

    @property
    def nutriscore_grades(self) -> list[str] | None:
        if self.nutriscore_max is None:
            return None
        return [g for g in "abcde" if g <= self.nutriscore_max]


# EU nutrition claim "low sugar": at most 5 g per 100 g (Reg. (EC) 1924/2006, Annex). The
# 2.5 g/100 ml for drinks is not applied: OFF does not reliably tell solids from liquids.
LOW_SUGAR_MAX_G = 5.0


class ParsedQuery(BaseModel):
    """What the LLM may say about a query, and nothing more: it never names products."""

    semantic_text: str = Field(
        min_length=1, max_length=200, description="English, the kind of product, no constraints"
    )
    exclude_allergens: list[str] = Field(default=[], description="Allergen names, as the LLM gave")
    vegan_required: bool | None = None
    nutriscore_max: NutriScore | None = None
    sugar_level: Literal["low", "any"] = "any"
    unsupported: list[str] = Field(
        default=[], description="Constraints the user stated that no filter can express"
    )

    def to_filters(self) -> tuple[SearchFilters, list[str]]:
        """Deterministic mapping to database filters, with notes on anything that could not be
        applied (an allergen outside the 14 tracked ones is reported, not silently dropped)."""
        notes: list[str] = []
        tags: list[str] = []
        for name in self.exclude_allergens:
            try:
                tags.append(allergen_tag(name))
            except ValueError:
                notes.append(
                    f"{name!r} is not one of the 14 EU allergens OFF tags reliably; not filtered"
                )
        notes += [f"not filtered (no such filter): {c!r}" for c in self.unsupported]
        sugars_max = None
        if self.sugar_level == "low":
            sugars_max = LOW_SUGAR_MAX_G
            notes.append(
                f"low sugar = at most {LOW_SUGAR_MAX_G:g} g/100 g (EU claim); "
                "products with unknown sugar are excluded"
            )
        filters = SearchFilters(
            exclude_allergens=tags,
            vegan_only=bool(self.vegan_required),
            nutriscore_max=self.nutriscore_max,
            sugars_max=sugars_max,
        )
        return filters, notes


class ProductOut(BaseModel):
    code: str
    product_name: str
    brands: str | None
    vegan: str
    nutriscore_grade: str | None
    sugars_100g: float | None
    allergens_tags: list[str]
    traces_tags: list[str]
    categories_tags: list[str]


class SimilarProduct(ProductOut):
    score: float = Field(description="Cosine similarity to the query, higher is closer")


class SimilarResponse(BaseModel):
    product: ProductOut
    backend: Backend
    filters_applied: SearchFilters
    took_ms: float = Field(description="Vector search only, in the chosen backend")
    results: list[SimilarProduct]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    backend: Backend = "qdrant"
    limit: int = Field(default=10, ge=1, le=100)


class SearchResponse(BaseModel):
    query: str
    parsed: ParsedQuery = Field(description="The LLM's reading of the query (its only output)")
    filters_applied: SearchFilters = Field(description="What the database actually filtered on")
    notes: list[str]
    backend: Backend
    took_ms: float = Field(description="Vector search only, in the chosen backend")
    results: list[SimilarProduct]
