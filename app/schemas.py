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
    Unknown values are never a match: no Nutri-Score fails `nutriscore_max`, vegan status 'maybe' or
    'unknown' fails `vegan_only`."""

    exclude_allergens: list[str] = Field(
        default=[], description="Drops products listing these in allergens OR traces"
    )
    vegan_only: bool = False
    nutriscore_max: NutriScore | None = Field(
        default=None, description="Worst grade allowed: 'c' keeps a, b, c"
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
