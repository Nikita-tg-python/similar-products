import numpy as np
import pytest

from app import db
from app.embeddings import EMBEDDING_DIM, FakeEmbedder, embedding_text, get_embedder
from app.ingest.source import Product, iter_local_rows, normalize
from app.vectorstores.pgvector_store import PgvectorStore
from app.vectorstores.qdrant_store import point_id
from scripts.build_fixture import FIXTURE
from tests.test_ingest import ROOT


@pytest.fixture(scope="module")
def products():
    return [normalize(r) for r in iter_local_rows(ROOT / FIXTURE)]


@pytest.fixture(scope="module")
def vectors(products):
    return FakeEmbedder().encode([embedding_text(p) for p in products])


@pytest.fixture
def pgvector_store(pg_schema, products):
    db.apply_migrations(pg_schema)
    db.upsert_products(pg_schema, products)
    return PgvectorStore(pg_schema)


@pytest.fixture(params=["qdrant", "pgvector"])
def store(request):
    return request.getfixturevalue(f"{request.param}_store")


def test_embedding_text_is_name_categories_ingredients():
    p = Product(
        code="1",
        product_name="Choco cookies",
        categories_tags=["en:biscuits", "fr:biscuits-au-chocolat"],
        ingredients_text="flour, sugar",
    )
    assert embedding_text(p) == "Choco cookies. biscuits, biscuits au chocolat. flour, sugar"
    assert embedding_text(Product(code="2", product_name="Water")) == "Water"


def test_tests_use_the_fake_embedder():
    assert isinstance(get_embedder(), FakeEmbedder)


def test_fake_embeddings_are_normalized_and_deterministic():
    a = FakeEmbedder().encode(["vegan oat cookies", "vegan oat cookies", "", "sparkling water"])
    assert a.shape == (4, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a[0], a[1])
    assert a[0] @ a[1] > a[0] @ a[3]


def test_point_ids_are_stable_uuids():
    assert point_id("0000101209159") == point_id("0000101209159")
    assert point_id("0000101209159") != point_id("101209159")  # leading zeros matter


def test_store_contract(store, products, vectors):
    store.upsert(products, vectors)
    store.upsert(products, vectors)  # idempotent
    assert store.count() == len(products)

    code = products[0].code
    assert np.allclose(store.get_vector(code), vectors[0], atol=1e-6)
    assert store.get_vector("no-such-code") is None

    hits = store.search(vectors[0], limit=5)
    assert hits[0].code == code and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert code not in {h.code for h in store.search(vectors[0], limit=5, exclude_code=code)}


def test_qdrant_collection_is_created_once(qdrant_store):
    assert qdrant_store.ensure_collection() is False  # the fixture already created it
    info = qdrant_store.client.get_collection(qdrant_store.collection)
    assert info.config.params.vectors.size == EMBEDDING_DIM
    assert info.config.params.vectors.distance.value == "Cosine"


def test_both_stores_agree_on_neighbours(qdrant_store, pgvector_store, products, vectors):
    """Same vectors in, (near-)same neighbours out: the premise of the comparison in KAN-61."""
    for s in (qdrant_store, pgvector_store):
        s.upsert(products, vectors)
    for i in range(0, len(products), 40):
        q = [
            h.code for h in qdrant_store.search(vectors[i], limit=10, exclude_code=products[i].code)
        ]
        p = [
            h.code
            for h in pgvector_store.search(vectors[i], limit=10, exclude_code=products[i].code)
        ]
        assert len(set(q) & set(p)) >= 8, (q, p)
