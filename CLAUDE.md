# Similar Products — правила роботи для Claude Code

Сервіс за кодом продукту або текстовим описом («веганське печиво без горіхів з низьким вмістом цукру»)
знаходить схожі продукти з Open Food Facts і фільтрує їх за алергенами, Nutri-Score і веганським статусом.
Епік KAN-53, задачі KAN-54…KAN-64.

## Стек

Python 3.12 (uv), FastAPI, PostgreSQL 16 + pgvector (psycopg 3; SQL-міграції `migrations/*.sql`,
ідемпотентні, застосовуються на старті), Qdrant (self-hosted), Redis + RQ (черга
інжесту), `sentence-transformers` (`all-MiniLM-L6-v2`, CPU), pyarrow + `HfFileSystem` (стрім row groups),
OpenTelemetry, pydantic v2, pytest, ruff. Docker / docker compose, kind. LLM: Gemini (за замовчуванням)
або Groq — лише безкоштовні тарифи без картки.

## Архітектура (зафіксована)

```
HF parquet (stream, ≥50k) → [продюсер] → Redis/RQ → [воркери: нормалізація → Postgres,
                                                     ембединг → Qdrant і pgvector паралельно]
GET  similar за кодом → вектор продукту → kNN (Qdrant | pgvector) + фільтри
POST текстовий запит  → LLM (1 виклик) → фільтри + смисловий текст → ембединг → kNN + фільтри
```

## Правила

- **Які продукти підходять, вирішує тільки детермінований код.** LLM лише парсить вільний текст у
  структуровані фільтри (`exclude_allergens`, `vegan_required`, `nutriscore_max`, `sugar_level`,
  смисловий текст); добір, фільтрація й ранжування — у коді над даними з БД.
- LLM викликається тільки через `LLMClient` (`app/llm/`); провайдер обирає `LLM_PROVIDER` (gemini | groq | fake).
- **Тести не ходять у мережу і не викликають живу LLM**: fixture-parquet (~200–500 рядків), `FakeLLM`,
  без завантаження моделей і датасетів з інтернету.
- Qdrant і pgvector індексують одну й ту саму вибірку тими самими ембедингами — інакше порівняння нечесне.
- Ніколи не питати користувача про уточнення в рантаймі. Невизначеність → явне припущення в README
  («Припущення»), з посиланням на файл, де воно закладене.
- Невідомий/кривий запит → структурована помилка 4xx, ніколи не 500.
- Секрети тільки в `.env` (не комітиться); нові змінні — в `.env.example`.
- Стиль: ruff (line-length 100), типи на публічних функціях, коментарі пояснюють «чому», а не «що».
- Один коміт на задачу (`KAN-NN: ...`); задача закрита лише після ручної перевірки критеріїв.

## Структура

- `app/main.py` — FastAPI, `GET /health` (postgres, qdrant, redis); `app/config.py` — налаштування з `.env`
- `app/api/` — роутери; `app/ingest/` — стрім HF, черга, воркери; `app/vectorstores/` — Qdrant і pgvector
  за спільним інтерфейсом; `app/llm/` — `LLMClient` і провайдери
- `eval/` — порівняння Qdrant vs pgvector; `k8s/` — маніфести для kind; `tests/` — офлайн-тести

## Команди

- `make up` / `make down` — усі сервіси (docker compose), API на `localhost:${API_PORT:-8000}` (`/health`, `/docs`)
- `make test` — pytest без мережі й ключів; `make lint` — ruff
- `make sample` — ≥50k корисних рядків з HF → `data/sample_50k.parquet` (~130 МБ трафіку, ~1.5 хв);
  `make fixture` — `tests/fixtures/sample_products.parquet` з вибірки (тести перевіряють покриття `CASES`)
- `make ingest` — вибірка → RQ `ingest` (батчі по 500) → `worker` × 2 → upsert у `products` (ідемпотентно)
- Інтеграційні тести з Postgres працюють у тимчасовій схемі (`tests/conftest.py: pg_schema`) і
  пропускаються, якщо БД не піднята
- Нова міграція — новий файл `migrations/NNN_*.sql`, ідемпотентний (`IF NOT EXISTS`); застосовані не правити.
  Після зміни міграцій — `make up` (вони запікаються в образ)
