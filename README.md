# Similar Products

Пошук схожих продуктів з [Open Food Facts](https://huggingface.co/datasets/openfoodfacts/product-database)
за кодом продукту або текстовим описом («веганське печиво без горіхів з низьким вмістом цукру») з
фільтрами за алергенами, Nutri-Score і веганським статусом. Вектори зберігаються паралельно в Qdrant і
pgvector, щоб чесно порівняти обидві бази на однакових даних.

## Архітектура

> Заповнюється по ходу KAN-55…KAN-62: mermaid-схема інжесту (HF stream → Redis/RQ → воркери → Postgres +
> Qdrant + pgvector) і пошуку (код / текст → LLM-фільтри → kNN + детерміновані фільтри).

## Запуск

Потрібен лише Docker.

```bash
cp .env.example .env   # вписати GEMINI_API_KEY або GROQ_API_KEY
make up                # postgres (pgvector), qdrant, redis, api, 2 × worker
curl localhost:8000/health
make sample            # ≥50k продуктів з HF → data/sample_50k.parquet (~1.5 хв)
make ingest            # вибірка → Redis/RQ → воркери → Postgres + ембединги → pgvector і Qdrant (~20 хв)
make check-stores      # однакові вектори в обох базах, перетин сусідів
```

Якщо порти зайняті — змініть `API_PORT`, `POSTGRES_HOST_PORT`, `QDRANT_HOST_PORT`, `REDIS_HOST_PORT` у `.env`.

> Розгортання на kind — KAN-63.

### Інжест

Продюсер ([app/ingest/producer.py](app/ingest/producer.py)) застосовує міграції, кладе вибірку в RQ-чергу
`ingest` батчами по 500 рядків і стежить за прогресом (скільки джобів зроблено, скільки в черзі, скільки
рядків записано). Воркери (`rq worker ingest`, сервіс `worker`, 2 репліки;
[app/ingest/worker.py](app/ingest/worker.py)) нормалізують сирі рядки й роблять upsert у `products` за `code`.
Потім рахують ембединги батчем ([app/embeddings.py](app/embeddings.py), `all-MiniLM-L6-v2`, CPU) і пишуть
**ті самі вектори паралельно** в pgvector (`products.embedding`) і в Qdrant (колекція `products`, cosine,
HNSW `m=16`, `ef_construct=64` — як у pgvector; payload — поля-фільтри з індексами).
[app/vectorstores/](app/vectorstores/)

- **Ідемпотентність:** `INSERT ... ON CONFLICT (code) DO UPDATE`. Повторний `make ingest` лишає 50 000
  рядків, лише оновлює `updated_at`.
- **Помилки:** кривий рядок не валить батч, а рахується як `invalid`. Помилка БД валить батч, і RQ
  повторює його (3 спроби: через 5, 30 і 60 с). Після цього батч іде у failed registry, а продюсер
  виходить з кодом 1.
- **Убитий воркер не блокує чергу.** Під час розробки перезапуск воркерів посеред батча лишив у Postgres
  відкриту транзакцію від мертвого клієнта, і наступний джоб на тих самих рядках чекав на блокування.
  Тепер воркер має `stop_grace_period: 2m`, тож поточний батч встигає дописатися. Postgres має
  TCP keepalive 60/10/3, тож мертві з'єднання відрізаються за ~90 с. З'єднання мають
  `lock_timeout=60s`: джоб падає, і RQ його повторює.
- **Замір 2026-09-30:** без ембедингів 50 000 рядків, 100 джобів, 2 воркери — **13 с**. З ембедингами —
  ~20 хв. На батч з 500 рядків кодування займає ~20 с, запис у pgvector ~1.3 с, у Qdrant ~0.3 с (це
  видно в логах воркера, `batch done`). CPU-torch у Linux-VM Docker на Mac кодує ~25 текстів/с на
  воркер, а нативно на macOS — 177/с.

## Дані

`openfoodfacts/product-database` (HF, `food.parquet`: 7.86 ГБ, 4.77M рядків, 145 колонок, ~4.7k row groups
по ~1k рядків, відсортовано за штрихкодом). Файл ніколи не завантажується цілком:

```bash
make sample    # → data/sample_50k.parquet (не комітиться)
make fixture   # → tests/fixtures/sample_products.parquet (комітиться, для офлайн-тестів)
```

Замір 2026-09-30: 50 000 корисних рядків з 125 593 прочитаних за **92 с**, завантажено **130.5 МБ = 1.7 %**
файлу (з них ~104 МБ — футер parquet з метаданими, решта — лише 16 потрібних колонок по ~0.2 МБ на row
group). Читач — [app/ingest/source.py](app/ingest/source.py).

Реальна схема відрізняється від картки датасету: `product_name` і `ingredients_text` — списки
`{lang, text}`, `nutriments` — список `{name, 100g, ...}`, веганський статус — у `ingredients_analysis_tags`.
Нормалізований запис — `Product` у [app/ingest/source.py](app/ingest/source.py).

Fixture (326 рядків) покриває кожен випадок з `CASES` у [scripts/build_fixture.py](scripts/build_fixture.py):
vegan yes/no/maybe/unknown, Nutri-Score a–e і невідомий, цукор низький/високий/невідомий, алергени (горіхи,
арахіс, молоко, глютен, яйця, соя), лише сліди горіхів, печиво, у т.ч. приклад із задачі.

### Схема Postgres

`products` ([migrations/](migrations/)): поля-фільтри (`allergens_tags`, `traces_tags`, `categories_tags`,
`vegan`, `nutriscore_grade`, `sugars_100g`, …) і `embedding vector(384)`. Індекси:

- `products_embedding_hnsw` — HNSW, `vector_cosine_ops`, `m=16`, `ef_construction=64` (дефолти pgvector);
- GIN на `allergens_tags`, `traces_tags`, `categories_tags`.

`EXPLAIN ANALYZE` на копії 50 000 продуктів з випадковими векторами (pgvector 0.8.6):

| Запит | План | Час |
| --- | --- | --- |
| `ORDER BY embedding <=> $1 LIMIT 10` | `Index Scan using products_embedding_hnsw` | 1.7 мс |
| те саме + `vegan='yes' AND NOT allergens_tags && '{en:nuts,en:peanuts}' AND sugars_100g <= 5`, `hnsw.iterative_scan = relaxed_order` | `Index Scan using products_embedding_hnsw` + Filter, 10 з 10 рядків | 7.7 мс |
| `categories_tags && '{en:biscuits}'` | `Bitmap Index Scan on products_categories_gin` | 6.8 мс |

## Приклади запитів

### Схожі за кодом продукту

```bash
# 3700036906990 — «Pâte à tartiner 20% de noisettes»
curl "localhost:8000/products/3700036906990/similar?limit=5"                           # backend=qdrant за замовчуванням
curl "localhost:8000/products/3700036906990/similar?backend=pgvector&exclude_allergens=nuts"
curl "localhost:8000/products/3700036906990/similar?exclude_allergens=nuts,milk&vegan_only=true&nutriscore_max=c"
```

Параметри:
- `backend=qdrant|pgvector`;
- `limit` від 1 до 100;
- `exclude_allergens` повторюється або йде через кому. Приймає 14 алергенів ЄС і синоніми (`soy`,
  `peanut`, `dairy`, `sesame`…). Відкидає продукт, якщо алерген є в `allergens_tags` **або**
  `traces_tags`;
- `vegan_only`;
- `nutriscore_max`: найгірша допустима оцінка.

Фільтри застосовуються всередині бази: WHERE з iterative HNSW scan у pgvector, payload-filter у Qdrant.

Відповідь містить `product`, `backend`, `filters_applied` (нормалізовані: `["en:nuts"]`), `took_ms`
(лише векторний пошук) і `results` зі `score` та полями-фільтрами. Помилки мають одну форму
`{"error": {"code", "message", "details"?}}`: 404 `product_not_found`, 409 `not_indexed`,
422 `invalid_request`.

### Текстовий запит

> KAN-60: «веганське печиво без горіхів з низьким вмістом цукру».

## Порівняння Qdrant vs pgvector

> KAN-61: latency p50/p95 і recall@10 відносно brute-force косинуса на тій самій вибірці.

| База | p50, мс | p95, мс | recall@10 |
| --- | --- | --- | --- |
| Qdrant | — | — | — |
| pgvector | — | — | — |

## Трейсинг

> KAN-62: OpenTelemetry, консольний експортер за замовчуванням, опційно OTLP → Jaeger.

## Тести

```bash
make test   # без мережі й живої LLM
make lint
```

## Структура

```
app/            FastAPI (main.py, config.py, health.py), db.py (міграції, upsert)
migrations/     SQL-схема, ідемпотентна, застосовується на старті
app/api/        роутери пошуку (search.py: GET /products/{code}/similar)
app/schemas.py  фільтри пошуку й форми відповідей; app/errors.py — формат помилок
app/ingest/     стрім HF, черга Redis/RQ, воркери
app/vectorstores/  Qdrant і pgvector за спільним інтерфейсом
app/llm/        LLMClient і провайдери (gemini | groq | fake)
eval/           порівняння Qdrant vs pgvector
scripts/        build_sample.py (HF → data/), build_fixture.py (→ tests/fixtures/)
data/           локальна вибірка, не комітиться
k8s/            маніфести для kind
tests/          офлайн-тести
```

## Припущення

Кожне — з посиланням на файл, де воно закладене.

- **Читання без `datasets`.** `load_dataset(..., streaming=True)` за 20 хв не віддав навіть 5k рядків
  (тягне всі 145 колонок дрібними запитами). Тому pyarrow через `HfFileSystem` читає окремі row groups і
  лише потрібні колонки, паралельно у 8 потоків. Це теж стрімінг, без повного завантаження.
  [app/ingest/source.py](app/ingest/source.py)
- **Вибірка — випадкові row groups (seed 0), а не перші N.** Файл відсортований за штрихкодом, тож перші
  50k — один вузький діапазон кодів. Випадкові групи дають зріз усієї бази, і він відтворюваний.
  [app/ingest/source.py](app/ingest/source.py)
- **У вибірку йдуть лише корисні для пошуку рядки:** є штрихкод і назва, продукт не `obsolete`, і є хоча б
  категорії або склад. Під ці умови підпадає ~40 % рядків. [app/ingest/source.py](app/ingest/source.py) (`normalize`)
- **Мова тексту:** англійська, якщо є, інакше мова продукту, інакше `main`. Англійська назва є лише у
  ~37 % вибірки, найчастіші мови — fr 37 %, en 31 %, de 9 %. Це треба врахувати під час вибору моделі
  ембедингів (KAN-58). [app/ingest/source.py](app/ingest/source.py) (`pick_text`)
- **Невідоме — це `None`/`unknown`, а не «ні»:** Nutri-Score `unknown`/`not-applicable`, цукор поза 0–100 г,
  відсутній веганський тег. [app/ingest/source.py](app/ingest/source.py)
- **Батчі по 500 рядків, а не джоб на рядок.** RQ форкає процес на кожен джоб, тож 50k однорядкових
  джобів переважно форкалися б. Джоб несе сирі рядки, а нормалізація відбувається у воркері, щоб правила
  жили в одному місці. [app/ingest/producer.py](app/ingest/producer.py)
- **Міграції** (`migrations/*.sql`, ідемпотентні) застосовують API на старті і продюсер перед
  інжестом. Advisory lock не дає їм гонитися. [app/db.py](app/db.py)
- **Колонка `vegan`, а не `vegan_status`** (як у тексті KAN-57): назва вже була в `Product` з KAN-55,
  а значення ті самі (`yes`/`no`/`maybe`/`unknown`). [migrations/001_products.sql](migrations/001_products.sql)
- **Вектор і індекси — окрема міграція `002`**, а не правка `001`. `001` вже застосована на заповненій
  базі, і `CREATE TABLE IF NOT EXISTS` не додав би в неї колонку.
  [migrations/002_pgvector.sql](migrations/002_pgvector.sql)
- **Фільтрований kNN у pgvector — через `hnsw.iterative_scan = relaxed_order`.** Без нього HNSW віддає
  лише `ef_search` (40) кандидатів до фільтра, і рідкісний фільтр повертав би менше 10 результатів.
- **Модель ембедингів — `all-MiniLM-L6-v2`, як у задачі.** Порівнював на 5k продуктів, 5 запитів,
  P@10 за очікуваною категорією: англійські запити 0.66 проти 0.56 у
  `paraphrase-multilingual-MiniLM-L12-v2`, до того ж вона швидша. Українські запити погано тягнуть
  обидві (0.04 / 0.24), тому парсер запиту (KAN-60) передає в ембединг англійський текст. Продукти
  добре кодуються англійською навіть французькі, бо теги категорій OFF англійські. Модель задається
  в `EMBEDDING_MODEL`. [app/embeddings.py](app/embeddings.py)
- **Текст для ембедингу:** назва, потім категорії без мовного префікса («hazelnut spreads»), потім
  склад. Медіана — 48 токенів, 4 % текстів обрізаються на 256.
- **Id у Qdrant — UUIDv5 від `code`**: Qdrant приймає лише uint або UUID, а штрихкоди мають провідні
  нулі. `code` лежить у payload. [app/vectorstores/qdrant_store.py](app/vectorstores/qdrant_store.py)
- **Воркер — `SimpleWorker`** (без fork на кожен джоб): модель вантажиться раз на воркер, і torch не
  переживає fork. Модель завантажується з HF при першому запуску в volume `hf_cache`.
- **Обидві бази пишуться в одному джобі.** Якщо запис в одну з них падає, падає весь батч, і RQ його
  повторює. Усі записи — upsert, тож бази не розходяться назавжди.
- **Невідоме не проходить фільтр:** `vegan_only` пропускає лише `yes` (не `maybe` і не `unknown`),
  `nutriscore_max` відкидає продукти без оцінки. [app/schemas.py](app/schemas.py)
- **Горіхи = `en:nuts` (деревні), арахіс окремо (`en:peanuts`)**, як у регламенті ЄС 1169/2011.
  «Без горіхів» з арахісом — це `exclude_allergens=nuts,peanuts`. [app/schemas.py](app/schemas.py)
- **Фільтр алергенів довіряє тегам OFF, а вони неповні.** Із 2910 продуктів вибірки, у назві чи складі
  яких згадано фундук, мигдаль, волоський горіх, кеш'ю тощо, у 729 (25 %) немає `en:nuts` ні в
  алергенах, ні в слідах. `exclude_allergens=nuts` їх не відкидає (приклад: «Pâte à tartiner
  noisettes et cacao»). Це обмеження даних, а не фільтра.
- **Однаковий `ef` пошуку в обох базах** — `SEARCH_EF = 100` (pgvector `hnsw.ef_search`, Qdrant
  `hnsw_ef`). [app/vectorstores/base.py](app/vectorstores/base.py)
- **Орієнтири для цукру у fixture:** «низький» ≤ 5 г/100 г (claim ЄС, Reg. (EC) 1924/2006), «високий»
  > 22.5 г/100 г (UK FSA). У вибірці 50k лише 3 веганських печива без горіхів з цукром ≤ 5 г, і 54 веганських
  печива загалом. Остаточний поріг «низький цукор» для пошуку фіксується в KAN-59/KAN-60.
  [scripts/build_fixture.py](scripts/build_fixture.py)
