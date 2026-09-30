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
make up                # postgres (pgvector), qdrant, redis, api
curl localhost:8000/health
```

Якщо порти зайняті — змініть `API_PORT`, `POSTGRES_HOST_PORT`, `QDRANT_HOST_PORT`, `REDIS_HOST_PORT` у `.env`.

> Інжест вибірки — KAN-55/KAN-56; розгортання на kind — KAN-63.

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

## Приклади запитів

> KAN-59/KAN-60: пошук за кодом продукту, текстовий запит (у т.ч. «веганське печиво без горіхів з
> низьким вмістом цукру»).

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
app/            FastAPI (main.py, config.py, health.py)
app/api/        роутери пошуку
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
- **Орієнтири для цукру у fixture:** «низький» ≤ 5 г/100 г (claim ЄС, Reg. (EC) 1924/2006), «високий»
  > 22.5 г/100 г (UK FSA). У вибірці 50k лише 3 веганських печива без горіхів з цукром ≤ 5 г, і 54 веганських
  печива загалом. Остаточний поріг «низький цукор» для пошуку фіксується в KAN-59/KAN-60.
  [scripts/build_fixture.py](scripts/build_fixture.py)
