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
k8s/            маніфести для kind
tests/          офлайн-тести
```

## Припущення

> Кожне — з посиланням на файл, де воно закладене. Зокрема: поріг «низький цукор», яка вибірка з 4.83M
> береться (перші N зі стріму чи випадкова), модель ембедингів і текст для неї.
