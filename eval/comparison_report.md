# Qdrant vs pgvector

Згенеровано `make compare` (`eval/compare_backends.py`). Не редагувати вручну.

Умови: 50000 векторів × 384, cosine; HNSW в обох `m=16`, `ef_construct=64`, пошук `ef=100`; Qdrant: 50000 точок, indexed_vectors_count=51000 (лічильник Qdrant: менше за точки — малий сегмент шукається перебором, більше — після перезапису ще не прибрані старі версії); 20 запитів × 20 прогонів після 3 прогрівних; aarch64, Linux (Docker), seed 0. Селективність фільтрів: «no filters» проходить 100.0% продуктів; «vegan, no nuts, sugar <= 5 g» проходить 5.3% продуктів.

| Сценарій | Backend | p50, мс | p95, мс | recall@10 (середній) | recall@10 (мін.) |
| --- | --- | ---: | ---: | ---: | ---: |
| no filters | qdrant (gRPC) | 6.43 | 8.10 | 0.980 | 0.8 |
| no filters | qdrant (HTTP) | 47.16 | 49.31 | 0.980 | 0.8 |
| no filters | pgvector | 3.36 | 5.21 | 0.985 | 0.8 |
| vegan, no nuts, sugar <= 5 g | qdrant (gRPC) | 2.96 | 4.19 | 1.000 | 1.0 |
| vegan, no nuts, sugar <= 5 g | qdrant (HTTP) | 43.80 | 45.08 | 1.000 | 1.0 |
| vegan, no nuts, sugar <= 5 g | pgvector | 9.73 | 34.33 | 0.925 | 0.7 |

recall@10 — частка точного top-10 (brute-force косинус у numpy по всіх векторах,
що проходять фільтр), яку повернула база. Латентність — виклик `store.search()`
з клієнта, разом з мережею.

**qdrant (HTTP)**: сталі ~40 мс поверх — це транспорт, а не пошук. Той самий
REST-запит через `httpx` (на ньому REST-клієнт Qdrant) займає ~46 мс, через
`http.client` — ~3 мс; найімовірніше, Nagle + delayed ACK на невеликих POST.
Тому API ходить у Qdrant по gRPC.

## Recall@10 по запитах

| Сценарій | Запит | qdrant (gRPC) | qdrant (HTTP) | pgvector |
| --- | --- | ---: | ---: | ---: |
| no filters | product 4101080050186 | 1.0 | 1.0 | 1.0 |
| no filters | product 9381840021341 | 1.0 | 1.0 | 1.0 |
| no filters | product 5000119095282 | 1.0 | 1.0 | 1.0 |
| no filters | product 00523035 | 0.9 | 0.9 | 1.0 |
| no filters | product 3502110005373 | 1.0 | 1.0 | 1.0 |
| no filters | product 6798381876591 | 1.0 | 1.0 | 1.0 |
| no filters | product 5704383009159 | 1.0 | 1.0 | 1.0 |
| no filters | product 4337256407670 | 1.0 | 1.0 | 1.0 |
| no filters | product 3760019012001 | 0.9 | 0.9 | 1.0 |
| no filters | product 54697485 | 0.8 | 0.8 | 0.8 |
| no filters | text 'chocolate biscuits' | 1.0 | 1.0 | 1.0 |
| no filters | text 'sparkling mineral water' | 1.0 | 1.0 | 1.0 |
| no filters | text 'greek yogurt' | 1.0 | 1.0 | 1.0 |
| no filters | text 'breakfast cereals' | 1.0 | 1.0 | 1.0 |
| no filters | text 'hazelnut chocolate spread' | 1.0 | 1.0 | 1.0 |
| no filters | text 'smoked salmon' | 1.0 | 1.0 | 0.9 |
| no filters | text 'tomato pasta sauce' | 1.0 | 1.0 | 1.0 |
| no filters | text 'dark chocolate' | 1.0 | 1.0 | 1.0 |
| no filters | text 'orange juice' | 1.0 | 1.0 | 1.0 |
| no filters | text 'potato chips crisps' | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | product 4101080050186 | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | product 9381840021341 | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | product 5000119095282 | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | product 00523035 | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | product 3502110005373 | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | product 6798381876591 | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | product 5704383009159 | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | product 4337256407670 | 1.0 | 1.0 | 0.7 |
| vegan, no nuts, sugar <= 5 g | product 3760019012001 | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | product 54697485 | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | text 'chocolate biscuits' | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | text 'sparkling mineral water' | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | text 'greek yogurt' | 1.0 | 1.0 | 0.7 |
| vegan, no nuts, sugar <= 5 g | text 'breakfast cereals' | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | text 'hazelnut chocolate spread' | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | text 'smoked salmon' | 1.0 | 1.0 | 0.8 |
| vegan, no nuts, sugar <= 5 g | text 'tomato pasta sauce' | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | text 'dark chocolate' | 1.0 | 1.0 | 0.9 |
| vegan, no nuts, sugar <= 5 g | text 'orange juice' | 1.0 | 1.0 | 1.0 |
| vegan, no nuts, sugar <= 5 g | text 'potato chips crisps' | 1.0 | 1.0 | 1.0 |
