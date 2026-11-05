# RAG Telegram Bot

Telegram-бот, що відповідає на запитання за власною базою знань (Retrieval-Augmented Generation).

**Стек:** Aiogram 3 · PostgreSQL 16 + pgvector · SQLAlchemy 2.0 (async) · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · Docker Compose.

## Архітектура

```
Handler  ──►  Service  ──►  Repository  ──►  Database
handlers/     services/     repositories/    db.py / models.py
              clients/ (OpenAI)
```

| Шар | Файл | Відповідальність |
|---|---|---|
| Config | `config.py` | `pydantic-settings`, усі секрети лише з `.env` |
| DB | `db.py`, `models.py` | `AsyncEngine`, `async_sessionmaker`, `CREATE EXTENSION vector`, моделі `User` та `KnowledgeEmbedding` (`Vector(1536)`) |
| Repositories | `repositories/users.py` | `UserRepository.upsert_user` (`INSERT … ON CONFLICT DO UPDATE`) |
| | `repositories/knowledge.py` | `KnowledgeRepository.similarity_search` — оператор `<=>` (cosine distance) + відсікання за `RAG_THRESHOLD` |
| Services | `services/embeddings.py` | `EmbeddingService` — синхронний OpenAI SDK у `asyncio.to_thread` |
| | `services/rag.py` | `RagService` — чанкінг, пошук Top-3, побудова промпта, threshold-guard |
| Clients | `clients/openai_client.py` | `OpenAIClient` — `AsyncOpenAI`, таймаути, ретраї, обробка помилок |
| Handlers | `handlers/start.py`, `handlers/text.py` | `/start` та текстові запитання |
| Entry | `bot.py` | одноразове створення сервісів і передача в хендлери через DI Aiogram (`Dispatcher(**workflow_data)`) |
| Ingest | `init_db.py` | завантаження `knowledge_base.txt` → чанки → вектори → БД |

### Ключові правила
- **Немає сирого SQL** у хендлерах та `RagService` — лише репозиторії на SQLAlchemy.
- **Поріг релевантності:** якщо найближчий чанк має cosine distance > `RAG_THRESHOLD`, LLM **не викликається**, а користувач отримує:
  «На жаль, у базі знань немає інформації для відповіді на це запитання.»
  Відстань: `0` — ідентичні тексти, `2` — протилежні. Типові значення порогу — `0.4…0.6`.
- **Безпечні помилки:** усі винятки логуються через `logger.exception()`, користувач бачить лише нейтральне повідомлення.
- `init_db.py` оновлює **тільки** таблицю `knowledge_embeddings`; таблиця `users` ніколи не видаляється. Вектори генеруються до видалення старих даних, заміна — в одній транзакції.

## Запуск через Docker Compose

```bash
cp .env.example .env        # заповніть BOT_TOKEN, OPENAI_API_KEY, DB_PASSWORD
docker compose up -d db     # дочекайтеся статусу healthy
docker compose run --rm bot python init_db.py   # індексація knowledge_base.txt
docker compose up -d --build bot
docker compose logs -f bot
```

Після зміни `knowledge_base.txt` перебудуйте образ і повторіть індексацію:

```bash
docker compose build bot && docker compose run --rm bot python init_db.py
```

## Локальний запуск (без Docker для бота)

Потрібен PostgreSQL із pgvector. Найпростіше — підняти лише `db` і відкрити порт
(додайте `ports: ["5432:5432"]` до сервісу `db`), у `.env` вкажіть `DB_HOST=localhost`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python init_db.py               # або: python init_db.py path/to/file.txt
python bot.py
```

## Формат бази знань

`knowledge_base.txt` (UTF-8). Якщо файл містить маркери `Chunk N:` на початку рядка, вони
визначають межі чанків. Інакше текст ділиться за абзацами на вікна до 1500 символів з
перекриттям 200 символів. Задовгі розділи також доділюються.

## Змінні середовища

| Змінна | Обов'язкова | За замовчуванням |
|---|---|---|
| `BOT_TOKEN` | так | — |
| `OPENAI_API_KEY` | так | — |
| `DB_USER`, `DB_PASSWORD`, `DB_NAME` | так | — |
| `DB_HOST` / `DB_PORT` | ні | `localhost` / `5432` (у compose примусово `db`) |
| `RAG_THRESHOLD` | ні | `0.5` |
| `RAG_TOP_K` | ні | `3` |
| `OPENAI_CHAT_MODEL` | ні | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | ні | `text-embedding-3-small` |
| `OPENAI_TIMEOUT` | ні | `30` (секунд) |
| `LOG_LEVEL` | ні | `INFO` |
