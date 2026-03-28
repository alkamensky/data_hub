from __future__ import annotations
#проблемы на 28.03.2026
#поменять порядок, чтобы файл не удалялся до добавления вектора в бд
#добавить индепотентность
#проработать дубли, так как файлы могут называться идентично, иметь одну дату, но 1 будет обновлённой версией второго или хуже актуализацией за год
#отработать папки и подпапки (кстати и с дублями поможет)
import json
import logging
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import openai
import pandas as pd
import psycopg
from tqdm import tqdm


# Ключи и модели

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")
YANDEX_GPT_MODEL = f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt-lite/latest"
YANDEX_EMBED_DOC_MODEL = f"emb://{YANDEX_CLOUD_FOLDER}/text-search-doc/latest"

POSTGRES_DSN = ( #оставил базовый для курсовой, в работе надо будет настроить на env, постгрес в докере

)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "10"))
FILE_READY_WAIT_SECONDS = int(os.getenv("FILE_READY_WAIT_SECONDS", "2"))

PREVIEW_MAX_ROWS = int(os.getenv("PREVIEW_MAX_ROWS", "20"))
PREVIEW_MAX_COLS = int(os.getenv("PREVIEW_MAX_COLS", "20"))

BASE_DIR = Path(__file__).resolve().parent
INCOMING_DIR = Path(os.getenv("INCOMING_DIR", BASE_DIR / "incoming")).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage")).resolve()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs")).resolve()
LOG_FILE = LOG_DIR / "scanner_service.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()



def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("scanner_service")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logging()



client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
    project=YANDEX_CLOUD_FOLDER,
)




STOP_REQUESTED = False


def handle_shutdown(signum, frame) -> None:
    "Функция отключения скрипта если завис или висит без дела"
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logger.info("Получен сигнал остановки: %s. Завершаю сервис...", signum)


try:
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
except Exception:
    pass


@dataclass
class FileMetadata:
    filename: str
    file_path: str
    file_date: datetime | None
    file_size: int
    file_extension: str



def is_supported_file(file_path: Path) -> bool: #нужно будет дописать до текстовых файлов, либо даже мультимодал, если разрешат апи оставить
    "отсечка несуществующих файлов, временных файлов и файлов нетабличного типа"
    if not file_path.is_file():
        return False

    if file_path.name.startswith("~$"):
        return False

    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_file_ready(file_path: Path, wait_seconds: int = FILE_READY_WAIT_SECONDS) -> bool:
    "защита от битых и удалённых файлов (когда человек передумал загружать и резко удалил)"
    if not file_path.exists() or not file_path.is_file():
        return False

    try:
        size_before = file_path.stat().st_size
        time.sleep(wait_seconds)
        size_after = file_path.stat().st_size
    except OSError:
        return False

    return size_before == size_after


def scan_incoming(incoming_dir: str | Path) -> list[Path]:
    "проверка путей"
    incoming_path = Path(incoming_dir).expanduser().resolve()

    if not incoming_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {incoming_path}")

    if not incoming_path.is_dir():
        raise NotADirectoryError(f"Это не папка: {incoming_path}")

    found_files: list[Path] = []

    for file_path in incoming_path.rglob("*"):
        if not is_supported_file(file_path):
            continue

        if not is_file_ready(file_path):
            continue

        found_files.append(file_path)

    found_files.sort(key=lambda p: str(p).lower())
    return found_files


def extract_basic_metadata(file_path: str | Path) -> FileMetadata:
    "проверка путей файла и сбор метаданных"
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if not path.is_file():
        raise ValueError(f"Указанный путь не является файлом: {path}")

    stat = path.stat()

    return FileMetadata(
        filename=path.name,
        file_path=str(path),
        file_date=datetime.fromtimestamp(stat.st_mtime),
        file_size=stat.st_size,
        file_extension=path.suffix.lower(),
    )



def _read_csv_safely(file_path: Path) -> pd.DataFrame:
    "чтение csv разных кодировок"
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]

    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Не удалось прочитать CSV: {file_path}. Последняя ошибка: {last_error}"
    )


def _normalize_value(value) -> str:
    "очищение от NaN и длинных строк"
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if len(text) > 500:
        return text[:500] + "..."
    return text


def _dataframe_preview(
    df: pd.DataFrame,
    max_rows: int = PREVIEW_MAX_ROWS, #выбрал формат 20 на 20, не знаю, думаю можно и 100 на 100, просто ориентир на локалку был
    max_cols: int = PREVIEW_MAX_COLS,
) -> dict:
    "обрезка датафреймов для поступления нейронки"
    df_limited = df.iloc[:max_rows, :max_cols].copy()

    preview_rows = []
    for _, row in df_limited.iterrows():
        row_dict = {}
        for col, value in row.items():
            row_dict[str(col)] = _normalize_value(value)
        preview_rows.append(row_dict)

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [str(col) for col in df.columns[:max_cols]],
        "sample_rows": preview_rows,
    }


def _build_file_summary(file_path: Path) -> dict:
    "сборка данных о листах и файле"
    ext = file_path.suffix.lower()

    if ext == ".csv":
        df = _read_csv_safely(file_path)

        return {
            "filename": file_path.name,
            "file_type": ext,
            "sheet_count": 1,
            "sheet_names": ["CSV"],
            "sheets": [
                {
                    "sheet_name": "CSV",
                    **_dataframe_preview(df),
                }
            ],
        }

    if ext in {".xlsx", ".xls"}:
        with pd.ExcelFile(file_path) as excel_file:
            sheet_names = excel_file.sheet_names

            sheets_summary = []
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    sheets_summary.append(
                        {
                            "sheet_name": sheet_name,
                            **_dataframe_preview(df),
                        }
                    )
                except Exception as exc:
                    sheets_summary.append(
                        {
                            "sheet_name": sheet_name,
                            "error": f"Не удалось прочитать лист: {exc}",
                        }
                    )

        return {
            "filename": file_path.name,
            "file_type": ext,
            "sheet_count": len(sheet_names),
            "sheet_names": sheet_names,
            "sheets": sheets_summary,
        }

    raise ValueError(f"Неподдерживаемый формат файла: {file_path}")


def generate_file_description(file_path: str | Path) -> str: #вот тут меня пробило с цензуры яндекса "Я не могу обсуждать такое" (я скинул файл с налогами)
    "подаём нейронке кусок датафрейма и генерируем описание"
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    summary = _build_file_summary(path)

    instructions = (
        "Ты анализируешь табличные файлы для системы поиска документов. "
        "Нужно составить одно общее краткое описание файла целиком на русском языке. "
        "Не перечисляй всё подряд сухим списком. "
        "Опиши, что это за файл, какая у него вероятная тематика, "
        "какие данные в нём содержатся и для чего он может быть полезен при поиске. "
        "Если в файле есть даты, периоды, показатели, отчёты, справочники, перечни, "
        "финансовые, технические или организационные данные — отрази это. "
        "Пиши 4-8 предложений. "
        "Не выдумывай то, чего нет в сводке."
    )

    response = client.responses.create(
        model=YANDEX_GPT_MODEL,
        temperature=0.2,
        instructions=instructions,
        input=json.dumps(summary, ensure_ascii=False, indent=2),
        max_output_tokens=600,
    )

    description = response.output_text.strip()

    if not description:
        raise ValueError("LLM вернула пустое описание файла")

    return description



def build_search_text(
    filename: str,
    file_path: str,
    file_date: datetime | None,
    description_text: str,
) -> str:
    parts = [
        f"Имя файла: {filename}",
        f"Путь к файлу: {file_path}",
    ]

    if file_date is not None:
        parts.append(f"Дата файла: {file_date.isoformat(sep=' ', timespec='seconds')}")

    parts.append(f"Описание файла: {description_text}")

    return "\n".join(parts)


def make_embedding(text: str) -> list[float]:
    "переводим описание в вектор"
    if not text or not text.strip():
        raise ValueError("Нельзя построить embedding для пустого текста")

    response = client.embeddings.create(
        model=YANDEX_EMBED_DOC_MODEL,
        input=text,
        encoding_format="float",
    )

    vector = response.data[0].embedding

    if not vector:
        raise ValueError("Embeddings API вернул пустой вектор")

    return vector


# =========================
# DATABASE
# =========================

def _vector_to_pg_literal(vector: list[float]) -> str:
    "форматируем под постгрес"
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def save_document(
    filename: str,
    file_path: str,
    file_date: datetime | None,
    description_text: str,
    embedding: list[float],
) -> None:
    "оформление и интеграция в бд"
    document_id = str(uuid.uuid4())
    embedding_literal = _vector_to_pg_literal(embedding)

    sql = """
    INSERT INTO documents (
        id,
        filename,
        file_path,
        file_date,
        description_text,
        embedding
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        %s::vector
    )
    """

    with psycopg.connect(POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    document_id,
                    filename,
                    file_path,
                    file_date,
                    description_text,
                    embedding_literal,
                ),
            )
        conn.commit()


def ensure_documents_table() -> None:
    "создание таблицы в бд кодом"
    sql_extension = "CREATE EXTENSION IF NOT EXISTS vector;"

    sql_table = """
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_date TIMESTAMP NULL,
        description_text TEXT NOT NULL,
        embedding VECTOR(256) NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """

    with psycopg.connect(POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_extension)
            cur.execute(sql_table)
        conn.commit()


# def print_documents_table(limit: int = 20) -> None:
#     sql = """
#     SELECT
#         id,
#         filename,
#         file_path,
#         file_date,
#         LEFT(description_text, 300) AS description_preview,
#         created_at
#     FROM documents
#     ORDER BY created_at DESC
#     LIMIT %s
#     """
#
#     with psycopg.connect(POSTGRES_DSN) as conn:
#         with conn.cursor() as cur:
#             cur.execute(sql, (limit,))
#             rows = cur.fetchall()
#
#     logger.info("=" * 120)
#     logger.info("Содержимое таблицы documents:")
#     logger.info("=" * 120)
#
#     if not rows:
#         logger.info("Таблица пустая")
#         return
#
#     for index, row in enumerate(rows, start=1):
#         doc_id, filename, file_path, file_date, description_preview, created_at = row
#
#         logger.info("Запись #%s", index)
#         logger.info("  id                  = %s", doc_id)
#         logger.info("  filename            = %s", filename)
#         logger.info("  file_path           = %s", file_path)
#         logger.info("  file_date           = %s", file_date)
#         logger.info("  description_preview = %s", description_preview)
#         logger.info("  created_at          = %s", created_at)



def _make_unique_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_to_storage(
    file_path: str | Path,
    incoming_dir: str | Path,
    storage_dir: str | Path,
    retries: int = 5,
    delay_seconds: float = 1.0,
) -> str:
    "перенос из папки income в storage псоле обработки и операций с бд"
    source = Path(file_path).expanduser().resolve()
    incoming_root = Path(incoming_dir).expanduser().resolve()
    storage_root = Path(storage_dir).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Файл не найден: {source}")

    if not source.is_file():
        raise ValueError(f"Путь не является файлом: {source}")

    try:
        relative_path = source.relative_to(incoming_root)
    except ValueError as exc:
        raise ValueError(
            f"Файл {source} не находится внутри incoming {incoming_root}"
        ) from exc

    target = storage_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _make_unique_path(target)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            shutil.move(str(source), str(target))
            return str(target)
        except PermissionError as exc:
            last_error = exc
            logger.warning(
                "Файл временно занят другим процессом. Попытка перемещения %s/%s: %s",
                attempt,
                retries,
                source,
            )
            time.sleep(delay_seconds)

    raise PermissionError(
        f"Не удалось переместить файл после {retries} попыток: {source}. "
        f"Последняя ошибка: {last_error}"
    )



FILE_STAGE_TOTAL = 6 #для прогресс бара 6 этапов обработки файла


def create_file_progress(file_path: str | Path) -> tqdm:
    filename = Path(file_path).name
    return tqdm(
        total=FILE_STAGE_TOTAL,
        desc=f"{filename}",
        unit="stage",
        dynamic_ncols=True,
        leave=False,
    )


def advance_stage(progress: tqdm, stage_text: str) -> None:
    progress.set_postfix_str(stage_text)
    progress.update(1)



def process_one_file(
    file_path: str | Path,
    incoming_dir: str | Path,
    storage_dir: str | Path,
) -> None:
    progress = create_file_progress(file_path)

    try:
        progress.set_postfix_str("чтение метаданных")
        metadata = extract_basic_metadata(file_path)
        logger.info("Метаданные считаны: %s", metadata.filename)
        advance_stage(progress, "метаданные готовы")

        progress.set_postfix_str("генерация описания")
        description = generate_file_description(file_path)
        logger.info("Описание сгенерировано: %s", metadata.filename)
        advance_stage(progress, "описание готово")

        progress.set_postfix_str("перенос в storage")
        time.sleep(1)
        stored_path = move_to_storage(file_path, incoming_dir, storage_dir)
        logger.info("Файл перенесён в storage: %s", stored_path)
        advance_stage(progress, "файл перенесён")

        progress.set_postfix_str("сбор search text")
        search_text = build_search_text(
            filename=metadata.filename,
            file_path=stored_path,
            file_date=metadata.file_date,
            description_text=description,
        )
        advance_stage(progress, "search text готов")

        progress.set_postfix_str("построение embedding")
        embedding = make_embedding(search_text)
        logger.info("Embedding построен: размер %s", len(embedding))
        advance_stage(progress, "embedding готов")

        progress.set_postfix_str("сохранение в БД")
        save_document(
            filename=metadata.filename,
            file_path=stored_path,
            file_date=metadata.file_date,
            description_text=description,
            embedding=embedding,
        )
        logger.info("Запись сохранена в БД: %s", metadata.filename)
        advance_stage(progress, "готово")

    finally:
        progress.close()


def process_new_files(
    incoming_dir: str | Path,
    storage_dir: str | Path,
) -> None:
    files = scan_incoming(incoming_dir)
    logger.info("Найдено файлов для обработки: %s", len(files))

    if not files:
        return

    success_count = 0
    error_count = 0

    for file_path in files:
        if STOP_REQUESTED:
            logger.info("Остановка запрошена. Прерываю текущую обработку пачки.")
            break

        try:
            logger.info("Начинаю обработку: %s", file_path)
            process_one_file(file_path, incoming_dir, storage_dir)
            success_count += 1
        except Exception as exc:
            error_count += 1
            logger.exception("Ошибка при обработке %s: %s", file_path, exc)

    logger.info(
        "Цикл обработки завершён. Успешно: %s, ошибок: %s",
        success_count,
        error_count,
    )



def validate_startup() -> None:
    if not YANDEX_CLOUD_FOLDER:
        raise ValueError("Не задана переменная окружения YANDEX_CLOUD_FOLDER")

    if not YANDEX_CLOUD_API_KEY:
        raise ValueError("Не задана переменная окружения YANDEX_CLOUD_API_KEY")

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")


if __name__ == "__main__":
    try:
        validate_startup()

        logger.info(
            "Сервис запущен. Сканирование папки incoming каждые %s секунд...",
            SCAN_INTERVAL_SECONDS,
        )
        logger.info("Incoming folder: %s", INCOMING_DIR)
        logger.info("Storage folder: %s", STORAGE_DIR)

        while not STOP_REQUESTED:
            cycle_started = time.time()

            try:
                logger.info("=" * 100)
                logger.info("Новый цикл сканирования...")
                logger.info("=" * 100)

                ensure_documents_table()
                process_new_files(
                    incoming_dir=INCOMING_DIR,
                    storage_dir=STORAGE_DIR,
                )

            except Exception as exc:
                logger.exception("Общая ошибка выполнения: %s", exc)

            elapsed = time.time() - cycle_started
            sleep_seconds = max(1, SCAN_INTERVAL_SECONDS - int(elapsed))

            if STOP_REQUESTED:
                break

            time.sleep(sleep_seconds)

        logger.info("Сервис остановлен.")

    except Exception as exc:
        logger.exception("Критическая ошибка запуска: %s", exc)
        sys.exit(1)