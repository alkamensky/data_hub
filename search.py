from __future__ import annotations

from pathlib import Path
import uuid
#тут конечно... всё надо потом перепрочитать и переписать
import openai
import psycopg
import streamlit as st



YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")
YANDEX_EMBED_QUERY_MODEL = f"emb://{YANDEX_CLOUD_FOLDER}/text-search-query/latest"

POSTGRES_DSN = (

)

client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
    project=YANDEX_CLOUD_FOLDER,
)



def make_query_embedding(text: str) -> list[float]:
    """Векторизация запроса для RAG"""
    if not text or not text.strip():
        raise ValueError("Пустой поисковый запрос")

    response = client.embeddings.create(
        model=YANDEX_EMBED_QUERY_MODEL,
        input=text,
        encoding_format="float",
    )

    vector = response.data[0].embedding

    if not vector:
        raise ValueError("Embeddings API вернул пустой вектор для запроса")

    return vector


def vector_to_pg_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vector) + "]"



def search_documents(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = make_query_embedding(query)
    query_vector = vector_to_pg_literal(query_embedding)

    sql = """
    SELECT
        id,
        filename,
        file_path,
        file_date,
        description_text,
        1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    with psycopg.connect(POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector, top_k))
            rows = cur.fetchall()

    results = []
    for row in rows:
        doc_id, filename, file_path, file_date, description_text, similarity = row
        results.append(
            {
                "id": str(doc_id),
                "filename": filename,
                "file_path": file_path,
                "file_date": file_date,
                "description_text": description_text,
                "similarity": float(similarity) if similarity is not None else 0.0,
            }
        )

    return results



def main() -> None:
    st.set_page_config(page_title="Умный поиск файлов", layout="wide")

    st.title("Умный поиск по файловому хранилищу")
    st.write("Введите естественный запрос, чтобы найти релевантные файлы из базы.")

    query = st.text_input(
        "Поисковый запрос",
        placeholder="Например: файл с расчётами добычи по участкам за 2023–2027 годы",
    )

    top_k = st.slider("Количество результатов", min_value=1, max_value=20, value=5)

    if st.button("Найти"):
        if not query.strip():
            st.warning("Введите текст запроса.")
            return

        with st.spinner("Ищу релевантные файлы..."):
            try:
                results = search_documents(query=query, top_k=top_k)
            except Exception as exc:
                st.error(f"Ошибка поиска: {exc}")
                return

        if not results:
            st.info("Ничего не найдено.")
            return

        st.success(f"Найдено результатов: {len(results)}")

        for index, item in enumerate(results, start=1):
            with st.container():
                st.subheader(f"{index}. {item['filename']}")
                st.write(f"**Путь:** `{item['file_path']}`")
                st.write(f"**Дата файла:** {item['file_date']}")
                st.write(f"**Похожесть:** {item['similarity']:.4f}")
                st.write("**Описание:**")
                st.write(item["description_text"])
                st.divider()


if __name__ == "__main__":
    main()