import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer


class PostgresClient:

    nltk.download("stopwords", quiet=True)

    def __init__(self, pg_host: str, pg_user: str, pg_password: str, pg_db: str):
        self.pg_host = pg_host
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_db = pg_db
        self.stop_words = set(stopwords.words("english"))
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def _make_conn(self):
        try:
            conn = psycopg2.connect(
                host=self.pg_host,
                user=self.pg_user,
                password=self.pg_password,
                dbname=self.pg_db,
            )
            register_vector(conn)
            return conn
        except Exception as e:
            print(f"Error connecting to Postgres: {str(e)}")
            return None

    def _normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _remove_stopwords(self, text: str) -> list:
        return [
            word.lower() for word in text.split() if word.lower() not in self.stop_words
        ]

    def insert_content_embeddings(self, data: list):
        try:
            conn = self._make_conn()
            cursor = conn.cursor()
            for record in data:
                try:
                    norm_embedding = self._normalize_embedding(record["embedding"])
                    cursor.execute(
                        """
                                INSERT INTO content_embeddings(document_id, tags, clean_text, embedding)
                                    VALUES(%s, %s, %s, %s);
                            """,
                        (
                            record["document_id"],
                            record["tags"],
                            record["clean_text"],
                            norm_embedding.tolist(),
                        ),
                    )
                except Exception as e:
                    print(f"Error inserting record: {str(e)}")
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error connecting to Postgres: {str(e)}")

    def semantic_search(self, query: list, n_results: int = 5):
        # convert query embedidng to the proper format
        query_embedding = self.embedding_model.encode(query)
        norm_query_embedding = self._normalize_embedding(np.array(query_embedding))
        query_embedding_str = ", ".join(map(str, norm_query_embedding.tolist()))

        # build query
        search = f"SELECT uid, document_id, tags, clean_text, 1 - (embedding <=> '[{query_embedding_str}]') as similarity_score FROM content_embeddings ORDER BY embedding <=> '[{query_embedding_str}]' LIMIT {n_results};"

        # establish a connection to the DB
        conn = self._make_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(search)
            results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Error conducting semantic search: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def tag_search(
        self, query: str, n_results: int = 10, similarity_threshold: float = 0.3
    ):
        keywords = self._remove_stopwords(query)
        if not keywords:
            return []

        conn = self._make_conn()
        try:
            cursor = conn.cursor()
            results = []

            for keyword in keywords:
                query = f"""
                        SELECT 
                            uid,
                            document_id,
                            tags, 
                            clean_text, 
                            MAX(similarity(tag::text, '{keyword}')) as max_tag_similarity
                        FROM 
                            content_embeddings, 
                            unnest(tags) as tag
                        WHERE
                            similarity(tag::text, '{keyword}') > {similarity_threshold}
                        GROUP BY 
                            uid, document_id, tags, clean_text
                        ORDER BY 
                            max_tag_similarity DESC
                        LIMIT {n_results}
                        """
                cursor.execute(query)
                results.extend(cursor.fetchall())

            # dedupe
            seen = {}
            for row in results:
                uid = row[0]
                score = row[-1]
                if uid not in seen or score > seen[uid][-1]:
                    seen[uid] = row

            return list(seen.values())
        except Exception as e:
            print(f"Error during tag search: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = 0.7):
        semantic_results = self.semantic_search(query, n_results=5)
        tag_results = self.tag_search(query, n_results=5)

        # create some dicts for results
        semantic_dict = {
            row[0]: {
                "document_id": row[1],
                "tags": row[2],
                "clean_text": row[3],
                "semantic_score": row[4],
                "tag_score": 0.0,
            }
            for row in semantic_results
        }
        tag_dict = {
            row[0]: {
                "document_id": row[1],
                "tags": row[2],
                "clean_text": row[3],
                "semantic_score": 0.0,
                "tag_score": row[4] * 0.8,
            }
            for row in tag_results
        }

        # merge the dicts by uid
        combined = {}

        for uid, entry in semantic_dict.items():
            combined[uid] = entry

        for uid, entry in tag_dict.items():
            if uid in combined:
                combined[uid]["tag_score"] = max(
                    combined[uid]["tag_score"], entry["tag_score"]
                )
            else:
                combined[uid] = entry

        # calculate hybrid score
        for uid in combined:
            semantic = combined[uid]["semantic_score"]
            tag = combined[uid]["tag_score"]
            combined[uid]["hybrid_score"] = alpha * semantic + (1 - alpha) * tag

        # sort by hybrid score
        top_results = sorted(
            combined.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:n_results]
        return top_results
