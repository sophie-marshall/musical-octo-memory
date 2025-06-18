import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


class PostgresClient:

    # class level configs
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __init__(
        self,
        pg_host: str = None,
        pg_user: str = None,
        pg_password: str = None,
        pg_db: str = None,
    ):
        # load environment variables
        load_dotenv()

        # Use provided values or fall back to environment variables
        self.pg_host = os.getenv("PG_HOST")
        self.pg_user = os.getenv("PG_USER")
        self.pg_password = os.getenv("PG_PASSWORD")
        self.pg_db = os.getenv("PG_DB")

        # Validate that we have all required connection parameters
        if not all([self.pg_host, self.pg_user, self.pg_password, self.pg_db]):
            raise ValueError(
                "Missing required database connection parameters. "
                "Either provide them explicitly or set environment variables."
            )

    @classmethod
    def _make_conn(cls, host=None, user=None, password=None, dbname=None):
        try:
            # Load environment variables if not provided
            load_dotenv()
            conn = psycopg2.connect(
                host=host or os.getenv(cls.ENV_VARS["host"]),
                user=user or os.getenv(cls.ENV_VARS["user"]),
                password=password or os.getenv(cls.ENV_VARS["password"]),
                dbname=dbname or os.getenv(cls.ENV_VARS["dbname"]),
            )
            register_vector(conn)
            return conn
        except Exception as e:
            print(f"Error connecting to Postgres: {str(e)}")
            return None

    @staticmethod
    def _normalize_embedding(embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    @staticmethod
    def _remove_stopwords(self, text: str) -> list:
        return [
            word.lower() for word in text.split() if word.lower() not in self.stop_words
        ]

    @classmethod
    def insert_content_embeddings(cls, data: list):
        try:
            conn = cls._make_conn()
            cursor = conn.cursor()
            for record in data:
                try:
                    norm_embedding = cls._normalize_embedding(record["embedding"])
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

    @classmethod
    def semantic_search(cls, query: list, n_results: int = 5):
        # convert query embedidng to the proper format
        query_embedding = cls.embedding_model.encode(query)
        norm_query_embedding = cls._normalize_embedding(np.array(query_embedding))
        query_embedding_str = ", ".join(map(str, norm_query_embedding.tolist()))

        # build query
        search = f"SELECT uid, document_id, tags, clean_text, 1 - (embedding <=> '[{query_embedding_str}]') as similarity_score FROM content_embeddings ORDER BY embedding <=> '[{query_embedding_str}]' LIMIT {n_results};"

        # establish a connection to the DB
        conn = cls._make_conn()
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

    @classmethod
    def tag_search(
        cls, query: str, n_results: int = 10, similarity_threshold: float = 0.3
    ):
        keywords = cls._remove_stopwords(query)
        if not keywords:
            return []

        conn = cls._make_conn()
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

    @classmethod
    def hybrid_search(cls, query: str, n_results: int = 5, alpha: float = 0.7):
        semantic_results = cls.semantic_search(query, n_results=5)
        tag_results = cls.tag_search(query, n_results=5)

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
