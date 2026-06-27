import psycopg2
import numpy as np
from datetime import datetime
import config

# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create a PostgreSQL connection.
    """
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    """
    Create required tables if they do not already exist.
    """

    conn = get_connection()
    cur = conn.cursor()

    try:

        cur.execute("""

        CREATE TABLE IF NOT EXISTS detection_runs (

            image_id VARCHAR(50) PRIMARY KEY,

            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            original_image_name VARCHAR(255),

            target_object VARCHAR(100),

            detected_count INTEGER,

            full_image_embedding FLOAT[]

        );

        """)

        cur.execute("""

        CREATE TABLE IF NOT EXISTS crop_embeddings (

            id SERIAL PRIMARY KEY,

            image_id VARCHAR(50)
                REFERENCES detection_runs(image_id)
                ON DELETE CASCADE,

            crop_label VARCHAR(100),

            crop_embedding FLOAT[],

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        );

        """)

        conn.commit()

        print("✅ Database initialized.")

    finally:

        cur.close()
        conn.close()


# ============================================================
# COSINE SIMILARITY
# ============================================================

def calculate_cosine_similarity(emb1, emb2):

    if emb1 is None or emb2 is None:
        return 0.0

    a = np.array(emb1, dtype=np.float32)
    b = np.array(emb2, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)
