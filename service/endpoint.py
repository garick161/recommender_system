from fastapi import FastAPI, Depends
from psycopg2.extras import RealDictCursor
import psycopg2
from typing import List
from post_model import PostGet
import table_loader


# Формирование сессий
def get_db():
    conn = psycopg2.connect(
        "postgresql://user:password@host:dbname",
        cursor_factory=RealDictCursor
    )
    return conn


# Выполнение запроса
app = FastAPI()
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time, limit: int=5, db=Depends(get_db)) -> List[PostGet]:
    top_posts = tuple(table_loader.get_top_posts(id, time, limit))
    with db.cursor() as cursor:
        cursor.execute(
                f"""
                SELECT *
                FROM post
                WHERE id IN {top_posts}
                """)
        result = cursor.fetchall()
        return result
