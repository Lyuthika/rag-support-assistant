import psycopg


DATABASE_URL = "postgresql://raguser:ragpassword@localhost:5432/ragdb"


with psycopg.connect(DATABASE_URL) as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT version();")
        result = cursor.fetchone()

        print("Database connection successful!")
        print("PostgreSQL:", result[0])