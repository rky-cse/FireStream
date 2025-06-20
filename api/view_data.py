import psycopg2

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "firestream_db"
DB_USER = "postgres"
DB_PASS = "postgres"

def main():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        # Example: Get all users
        cur.execute("SELECT * FROM users;")
        rows = cur.fetchall()
        print("Users:")
        for row in rows:
            print(row)

        # Example: Get all content
        cur.execute("SELECT * FROM content;")
        rows = cur.fetchall()
        print("\nContent:")
        for row in rows:
            print(row)

        # Example: Get watch history
        cur.execute("SELECT * FROM watch_history;")
        rows = cur.fetchall()
        print("\nWatch History:")
        for row in rows:
            print(row)

        # Example: Get content reactions
        cur.execute("SELECT * FROM content_reactions;")
        rows = cur.fetchall()
        print("\nContent Reactions:")
        for row in rows:
            print(row)

        # Similarly, you can fetch from other tables:
        # user_connections, viewing_groups, group_members, group_sessions,
        # user_contexts, content_tags, search_history, user_preferences
        # just by running: cur.execute("SELECT * FROM table_name;")

    conn.close()

if __name__ == "__main__":
    main()