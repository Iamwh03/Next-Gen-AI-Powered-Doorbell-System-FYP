import secrets
import aiosqlite
from pathlib import Path
import time

# Use an absolute path to avoid confusion and ensure the directory exists
DB = Path(__file__).parent / "data/token_store.db"
DB.parent.mkdir(exist_ok=True)


async def init_db():
    """Initializes the database and table."""
    try:
        async with aiosqlite.connect(str(DB)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT PRIMARY KEY,
                    email TEXT,
                    used INTEGER DEFAULT 0,
                    issued INTEGER
                )
            """)
            await db.commit()
        print(f"[TokenDB] Database initialized successfully at {DB}")
    except Exception as e:
        print(f"[TokenDB ERROR] Failed to initialize database: {e}")


async def issue(email: str) -> str:
    """Issues a new token and saves it to the database."""
    tok = secrets.token_urlsafe(32)
    issued_time = int(time.time())

    print(f"[TokenDB DEBUG] Attempting to issue token for email: {email}")

    try:
        async with aiosqlite.connect(str(DB)) as db:
            print("[TokenDB DEBUG] Database connection successful.")
            await db.execute(
                "INSERT INTO tokens (token, email, issued) VALUES (?, ?, ?)",
                (tok, email, issued_time)
            )
            await db.commit()
            print(f"[TokenDB DEBUG] ✅ Successfully saved token to DB: {tok}")
            return tok
    except Exception as e:
        # This will print any error that occurs during the database operation
        print(f"[TokenDB DEBUG] ❌ FAILED to save token to DB. Error: {e}")
        # Return the token anyway so the email sends, but we know it failed to save
        return tok


async def check(token: str) -> bool:
    """
    Checks if a token is valid and unused.
    This function is used by the admin panel and does NOT consume the token.
    """
    try:
        async with aiosqlite.connect(str(DB)) as db:
            cur = await db.execute(
                "SELECT used FROM tokens WHERE token=?", (token,)
            )
            row = await cur.fetchone()
            # Returns True if the token exists and is not used (used == 0)
            return bool(row and row[0] == 0)
    except Exception as e:
        print(f"[TokenDB ERROR] Error during token check: {e}")
        return False


async def consume(token: str) -> bool:
    """
    Atomically checks if a token is valid and marks it as used.
    This is for the final registration step. Returns True only on the first valid call.
    """
    try:
        async with aiosqlite.connect(str(DB)) as db:
            # The 'UPDATE ... WHERE used = 0' ensures this operation is atomic.
            cur = await db.execute(
                "UPDATE tokens SET used = 1 WHERE token = ? AND used = 0",
                (token,)  # <-- This is the corrected line
            )
            await db.commit()

            # cur.rowcount will be 1 if a row was updated, and 0 otherwise.
            if cur.rowcount == 1:
                print(f"[TokenDB] ✅ Token consumed successfully: {token}")
                return True
            else:
                print(f"[TokenDB] ❌ Token already used or invalid, not consumed: {token}")
                return False
    except Exception as e:
        print(f"[TokenDB ERROR] Error during token consumption: {e}")
        return False