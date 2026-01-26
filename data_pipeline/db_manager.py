import asyncpg
from loguru import logger
from .config import Config

class DBManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    chain TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    liquidity DOUBLE PRECISION, 
                    fdv DOUBLE PRECISION,
                    source TEXT,
                    PRIMARY KEY (time, address)
                );
            """)
            
            try:
                await conn.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);")
                logger.info("Converted ohlcv to Hypertable.")
            except Exception:
                logger.warning("TimescaleDB extension not found, using standard Postgres.")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_address ON ohlcv (address);")

    async def get_latest_timestamp(self, address):
        """Get the latest timestamp for a token's OHLCV data."""
        async with self.pool.acquire() as conn:
            val = await conn.fetchval("SELECT MAX(time) FROM ohlcv WHERE address = $1", address)
            return val

    async def delete_future_data(self, address, cutoff_dt):
        """Delete data points that are in the future (likely due to timezone mismatch)."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM ohlcv WHERE address = $1 AND time > $2", address, cutoff_dt)

    async def upsert_tokens(self, tokens):
        if not tokens: return
        async with self.pool.acquire() as conn:
            # tokens: list of (address, symbol, name, decimals, chain)
            await conn.executemany("""
                INSERT INTO tokens (address, symbol, name, decimals, chain, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (address) DO UPDATE 
                SET symbol = EXCLUDED.symbol, last_updated = NOW();
            """, tokens)

    async def batch_insert_ohlcv(self, records):
        if not records: return
        async with self.pool.acquire() as conn:
            try:
                async with conn.transaction():
                    # Create a temp table to stage the data
                    await conn.execute("""
                        CREATE TEMP TABLE IF NOT EXISTS tmp_ohlcv (
                            time TIMESTAMP,
                            address TEXT,
                            open DOUBLE PRECISION,
                            high DOUBLE PRECISION,
                            low DOUBLE PRECISION,
                            close DOUBLE PRECISION,
                            volume DOUBLE PRECISION,
                            liquidity DOUBLE PRECISION, 
                            fdv DOUBLE PRECISION,
                            source TEXT
                        ) ON COMMIT DROP;
                    """)

                    # Bulk copy into the temp table
                    await conn.copy_records_to_table(
                        'tmp_ohlcv',
                        records=records,
                        columns=['time', 'address', 'open', 'high', 'low', 'close', 
                                 'volume', 'liquidity', 'fdv', 'source'],
                        timeout=60
                    )

                    # Insert from temp table to main table with conflict handling
                    # Update existing records to ensure we have the latest version
                    await conn.execute("""
                        INSERT INTO ohlcv (time, address, open, high, low, close, volume, liquidity, fdv, source)
                        SELECT DISTINCT ON (time, address) * 
                        FROM tmp_ohlcv
                        ON CONFLICT (time, address) DO UPDATE 
                        SET open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            liquidity = EXCLUDED.liquidity,
                            fdv = EXCLUDED.fdv,
                            source = EXCLUDED.source;
                    """)
                    
            except Exception as e:
                logger.error(f"Batch insert error: {e}")