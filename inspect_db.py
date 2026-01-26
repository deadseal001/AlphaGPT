#!/usr/bin/env python3
import asyncio
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()

async def inspect_database():
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASSWORD", "password")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "crypto_quant")
    
    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    conn = await asyncpg.connect(dsn=dsn)
    
    # Check TOKENS columns
    print("\n=== TOKENS TABLE COLUMNS ===")
    tokens_columns = await conn.fetch("""
        SELECT column_name, data_type 
        FROM information_schema.columns
        WHERE table_name = 'tokens'
        ORDER BY ordinal_position;
    """)
    for col in tokens_columns:
        print(f"  {col['column_name']:<15} {col['data_type']}")
    
    # Check OHLCV columns
    print("\n=== OHLCV TABLE COLUMNS ===")
    ohlcv_columns = await conn.fetch("""
        SELECT column_name, data_type 
        FROM information_schema.columns
        WHERE table_name = 'ohlcv'
        ORDER BY ordinal_position;
    """)
    for col in ohlcv_columns:
        print(f"  {col['column_name']:<15} {col['data_type']}")
    
    # Check row counts
    print("\n=== ROW COUNTS ===")
    tokens_count = await conn.fetchval("SELECT COUNT(*) FROM tokens;")
    ohlcv_count = await conn.fetchval("SELECT COUNT(*) FROM ohlcv;")
    print(f"  Tokens: {tokens_count}")
    print(f"  OHLCV:  {ohlcv_count}")
    
    # Sample token row
    print("\n=== SAMPLE TOKENS ROW ===")
    token_sample = await conn.fetchrow("SELECT * FROM tokens LIMIT 1;")
    if token_sample:
        for k, v in token_sample.items():
            print(f"  {k}: {v}")
    else:
        print("  No data in tokens table")
    
    # Sample OHLCV row
    print("\n=== SAMPLE OHLCV ROW ===")
    ohlcv_sample = await conn.fetchrow("SELECT * FROM ohlcv LIMIT 1;")
    if ohlcv_sample:
        for k, v in ohlcv_sample.items():
            print(f"  {k}: {v}")
    else:
        print("  No data in ohlcv table")
    
    # Most recent OHLCV record for each token
    print("\n=== MOST RECENT OHLCV RECORD PER TOKEN ===")
    recent_records = await conn.fetch("""
        WITH latest AS (
            SELECT address, MAX(time) as max_ts
            FROM ohlcv
            GROUP BY address
        )
        SELECT 
            t.symbol,
            o.address,
            o.time,
            o.close,
            o.volume
        FROM ohlcv o
        JOIN latest l ON o.address = l.address AND o.time = l.max_ts
        LEFT JOIN tokens t ON o.address = t.address
        ORDER BY o.time DESC
        LIMIT 20;
    """)
    
    if recent_records:
        print(f"  {'Symbol':<12} {'Address':<12} {'Latest Timestamp':<22} {'Close':<12} {'Volume'}")
        print("  " + "-" * 90)
        for rec in recent_records:
            symbol = rec['symbol'] or 'Unknown'
            addr_short = rec['address'][:8] + "..."
            ts = rec['time'].strftime('%Y-%m-%d %H:%M:%S') if rec['time'] else 'N/A'
            close = f"{rec['close']:.6f}" if rec['close'] else 'N/A'
            volume = f"{rec['volume']:.2f}" if rec['volume'] else 'N/A'
            print(f"  {symbol:<12} {addr_short:<12} {ts:<22} {close:<12} {volume}")
    else:
        print("  No OHLCV data found")
    
    await conn.close()
    print("\n✓ Inspection complete!\n")

asyncio.run(inspect_database())