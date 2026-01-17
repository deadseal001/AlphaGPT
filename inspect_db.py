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
    
    await conn.close()
    print("\n✓ Inspection complete!\n")

asyncio.run(inspect_database())