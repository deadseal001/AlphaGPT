import asyncio
import os
from dotenv import load_dotenv
import asyncpg

load_dotenv()

async def main():
    dsn = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    conn = await asyncpg.connect(dsn)
    
    # Find full address
    rows = await conn.fetch("SELECT address, symbol FROM tokens WHERE address LIKE '7pskt%'")
    if not rows:
        print("Address not found")
        return

    address = rows[0]['address']
    print(f"Token: {address}")
    
    # Get recent OHLCV
    prices = await conn.fetch("SELECT time, close FROM ohlcv WHERE address = $1 ORDER BY time DESC LIMIT 10", address)
    for p in prices:
        print(f"{p['time']}: {p['close']}")
        
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
