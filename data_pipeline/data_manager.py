import asyncio
import aiohttp
from loguru import logger
from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider

class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.birdeye = BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    async def pipeline_sync_daily(self):
        logger.info("Step 1: Discovering trending tokens...")
        limit = 500 if Config.BIRDEYE_IS_PAID else 20
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)
            
            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长
            
            selected_tokens.append(t)
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")
        logger.info(f"📦 Tokens to fetch: {[t['symbol'] for t in selected_tokens]}")
        
        total_candles = 0
        
        async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
            for idx, t in enumerate(selected_tokens, 1):
                logger.info(f"🔄 Processing {idx}/{len(selected_tokens)}: {t['symbol']} ({t['address'][:8]}...)")
                
                records = await self.birdeye.get_token_history(session, t['address'])
                
                if records:
                    await self.db.batch_insert_ohlcv(records)
                    total_candles += len(records)
                    logger.info(f"💾 Inserted {len(records)} candles for {t['symbol']}. Total: {total_candles}")
                else:
                    logger.warning(f"⚠️  No data for {t['symbol']}")
                
                # Wait between tokens (except after the last one)
                if idx < len(selected_tokens):
                    logger.info(f"⏸️  Waiting 3 seconds before next token...")
                    await asyncio.sleep(3)
                
        logger.success(f"🎉 Pipeline complete! Total candles stored: {total_candles}")