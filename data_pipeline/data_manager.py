import asyncio
import aiohttp
from datetime import datetime, timezone
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

    async def pipeline_sync_daily(self, held_tokens=None):
        logger.info("Step 1: Discovering trending tokens...")
        limit = 500 if Config.BIRDEYE_IS_PAID else 20
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        
        # Fallback: if trending API fails, use existing tokens from database
        if not candidates:
            logger.warning("Failed to fetch trending tokens, using existing tokens from database...")
            # Get tokens that have recent OHLCV data (within last 24 hours)
            existing_query = """
                SELECT t.address, t.symbol, t.name, t.decimals, MAX(o.time) as last_update
                FROM tokens t
                JOIN ohlcv o ON t.address = o.address
                WHERE o.time >= NOW() - INTERVAL '24 hours'
                GROUP BY t.address, t.symbol, t.name, t.decimals
                ORDER BY last_update DESC
                LIMIT 100
            """
            result = await self.db.pool.fetch(existing_query)
            # Skip filter for database fallback (we trust these tokens already exist)
            candidates = [{'address': row['address'], 'symbol': row['symbol'], 'name': row['name'], 'decimals': row['decimals'], 'liquidity': float('inf'), 'fdv': float('inf')} for row in result]
            logger.info(f"Loaded {len(candidates)} tokens from database as fallback")
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)
            
            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长
            
            selected_tokens.append(t)

        # Merge held tokens
        existing_addrs = {t['address'] for t in selected_tokens}
        if held_tokens:
            for h_addr in held_tokens:
                if h_addr not in existing_addrs:
                    selected_tokens.append({
                        'address': h_addr,
                        'symbol': 'HELD_POS', 
                        'name': 'Held Position',
                        'decimals': 9,
                        'liquidity': 0, 'fdv': 0
                    })
                    existing_addrs.add(h_addr)
                    logger.info(f"➕ Added held token {h_addr} to fetch list")
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        # Only upsert tokens that have valid metadata (from trending list)
        tokens_to_upsert = [t for t in selected_tokens if t.get('symbol') != 'HELD_POS']
        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN) for t in tokens_to_upsert]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")
        
        total_candles = 0
        
        async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
            for idx, t in enumerate(selected_tokens, 1):
                sym = t.get('symbol', t['address'])
                logger.info(f"🔄 Processing {idx}/{len(selected_tokens)}: {sym} ({t['address'][:8]}...)")
                
                # Check for existing data for incremental fetch
                # Loop to ensure we catch up to real-time (handling API pagination/limits)
                max_loops = 10
                for loop_i in range(max_loops):
                    latest_ts = await self.db.get_latest_timestamp(t['address'])
                    time_from = None
                    
                    if latest_ts:
                        if latest_ts.tzinfo is None:
                            latest_ts_utc = latest_ts.replace(tzinfo=timezone.utc)
                        else:
                            latest_ts_utc = latest_ts

                        now_utc = datetime.now(timezone.utc)
                        
                        # Timezone/Future check
                        if latest_ts_utc > now_utc:
                            logger.warning(f"   ⚠️ Found future timestamp {latest_ts_utc} (Now: {now_utc}). Deleting bad data...")
                            await self.db.delete_future_data(t['address'], now_utc.replace(tzinfo=None))
                            # Reset loop to start fresh
                            time_from = None 
                        else:
                            # If we are within 5 minutes of now, we are done
                            if (now_utc - latest_ts_utc).total_seconds() < 300:
                                if loop_i == 0:
                                    logger.info(f"   ✅ Up to date (Loop {loop_i}).")
                                break
                            
                            time_from = int(latest_ts_utc.timestamp()) + 1
                            if loop_i == 0:
                                logger.info(f"   ↳ Incremental: Fetching since {latest_ts_utc}")
                            else:
                                logger.info(f"   ↳ Catch-up Loop {loop_i}: Fetching since {latest_ts_utc}")

                    records = await self.birdeye.get_token_history(session, t['address'], time_from=time_from)
                    
                    if records:
                        await self.db.batch_insert_ohlcv(records)
                        total_candles += len(records)
                        # Continue loop to see if there is more data
                        await asyncio.sleep(0.5) # Rate limit protection inside loop
                    else:
                        if loop_i == 0:
                            if not latest_ts: logger.warning(f"   ⚠️  No data for {sym}")
                        break
                
                # Wait between tokens (except after the last one)
                if idx < len(selected_tokens):
                    await asyncio.sleep(1.0)
                
        logger.success(f"🎉 Pipeline complete! Total candles stored: {total_candles}")