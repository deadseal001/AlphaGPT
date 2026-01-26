import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from loguru import logger
from ..config import Config
from .base import DataProvider

class BirdeyeProvider(DataProvider):
    def __init__(self):
        self.base_url = "https://public-api.birdeye.so"
        self.headers = {
            "X-API-KEY": Config.BIRDEYE_API_KEY,
            "accept": "application/json"
        }
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        
    async def get_trending_tokens(self, limit=100):
        url = f"{self.base_url}/defi/token_trending"
        # Birdeye enforces limit 1-20 for this endpoint
        limit = max(1, min(int(limit), 20))
        params = {
            "sort_by": "rank",
            "sort_type": "asc",
            "offset": "0",
            "limit": str(limit),
            "chain": Config.CHAIN
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_list = data.get('data', {}).get('tokens', [])
                        
                        results = []
                        for t in raw_list:
                            results.append({
                                'address': t['address'],
                                'symbol': t.get('symbol', 'UNKNOWN'),
                                'name': t.get('name', 'UNKNOWN'),
                                'decimals': t.get('decimals', 6),
                                'liquidity': t.get('liquidity', 0),
                                'fdv': t.get('fdv', 0)
                            })
                        return results
                    else:
                        error_text = await resp.text()
                        logger.error(f"Birdeye Trending Error: {resp.status}")
                        logger.warning(f"Birdeye Trending Response: {error_text}")
                        logger.warning(f"Birdeye Trending Params: {params}")
                        return []
            except Exception as e:
                logger.error(f"Birdeye Trending Exception: {e}")
                return []

    async def get_token_price(self, address: str):
        """Fetch current price for a single token."""
        url = f"{self.base_url}/defi/price"
        params = {"address": address}
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('data', {}).get('value', 0.0)
            except Exception as e:
                logger.error(f"Price fetch failed for {address}: {e}")
        return 0.0

    async def get_token_history(self, session, address, days=Config.HISTORY_DAYS, time_from=None, time_to=None, liquidity=0.0, fdv=0.0, retry_count=0, max_retries=5):
        if time_to is None:
            time_to = int(datetime.now().timestamp())
        
        if time_from is None:
            time_from = int((datetime.now() - timedelta(days=days)).timestamp())
        
        url = f"{self.base_url}/defi/ohlcv"
        params = {
            "address": address,
            "type": Config.TIMEFRAME,
            "time_from": time_from,
            "time_to": time_to
        }
    
        async with self.semaphore:
            try:
                logger.info(f"📊 Fetching OHLCV for {address[:8]}... (attempt {retry_count + 1}/{max_retries + 1})")
                logger.debug(f"Request params: {params}")
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get('data', {}).get('items', [])
                        if not items:
                            logger.warning(f"⚠️  No data returned for {address[:8]}")
                            return []
                        
                        logger.success(f"✅ Fetched {len(items)} candles for {address[:8]}")
                        formatted = []
                        for item in items:
                            formatted.append((
                                datetime.fromtimestamp(item['unixTime'], timezone.utc).replace(tzinfo=None), # time, force UTC naive
                                address,                                  # address
                                float(item['o']),                         # open
                                float(item['h']),                         # high
                                float(item['l']),                         # low
                                float(item['c']),                         # close
                                float(item['v']),                         # volume
                                float(liquidity),                         # liquidity
                                float(fdv),                               # fdv
                                'birdeye'                                 # source
                            ))
                        return formatted
                    elif resp.status == 429:
                        if retry_count >= max_retries:
                            logger.error(f"❌ Max retries reached for {address[:8]}, skipping...")
                            return []
                        
                        wait_time = 5 * (retry_count + 1)  # Exponential backoff: 5s, 10s, 15s, 20s, 25s
                        logger.warning(f"⏳ Rate limited for {address[:8]}, waiting {wait_time}s before retry {retry_count + 2}/{max_retries + 1}...")
                        await asyncio.sleep(wait_time)
                        # Fix argument passing
                        return await self.get_token_history(session, address, days=days, time_from=time_from, time_to=time_to, liquidity=liquidity, fdv=fdv, retry_count=retry_count + 1, max_retries=max_retries)
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ HTTP {resp.status} for {address[:8]}")
                        logger.debug(f"Error response: {error_text}")
                        return []
            except Exception as e:
                logger.error(f"❌ Exception for {address[:8]}: {e}")
                return []