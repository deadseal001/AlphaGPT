import asyncio
import json
import time
from loguru import logger
import pandas as pd
import os

from data_pipeline.data_manager import DataManager
from model_core.vm import StackVM
from model_core.data_loader import CryptoDataLoader
# Use our Paper Trader
from .paper_trader import PaperTrader
# Reuse Portfolio but with different file
from strategy_manager.portfolio import PortfolioManager
# Reuse Risk/Config
from strategy_manager.config import StrategyConfig

class PaperStrategyRunner:
    def __init__(self):
        logger.info("🎮 Starting Paper Trading Mode...")
        self.data_mgr = DataManager()
        
        # Use a separate state file for paper trading
        self.portfolio = PortfolioManager(state_file="paper_portfolio_state.json")
        
        # Initialize Paper Trader with 10 SOL
        self.trader = PaperTrader(initial_balance_sol=10.0)
        self.vm = StackVM()
        
        self.loader = CryptoDataLoader()
        self.token_map = {} 
        self.last_scan_time = 0
        
        try:
            with open("best_meme_strategy.json", "r") as f:
                data = json.load(f)
                self.formula = data if isinstance(data, list) else data.get("formula")
                # Fix: ensure formula is a list of ints
                if isinstance(self.formula, dict):
                    self.formula = self.formula.get("formula", [])
            logger.success(f"Loaded Strategy: {self.formula}")
        except FileNotFoundError:
            logger.critical("Strategy file not found! Please train model first.")
            exit(1)

    async def initialize(self):
        await self.data_mgr.initialize()
        bal = await self.trader.rpc.get_balance()
        logger.info(f"Bot Ready. Virtual Balance: {bal:.4f} SOL")

    async def run_loop(self):
        logger.info("🎮 Paper Runner Loop Started")
        
        while True:
            try:
                loop_start = time.time()
                
                # Sync data (Simulate live feed)
                if time.time() - self.last_scan_time > 120: 
                    logger.info("Scanning for new data...")
                    await self.data_mgr.pipeline_sync_daily()
                    self.last_scan_time = time.time()

                self.loader.load_data(limit_tokens=300)
                await self._build_token_mapping()

                await self.monitor_positions()
                
                if self.portfolio.get_open_count() < StrategyConfig.MAX_OPEN_POSITIONS:
                    await self.scan_for_entries()
                
                # Print Dashboard
                await self.print_dashboard()
                
                elapsed = time.time() - loop_start
                sleep_time = max(10, 60 - elapsed)
                logger.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Paper Loop Error: {e}")
                await asyncio.sleep(30)

    async def print_dashboard(self):
        # Calculate Total Equity
        cash = self.trader.balance
        holdings_value = 0.0
        
        logger.info("============== 📝 PAPER ACCOUNT DASHBOARD ==============")
        logger.info(f"💰 Cash Balance: {cash:.4f} SOL")
        logger.info(f"📦 Open Positions: {len(self.portfolio.positions)}")
        
        for token, pos in self.portfolio.positions.items():
            curr_price = await self._fetch_live_price_paper(token)
            val = pos.amount_held * curr_price
            holdings_value += val
            pnl_pct = (curr_price - pos.entry_price) / pos.entry_price * 100
            logger.info(f"   - {token[:8]}...: {pnl_pct:+.2f}% | Val: {val:.2f} SOL")
            
        total_equity = cash + holdings_value
        # Assuming 10.0 initial, or we should track initial better. 
        # For now hardcode 10.0 or use trader.initial_balance if we updated trader to store it.
        initial = 10.0 
        roi = (total_equity - initial) / initial * 100
        
        logger.info(f"💎 Total Equity: {total_equity:.4f} SOL")
        logger.info(f"📈 Total Return: {roi:+.2f}%")
        logger.info("======================================================")

    async def _build_token_mapping(self):
        query = """
        SELECT address, count(*) as cnt 
        FROM ohlcv 
        GROUP BY address 
        ORDER BY cnt DESC 
        LIMIT 300
        """
        df = pd.read_sql(query, self.loader.engine)
        addresses = df['address'].tolist()
        self.token_map = {addr: idx for idx, addr in enumerate(addresses)}

    async def _fetch_live_price_paper(self, token_address):
        q = f"SELECT time, close FROM ohlcv WHERE address='{token_address}' ORDER BY time DESC LIMIT 1"
        try:
            df = pd.read_sql(q, self.loader.engine)
            if not df.empty:
                ts = df.iloc[0]['time']
                price = float(df.iloc[0]['close'])
                logger.info(f"DEBUG: {token_address[:5]} Latest Time: {ts} Price: {price}")
                return price
        except Exception:
             pass
        return 0.0
         

    async def scan_for_entries(self):
        # Execute Strategy Logic
        res = self.vm.execute(self.formula, self.loader.feat_tensor)
        if res is None: return

        # Look at the LAST timestep result for trading
        latest_signals = res[-1, :] # Shape: [Num_Tokens]
        
        # Find Candidates
        candidates = []
        for addr, idx in self.token_map.items():
            if idx < len(latest_signals):
                score = float(latest_signals[idx])
                if score > 0.6: # Config threshold
                    candidates.append((addr, score))
        
        for addr, score in candidates:
            if addr in self.portfolio.positions: continue
            
            price = await self._fetch_live_price_paper(addr)
            if price <= 0: continue
            
            # Execute Paper Buy
            invest_sol = StrategyConfig.ENTRY_AMOUNT_SOL  # Fixed variable name
            success = await self.trader.buy(addr, invest_sol)
            
            if success:
                # Calculate estimated tokens received
                tokens_received = invest_sol / price
                self.portfolio.add_position(addr, "UNKNOWN", price, tokens_received, invest_sol)
                logger.info(f"🎮 Paper Position Opened: {addr} | Score: {score:.2f} | Price: {price}")

    async def monitor_positions(self):
        if not self.portfolio.positions: return

        for token_addr, pos in list(self.portfolio.positions.items()):
            current_price = await self._fetch_live_price_paper(token_addr)
            if current_price <= 0: continue

            self.portfolio.update_price(token_addr, current_price)
            
            # Simple Profit/StopLoss Logic
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            
            take_profit = 0.50 # +50%
            stop_loss = -0.20  # -20%
            
            if pnl_pct >= take_profit or pnl_pct <= stop_loss:
                # Sell everything
                est_return = pos.amount_held * current_price
                await self.trader.sell(token_addr, 1.0, est_value_sol=est_return)
                self.portfolio.close_position(token_addr)
                logger.info(f"🎮 CLOSED {token_addr} | PnL: {pnl_pct*100:.2f}% | Price: {current_price}")

if __name__ == "__main__":
    runner = PaperStrategyRunner()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(runner.initialize())
        loop.run_until_complete(runner.run_loop())
    except KeyboardInterrupt:
        loop.run_until_complete(runner.trader.close())
