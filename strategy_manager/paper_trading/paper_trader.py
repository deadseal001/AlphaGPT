import asyncio
import json
import os
from loguru import logger

class MockRPC:
    def __init__(self, trader):
        self.trader = trader
    
    async def get_balance(self):
        return self.trader.balance

class PaperTrader:
    def __init__(self, initial_balance_sol=10.0, state_file="paper_balance.json"):
        self.state_file = state_file
        self.initial_balance = initial_balance_sol
        self.load_state()
        self.rpc = MockRPC(self)
        logger.info(f"📝 Paper Trader Initialized. Virtual Balance: {self.balance} SOL")

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', self.initial_balance)
            except:
                self.balance = self.initial_balance
        else:
            self.balance = self.initial_balance

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({'balance': self.balance}, f)

    async def buy(self, token_address: str, amount_sol: float, slippage_bps=500):
        if self.balance < amount_sol:
            logger.warning(f"📝 [PAPER FAIL] Insufficient funds: {self.balance} < {amount_sol}")
            return False
        
        self.balance -= amount_sol
        self.save_state()
        logger.success(f"📝 [PAPER BUY] Spent {amount_sol:.4f} SOL on {token_address}. New Balance: {self.balance:.4f} SOL")
        return True

    async def sell(self, token_address: str, percentage: float = 1.0, slippage_bps=500, est_value_sol=0.0):
        if est_value_sol <= 0:
            logger.warning("📝 [PAPER WARN] Sell value is 0 or unknown, assuming breakdown.")
            
        self.balance += est_value_sol
        self.save_state()
        logger.success(f"📝 [PAPER SELL] Sold {percentage*100}% of {token_address} for {est_value_sol:.4f} SOL. New Balance: {self.balance:.4f} SOL")
        return True
    
    async def close(self):
        logger.info(f"📝 Paper Trading Session Ended. PnL: {self.balance - self.initial_balance:.4f} SOL")
