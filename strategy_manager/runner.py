import asyncio
import torch
import json
import time
from loguru import logger
import pandas as pd

from data_pipeline.data_manager import DataManager
from model_core.vm import StackVM
from model_core.data_loader import CryptoDataLoader
from model_core.ops import OPS_CONFIG
from model_core.factors import FeatureEngineer
from execution.trader import SolanaTrader
from execution.config import ExecutionConfig
from execution.utils import get_mint_decimals
from .config import StrategyConfig
from .portfolio import PortfolioManager
from .risk import RiskEngine

class StrategyRunner:
    def __init__(self):
        self.data_mgr = DataManager()
        self.portfolio = PortfolioManager()
        self.risk = RiskEngine()
        self.trader = SolanaTrader()
        self.vm = StackVM()
        
        self.loader = CryptoDataLoader()
        self.token_map = {} # {address: tensor_index} 用于快速查找特征
        self.last_scan_time = 0
        
        try:
            with open("best_meme_strategy.json", "r") as f:
                # 兼容早期版本
                data = json.load(f)
                self.formula = data if isinstance(data, list) else data.get("formula")
            logger.success(f"Loaded Strategy: {self.formula}")
        except FileNotFoundError:
            logger.critical("Strategy file not found! Please train model first.")
            exit(1)

    async def initialize(self):
        await self.data_mgr.initialize()
        bal = await self.trader.rpc.get_balance()
        logger.info(f"Bot Initialized. Wallet Balance: {bal:.4f} SOL")

    async def run_loop(self):
        logger.info(">_< | Strategy Runner Started (Live Mode)")
        
        while True:
            try:
                loop_start = time.time()
                
                # Sync data (Simulate live feed)
                if time.time() - self.last_scan_time > 900: # 15 min
                    logger.info("o.O | Syncing Data Pipeline...")
                    held_tokens = list(self.portfolio.positions.keys())
                    await self.data_mgr.pipeline_sync_daily(held_tokens=held_tokens)
                    self.last_scan_time = time.time()

                held_tokens = list(self.portfolio.positions.keys())
                # For live trading: strict 30 min staleness requirement
                self.loader.load_data(limit_tokens=300, mandatory_tokens=held_tokens, max_staleness_minutes=30)
                await self._build_token_mapping()

                await self.monitor_positions()
                
                if self.portfolio.get_open_count() < StrategyConfig.MAX_OPEN_POSITIONS:
                    await self.scan_for_entries()
                else:
                    logger.info("=-= | Max positions reached. Scanning skipped.")
                
                await self.print_dashboard()

                elapsed = time.time() - loop_start
                sleep_time = max(10, 60 - elapsed)
                logger.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Global Loop Error: {e}")
                await asyncio.sleep(30)

    async def _build_token_mapping(self):
        # Map tokens from loader exactly as they appear in the tensor
        if hasattr(self.loader, 'tokens'):
            self.token_map = {addr: idx for idx, addr in enumerate(self.loader.tokens)}
            logger.info(f"Mapped {len(self.token_map)} tokens for inference.")
        else:
            logger.error("Loader has no tokens attribute! Inference mapping will be wrong.")
            self.token_map = {}

    async def monitor_positions(self):
        if not self.portfolio.positions: return

        logger.info(f"o.O | Monitoring {len(self.portfolio.positions)} positions...")
        
        for token_addr, pos in list(self.portfolio.positions.items()):
            current_price = await self._fetch_live_price_sol(token_addr)
            if current_price <= 0:
                logger.warning(f"Could not fetch price for {pos.symbol}, skipping.")
                continue

            self.portfolio.update_price(token_addr, current_price)
            
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            
            if pnl_pct <= StrategyConfig.STOP_LOSS_PCT:
                logger.warning(f"!!! | STOP LOSS: {pos.symbol} PnL: {pnl_pct:.2%}")
                await self._execute_sell(token_addr, 1.0, "StopLoss")
                continue

            if not pos.is_moonbag and pnl_pct >= StrategyConfig.TAKE_PROFIT_Target1:
                logger.success(f"😄 | MOONBAG TP: {pos.symbol} PnL: {pnl_pct:.2%}")
                await self._execute_sell(token_addr, StrategyConfig.TP_Target1_Ratio, "Moonbag")
                pos.is_moonbag = True
                self.portfolio.save_state()
                continue

            max_gain = (pos.highest_price - pos.entry_price) / pos.entry_price
            drawdown = (pos.highest_price - current_price) / pos.highest_price
            
            if max_gain > StrategyConfig.TRAILING_ACTIVATION and drawdown > StrategyConfig.TRAILING_DROP:
                logger.warning(f"😠 | TRAILING STOP: {pos.symbol} Max: {max_gain:.2%} DD: {drawdown:.2%}")
                await self._execute_sell(token_addr, 1.0, "TrailingStop")
                continue

            if not pos.is_moonbag:
                ai_score = await self._run_inference(token_addr)
                if ai_score != -1 and ai_score < StrategyConfig.SELL_THRESHOLD:
                    logger.info(f"🤖 | AI EXIT: {pos.symbol} Score: {ai_score:.2f}")
                    await self._execute_sell(token_addr, 1.0, "AI_Signal")


    async def print_dashboard(self):
        # Calculate Total Equity
        cash = await self.trader.rpc.get_balance()
        holdings_value = 0.0
        
        # Fetch SOL price for USD conversion
        sol_address = "So11111111111111111111111111111111111111112"
        sol_price_usd = await self.data_mgr.birdeye.get_token_price(sol_address)
        if sol_price_usd == 0: sol_price_usd = 150.0 
        
        logger.info("============== 🚀 LIVE STRATEGY DASHBOARD ==============")
        logger.info(f"💰 Wallet Balance: {cash:.4f} SOL (${cash * sol_price_usd:.2f})")
        logger.info(f"📦 Open Positions: {len(self.portfolio.positions)}")
        
        for token, pos in self.portfolio.positions.items():
            # Price already updated in monitor_positions
            curr_price = pos.current_price if hasattr(pos, 'current_price') else 0.0 
            # If not updated recently, fetch again? monitor_positions updates it.
            if curr_price == 0:
                 curr_price = await self._fetch_live_price_sol(token)

            val = pos.amount_held * curr_price
            holdings_value += val
            pnl_pct = (curr_price - pos.entry_price) / pos.entry_price * 100
            
            logger.info(f"   - {pos.symbol} ({token[:4]}..): {pnl_pct:+.2f}% | Val: {val:.2f} SOL (${val * sol_price_usd:.2f})")
            
        total_equity = cash + holdings_value
        total_equity_usd = total_equity * sol_price_usd
        
        logger.info(f"💎 Total Equity: {total_equity:.4f} SOL (${total_equity_usd:.2f})")
        logger.info(f"ℹ️  SOL Price: ${sol_price_usd:.2f}")
        logger.info("======================================================")

    async def scan_for_entries(self):
        raw_signals = self.vm.execute(self.formula, self.loader.feat_tensor)
        
        if raw_signals is None: return

        latest_signals = raw_signals[:, -1]
        # Debug signal distribution
        try:
            sig_min = float(latest_signals.min().item())
            sig_max = float(latest_signals.max().item())
            sig_mean = float(latest_signals.mean().item())
            sig_std = float(latest_signals.std().item())
            logger.info(f"📊 | Signal stats (latest): min={sig_min:.6f}, max={sig_max:.6f}, mean={sig_mean:.6f}, std={sig_std:.6f}")
            if sig_std < 1e-6 and abs(sig_mean) < 1e-6:
                logger.warning("⚠️  | Sanity check: signals are flat (all ~0). Strategy may be degenerate.")
                logger.warning(f"🧪 | Strategy tokens: {self.formula}")
                logger.warning(f"🧪 | Strategy decoded: {self._decode_formula(self.formula)}")
                return
        except Exception:
            pass
        scores = torch.sigmoid(latest_signals).cpu().numpy() # 转为概率 0~1

        # Log signal score for each token (for debugging)
        idx_to_addr = {v: k for k, v in self.token_map.items()}
        for idx, score in enumerate(scores):
            token_addr = idx_to_addr.get(idx, "UNKNOWN")
            logger.info(f"🤖 | Signal: {token_addr} | Score: {score:.4f}")
        
        # 翻转排序，从高分到低分处理
        sorted_indices = scores.argsort()[::-1]
        
        # 反向查表：Index -> Address
        # (效率较低，但 Top 300 没关系)
        idx_to_addr = {v: k for k, v in self.token_map.items()}
        
        for idx in sorted_indices:
            score = float(scores[idx])
            
            if score < StrategyConfig.BUY_THRESHOLD:
                break # 后面的都不够分，不用看了
                
            token_addr = idx_to_addr.get(idx)
            if not token_addr: continue
            
            # 过滤已持仓
            if token_addr in self.portfolio.positions:
                logger.info(f"⏭️ | Skip {token_addr}: already in portfolio")
                continue
            
            # 从 loader 缓存获取该 Token 的最新流动性
            # raw_data_cache['liquidity']: [Tokens, Time]
            liq_usd = self.loader.raw_data_cache['liquidity'][idx, -1].item()
            
            logger.info(f"🔍 | Inspecting {token_addr} | Score: {score:.2f} | Liq: ${liq_usd:.0f}")
            
            is_safe = await self.risk.check_safety(token_addr, liq_usd)
            if is_safe:
                await self._execute_buy(token_addr, score)
            else:
                logger.info(f"⛔ | Risk check failed: {token_addr}")
                
                # 检查仓位上限
                if self.portfolio.get_open_count() >= StrategyConfig.MAX_OPEN_POSITIONS:
                    break

    def _decode_formula(self, formula_tokens):
        feat_names = ['RET', 'LIQ', 'PRESS', 'FOMO', 'DEV', 'LOG_VOL']
        ops = [cfg[0] for cfg in OPS_CONFIG]
        decoded = []
        for tok in formula_tokens:
            try:
                t = int(tok)
            except Exception:
                decoded.append(str(tok))
                continue
            if t < FeatureEngineer.INPUT_DIM:
                name = feat_names[t] if t < len(feat_names) else f"FEAT_{t}"
                decoded.append(name)
            else:
                op_idx = t - FeatureEngineer.INPUT_DIM
                if 0 <= op_idx < len(ops):
                    decoded.append(ops[op_idx])
                else:
                    decoded.append(f"OP_{t}")
        return decoded

    async def _execute_buy(self, token_addr, score):
        balance = await self.trader.rpc.get_balance()
        amount_sol = self.risk.calculate_position_size(balance)
        
        if amount_sol <= 0:
            logger.warning("Insufficient balance for new entry.")
            return

        logger.info(f"🎉 | EXECUTING BUY: {token_addr} | Amt: {amount_sol} SOL")
        
        amount_lamports = int(amount_sol * 1e9)
        quote = await self.trader.jup.get_quote(
            input_mint=ExecutionConfig.SOL_MINT,
            output_mint=token_addr,
            amount_integer=amount_lamports
        )
        
        if not quote:
            logger.error("Failed to get quote for buy.")
            return

        tx_signature = await self.trader.buy(token_addr, amount_sol)
        
        if tx_signature: # Assuming buy returns Sig or True
            # 更新 Portfolio
            # 由于链上查询余额有延迟，我们先用 Quote 的预估值 (outAmount) 记账
            # 为了防止连续重复下单
            expected_out = int(quote['outAmount'])
            
            decimals = await get_mint_decimals(token_addr, self.trader.rpc.client)
            token_amount_ui = expected_out / (10 ** decimals)
            
            entry_price_sol = amount_sol / token_amount_ui if token_amount_ui > 0 else 0
            
            self.portfolio.add_position(
                token=token_addr,
                symbol=f"Meme_{token_addr[:4]}", # 暂时用简写，后续可查 Metadata
                price=entry_price_sol,
                amount=token_amount_ui,
                cost_sol=amount_sol
            )
            logger.success(f"+ | Position Added: {token_amount_ui:.2f} units @ {entry_price_sol:.6f} SOL")

    async def _execute_sell(self, token_addr, ratio, reason):
        pos = self.portfolio.positions.get(token_addr)
        if not pos: return

        logger.info(f"- | EXECUTING SELL: {token_addr} | Ratio: {ratio:.0%} | Reason: {reason}")
        
        success = await self.trader.sell(token_addr, percentage=ratio)
        
        if success:
            new_amount = pos.amount_held * (1.0 - ratio)
            
            if ratio > 0.98 or new_amount * pos.entry_price < 0.001:
                self.portfolio.close_position(token_addr)
            else:
                self.portfolio.update_holding(token_addr, new_amount)
                
            logger.success(f"o.O | Trade Completed: {reason}")

    async def _run_inference(self, token_addr):
        idx = self.token_map.get(token_addr)
        if idx is None:
            return -1

        features = self.loader.feat_tensor[idx] # 此时是 2D Tensor
        
        features_batch = features.unsqueeze(0) # [1, F, T]
        
        res = self.vm.execute(self.formula, features_batch) # -> [1, Time]
        
        if res is None: return -1
        
        latest_logit = res[0, -1]
        score = torch.sigmoid(latest_logit).item()
        return score

    async def _fetch_live_price_sol(self, token_addr):
        try:
            # 1. 获取精度
            decimals = await get_mint_decimals(token_addr, self.trader.rpc.client)
            amount_1_unit = 10 ** decimals
            
            # 2. 询价: 1 Token -> ? SOL
            quote = await self.trader.jup.get_quote(
                input_mint=token_addr,
                output_mint=ExecutionConfig.SOL_MINT,
                amount_integer=amount_1_unit
            )
            
            if quote:
                out_lamports = int(quote['outAmount'])
                price_sol = out_lamports / 1e9
                return price_sol
            
        except Exception as e:
            logger.warning(f"Price fetch failed for {token_addr}: {e}")
        
        return 0.0

    async def shutdown(self):
        logger.info("O.o | Shutting down strategy runner...")
        await self.data_mgr.close()
        await self.trader.close()
        await self.risk.close()

if __name__ == "__main__":
    runner = StrategyRunner()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(runner.initialize())
        loop.run_until_complete(runner.run_loop())
    except KeyboardInterrupt:
        loop.run_until_complete(runner.shutdown())