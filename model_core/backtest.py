import torch

from .config import ModelConfig  # Import config

class MemeBacktest:
    def __init__(self):
        self.trade_size = ModelConfig.TRADE_SIZE_USD # Use config
        self.min_liq = ModelConfig.MIN_LIQUIDITY     # Use config
        self.base_fee = ModelConfig.BASE_FEE         # Use config

    def evaluate(self, factors, raw_data, target_ret):
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
        
        if self.min_liq <= 0:
            is_safe = torch.ones_like(liquidity)
        else:
            is_safe = (liquidity > self.min_liq).float()
            
        # LOWER THRESHOLD to 0.6 to encourage early trading
        position = (signal > 0.6).float() * is_safe
        
        # ... (slippage logic same as before) ...
        denom = liquidity + 1e-9
        if self.min_liq <= 0:
             impact_slippage = 0.0
        else:
             impact_slippage = self.trade_size / denom
        impact_slippage = torch.clamp(torch.tensor(impact_slippage, device=factors.device), 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        
        activity = position.sum(dim=1)
        
        # SOFTER PENALTY: Gradient penalty instead of hard cliff
        # If activity < 5, penalize by distance from 5.
        # e.g., activity 0 => penalty -2.5. activity 4 => penalty -0.5
        penalty = torch.clamp(5 - activity, min=0) * 0.5
        score = score - penalty
        
        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()