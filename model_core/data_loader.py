import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        
        # 1. Get top tokens
        top_query = f"""
        SELECT address FROM tokens 
        LIMIT {limit_tokens} 
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs: raise ValueError("No tokens found.")
        
        # 2. Get OHLCV data
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        
        print("Pivoting data table...")
        # Optimization: Pivot once for all columns
        pivot_df = df.pivot(index='time', columns='address')
        # Fill missing values
        pivot_df = pivot_df.ffill().fillna(0.0)

        def to_tensor(col_name):
            # Optimized: Access pre-pivoted data directly from MultiIndex
            if col_name in pivot_df.columns.levels[0]:
                data = pivot_df[col_name].values.T
                return torch.tensor(data, dtype=torch.float32, device=ModelConfig.DEVICE)
            else:
                 # Fallback if column missing (shouldn't happen with correct SQL)
                return torch.zeros((pivot_df.shape[1], pivot_df.shape[0]), device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        
        # Safe Return Calculation
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        
        # Avoid log(0) or division by zero
        # 1. Mask where prices are 0
        valid_mask = (t1 > 1e-6) & (t2 > 1e-6)
        
        # 2. Compute returns only on valid data (otherwise 0)
        ret_raw = t2 / (t1 + 1e-9)
        self.target_ret = torch.zeros_like(ret_raw)
        
        # Use where to safely compute log
        self.target_ret = torch.where(
            valid_mask, 
            torch.log(ret_raw + 1e-9), 
            torch.zeros_like(ret_raw)
        )
        
        # Clean up any remaining NaNs/Infs just in case
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_ret[:, -2:] = 0.0