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
        
    def load_data(self, limit_tokens=500, mandatory_tokens=None, days=None, max_staleness_minutes=30):
        print("Loading data from SQL...")
        
        # 1. Get top tokens WITH RECENT DATA (filter by recency)
        top_query = f"""
        WITH latest_times AS (
            SELECT address, MAX(time) as last_update
            FROM ohlcv
            GROUP BY address
            HAVING MAX(time) >= NOW() - INTERVAL '{max_staleness_minutes} minutes'
        )
        SELECT lt.address 
        FROM latest_times lt
        JOIN tokens t ON lt.address = t.address
        ORDER BY lt.last_update DESC
        LIMIT {limit_tokens}
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        
        print(f"✓ Found {len(addrs)} tokens with data updated in last {max_staleness_minutes} minutes")
        
        # Handle mandatory tokens (e.g., held positions)
        if mandatory_tokens:
            stale_mandatory = []
            for token in mandatory_tokens:
                if token not in addrs:
                    # Check if this token has ANY data
                    check_query = f"""
                    SELECT MAX(time) as last_update 
                    FROM ohlcv 
                    WHERE address = '{token}'
                    """
                    result = pd.read_sql(check_query, self.engine)
                    if not result.empty and result['last_update'][0] is not None:
                        addrs.append(token)
                        stale_mandatory.append(token)
                        print(f"⚠️  Added mandatory token {token[:8]}... (stale data)")
                    else:
                        print(f"❌ Skipping mandatory token {token[:8]}... (no data)")
            
            addrs = list(set(addrs))  # Remove duplicates
            
        if not addrs: raise ValueError("No tokens found with recent data.")
        
        # 2. Get OHLCV data
        addr_str = "'" + "','".join(addrs) + "'"
        
        time_filter = ""
        if days:
            time_filter = f"AND time >= NOW() - INTERVAL '{days} days'"
            
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        {time_filter}
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)

        if df.empty:
            print("⚠️  No OHLCV rows returned from DB for selected tokens/time range.")
        
        print("Pivoting data table...")
        # Optimization: Pivot once for all columns
        pivot_df = df.pivot(index='time', columns='address')
        
        # More conservative gap filling strategy
        # Only forward-fill small gaps (e.g., 5 minutes), don't fill large gaps with stale data
        pivot_df = pivot_df.fillna(method='ffill', limit=5)  # Limit ffill to 5 time periods
        pivot_df = pivot_df.fillna(0.0)  # Fill remaining with 0

        # Filter out tokens with too many zero closes (likely stale/invalid)
        if 'close' in pivot_df.columns.levels[0]:
            close_df = pivot_df['close']
            zero_frac = (close_df.values == 0).mean(axis=0)
            keep_mask = zero_frac <= 0.3
            keep_addrs = close_df.columns[keep_mask].tolist()
            dropped = len(close_df.columns) - len(keep_addrs)
            if dropped > 0:
                print(f"⚠️  Dropping {dropped} tokens with >30% zero closes")
                pivot_df = pivot_df.loc[:, pivot_df.columns.get_level_values(1).isin(keep_addrs)]
        
        print(f"✓ Loaded data for {len(pivot_df.columns.levels[1])} tokens, {len(pivot_df)} time steps")

        # Basic data sanity stats
        try:
            latest_ts = pivot_df.index.max()
            close_df = pivot_df['close'] if 'close' in pivot_df.columns.levels[0] else None
            if close_df is not None:
                nonzero_close = (close_df.values != 0).sum()
                total_close = close_df.size
                close_min = float(close_df.values.min()) if total_close else 0.0
                close_max = float(close_df.values.max()) if total_close else 0.0
                print(f"📊 Data stats: latest_ts={latest_ts} | close_nonzero={nonzero_close}/{total_close} | close_min={close_min:.6g} | close_max={close_max:.6g}")
        except Exception:
            pass
        
        # Save loaded tokens (address level only)
        # pivot_df columns are MultiIndex: (field, address)
        self.tokens = list(pivot_df.columns.levels[1])

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