import torch
import os
from dotenv import load_dotenv
load_dotenv()  # Add this line
class ModelConfig:
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("🚀 Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("🚀 Using NVIDIA GPU")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️  Using CPU")

    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 2048
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 0.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = 6