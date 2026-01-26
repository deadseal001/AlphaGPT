# model_core/engine.py (建议全文件替换)

import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import numpy as np # 需要 import numpy

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .ops import OPS_CONFIG
from .backtest import MemeBacktest

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        """
        self.loader = CryptoDataLoader()
        # For training: use 24 hour staleness (Birdeye API has delays, not truly real-time)
        self.loader.load_data(days=7, max_staleness_minutes=1440)  # 24 hours
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()

        # Penalize degenerate formulas dominated by JUMP
        jump_idx = next((i for i, cfg in enumerate(OPS_CONFIG) if cfg[0] == 'JUMP'), None)
        self.jump_token = self.vm.feat_offset + jump_idx if jump_idx is not None else None
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            # 生成阶段
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            # --- 优化核心：Batch 内去重 (Batch Deduplication) ---
            # 转换成 list of tuples 以便哈希去重
            seqs_list = seqs.tolist()
            seqs_tuples = [tuple(s) for s in seqs_list]
            unique_formulas = set(seqs_tuples)
            
            # 缓存当前 Batch 的计算结果
            formula_rewards_map = {}
            
            # 只对唯一的公式进行回测
            for formula_tuple in unique_formulas:
                formula = list(formula_tuple)
                
                # VM 执行
                res = self.vm.execute(formula, self.loader.feat_tensor)
                
                if res is None:
                    formula_rewards_map[formula_tuple] = -5.0
                    continue
                
                if res.std() < 1e-4:
                    formula_rewards_map[formula_tuple] = -2.0
                    continue

                # Penalize formulas that overuse JUMP (often collapses signals to ~0)
                if self.jump_token is not None:
                    jump_count = sum(1 for t in formula if int(t) == self.jump_token)
                    if jump_count > 2:
                        formula_rewards_map[formula_tuple] = -3.0 - 0.5 * (jump_count - 2)
                        continue
                
                # 回测
                score, ret_val, big_drawdowns = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)

                # Add mild variance bonus to discourage flat signals
                score = score + 0.1 * torch.tanh(res.std())
                formula_rewards_map[formula_tuple] = score
                
                # 记录最佳
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Big Drawdowns {big_drawdowns.mean().item():.2f} | Formula {formula}")
            
            # 将分数映射回原来的 Batch 索引
            for i in range(bs):
                rewards[i] = formula_rewards_map[seqs_tuples[i]]
            # ----------------------------------------------------
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            loss = loss.mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue 
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)

        # Save results
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")

if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()