MaxRL (arXiv 2602.02710) on 17x17 mazes.

flow: train.py (SFT, 1500 steps) -> train_grpo.py (MaxRL)
      rollouts -> binary reward -> A = (r - r_hat)/(r_hat + eps) -> on-policy policy-gradient step
