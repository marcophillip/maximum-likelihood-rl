"""
Run the RL policy on data/val.json and compare each generated path to the label.
"""
import argparse
import json
import os

import torch

from config import NanoQwenConfig
from dataset import MazeDataset_RL, _tokenizer
from model import NanoQwen
from rewards import MazeEnv, parse_moves


@torch.no_grad()
def generate(model, prompt_ids, done_id, max_new_tokens, temperature):
    x = prompt_ids
    done = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    for _ in range(max_new_tokens):
        logits, _ = model(x, return_last_only=True)
        if temperature == 0:
            next_ids = logits.argmax(dim=-1, keepdim=True)
        else:
            next_ids = torch.multinomial((logits / temperature).softmax(-1), num_samples=1)
        x = torch.cat([x, next_ids], dim=1)
        done |= next_ids.squeeze(1) == done_id
        if done.all():
            break
    return x[:, prompt_ids.size(1):]


def score(prompt, moves, label, optimal_len):
    flags = {"exact_match": moves == label, "solved": False, "optimal": False,
             "format_err": moves is None}
    if moves is not None:
        outcome = MazeEnv(prompt.split()).simulate(moves)
        flags["solved"] = outcome["solved"]
        flags["optimal"] = outcome["solved"] and outcome["steps"] == optimal_len
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/maxrl/step_3000.pt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    cfg = NanoQwenConfig()
    model = NanoQwen(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(cfg.device).eval()

    tokenizer = _tokenizer()
    done_id = tokenizer.token_to_id("DONE")
    val_ds = MazeDataset_RL(tokenizer=tokenizer, json_file_path="data/val.json")
    labels = [e["sequence"].split("PATH_START")[1].split("DONE")[0].split() for e in val_ds.file]

    os.makedirs("eval_results", exist_ok=True)
    out_path = os.path.join("eval_results", os.path.splitext(os.path.basename(args.ckpt))[0] + ".jsonl")

    with open(out_path, "w") as f:
        for s in range(0, len(val_ds), args.batch_size):
            idxs = range(s, min(s + args.batch_size, len(val_ds)))
            prompts = [val_ds[i][0] for i in idxs]
            prompt_ids = torch.tensor([e.ids for e in tokenizer.encode_batch(prompts)],
                                      dtype=torch.long, device=cfg.device)
            gen = generate(model, prompt_ids, done_id, args.max_new_tokens, args.temperature)

            for row, i in enumerate(idxs):
                toks = [tokenizer.id_to_token(t) for t in gen[row].tolist()]
                moves = parse_moves(toks)
                flags = score(prompts[row], moves, labels[i], val_ds[i][1])
                if moves is not None:
                    generated = moves
                else:  # format error: keep the raw tokens up to DONE (or all of them)
                    generated = toks[:toks.index("DONE") + 1] if "DONE" in toks else toks
                record = {"idx": i, "label": labels[i], "generated": generated, "flags": flags}
                f.write(json.dumps(record) + "\n")
                print(f"{i:3d} | {' '.join(labels[i])} | {' '.join(generated)} | {flags}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
