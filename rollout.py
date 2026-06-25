import torch
from config import RLHFConfig
import torch.nn.functional as F
from dataset import _tokenizer


@torch.no_grad()
def rollout(model, prompt, config):
    """
    prompt -- prompt
    G: number of completions to sample per prompt (included in config)
    """

    device = next(model.parameters()).device
    model.eval()

    captured_logprobs=[]
    captured_masks=[]

    seq = prompt.to(device).to(torch.long)
    prompt_len = seq.shape[1]


    seq = seq.expand(config.G,-1).clone()  


    done = torch.zeros(config.G, dtype=torch.bool, device=device)

    max_new_tokens = config.max_seq_len - prompt_len
    for _ in range(max_new_tokens):
        if done.all():
            break

        logits, _ = model(seq)
        log_probs = F.log_softmax(logits, dim=-1)
        next_token = torch.multinomial(log_probs.exp(), 1) 

        
        
        step_mask = (~done).float().to(device)  # step mask is 0 for a seq already finished and 1 active ones
        token_logp = log_probs.gather(-1, next_token).squeeze(-1)
        token_logp = token_logp * step_mask
        
        next_token[done] = torch.tensor(tokenizer.encode("<pad>").ids, device=device)
        
        captured_logprobs.append(token_logp)
        captured_masks.append(step_mask)

        seq = torch.cat([seq, next_token], dim=1)
        done |= next_token.squeeze(-1).eq(torch.tensor(tokenizer.encode("DONE").ids, device=device))

        done |= next_token.squeeze(-1).eq(torch.tensor(tokenizer.encode("<eos>").ids, device=device))

    captured_logprobs = torch.stack(captured_logprobs, dim=1)
    return seq, captured_logprobs


if __name__ == "__main__":
    from model import NanoQwen
    from config  import RLHFConfig
    from dataset import MazeDataset, _tokenizer, collate_fn
    tokenizer=_tokenizer()
    
    config = RLHFConfig()
    m = NanoQwen(config).to("cuda")

    val_dataset = MazeDataset(tokenizer=tokenizer, 
                                       json_file_path="data/val.json")
    prompt = next(iter(val_dataset))[0].to("cuda")
    prompt = prompt.view(1, -1)


    g = rollout(m, prompt, config)     

    print(g[0].shape)                             