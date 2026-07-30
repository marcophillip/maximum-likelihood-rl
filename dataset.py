from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers import Tokenizer, models as tok_models, pre_tokenizers, AddedToken
from config import NanoQwenConfig
import json
import torch
import torch.nn.functional as F


MAZE_VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "GRID_START": 4, "GRID_END": 5, "PATH_START": 6, "DONE": 7,
    "PATH": 8, "WALL": 9, "GOAL": 10, "START": 11, "NEWLINE": 12,
    "UP": 13, "DOWN": 14, "LEFT": 15, "RIGHT": 16, "\n": 17, "PATH_END":18
}

def _tokenizer():
    tok = Tokenizer(tok_models.WordLevel(vocab=MAZE_VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    for t in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        tok.add_special_tokens([AddedToken(t, special=True)])

    tok.enable_truncation(
        max_length=NanoQwenConfig.max_seq_len,
        direction="right"  # Options: "right" to drop trailing tokens, "left" to drop leading tokens
    )

    pad_id = tok.token_to_id("<pad>")
    # tok.enable_padding(
    #     length=NanoQwenConfig.max_seq_len,    # Use 'length' here to specify the exact target token limit# Gets the integer ID of your [PAD] token
    #     pad_token="<pad>",
    #     pad_id=pad_id,
    #     direction="right"  # Pads on the right side
# )

    return tok


class MazeDataset(Dataset):
    def __init__(self, tokenizer, json_file_path: str):
        self.tokenizer = tokenizer

        with open(json_file_path, "r") as f:
            self.file = json.loads(f.read())


    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, index):
        sequence = self.file[index]["sequence"]
        encoded = self.tokenizer.encode(sequence).ids  # full sequence


        path_start_token = self.tokenizer.token_to_id("PATH_START")
        split_idx = encoded.index(path_start_token) + 1  # +1 to include PATH_START in context

        input_ids = encoded[:-1]
        targets = encoded[1:]
        labels = [-100] * (split_idx - 1) + targets[split_idx - 1:]

        input_ids = input_ids[:512]
        labels    = labels[:512]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        
def collate_fn(batch, pad_token_id):
    input_ids, labels = zip(*batch)

    # pad to longest sequence in the batch
    max_len = max(x.size(0) for x in input_ids)

    input_ids_padded = []
    labels_padded    = []

    for ids, lbl in zip(input_ids, labels):
        pad_len = max_len - ids.size(0)
        input_ids_padded.append(F.pad(ids, (0, pad_len), value=pad_token_id))
        labels_padded.append(F.pad(lbl, (0, pad_len), value=-100))  # -100 so padding is ignored in loss

    return (
        torch.stack(input_ids_padded), 
        torch.stack(labels_padded),  
    )


class MazeDataset_RL(Dataset):
    def __init__(self, tokenizer, json_file_path: str):
        self.tokenizer = tokenizer
        with open(json_file_path, "r") as f:
            self.file = json.loads(f.read())

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        seq = self.file[index]["sequence"]
        prompt = seq.split("PATH_START")[0] + "PATH_START"   # grid + PATH_START, no moves
        optimal_len = self.file[index]["optimal_path_length"]
        return prompt, optimal_len

# if __name__ == "__main__":
#     from tqdm import tqdm
#     tokenizer = _tokenizer()
#     dataset = MazeDataset(tokenizer=tokenizer, json_file_path="data/train.json")
#     for d in tqdm(dataset):
#         pass