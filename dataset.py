from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer, models as tok_models, pre_tokenizers, AddedToken
from config import NanoQwenConfig

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))


MAZE_VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "GRID_START": 4, "GRID_END": 5, "PATH_START": 6, "DONE": 7,
    "PATH": 8, "WALL": 9, "GOAL": 10, "START": 11, "NEWLINE": 12,
    "UP": 13, "DOWN": 14, "LEFT": 15, "RIGHT": 16, "\n": 17,
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
    tok.enable_padding(
        length=NanoQwenConfig.max_seq_len,    # Use 'length' here to specify the exact target token limit# Gets the integer ID of your [PAD] token
        pad_token="<pad>",
        pad_id=pad_id,
        direction="right"  # Pads on the right side
)

    return tok


if __name__ == "__main__":
    tokenizer = _tokenizer()
    print(tokenizer.encode("hello world"))
    print(tokenizer.decode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
 