from dataset import MazeDataset, _tokenizer, collate_fn
from model import NanoQwen
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from config import NanoQwenConfig
from functools import partial




class MazeSFTTrainer:
    def __init__(self, config):

        self.model = NanoQwen(config)
        self.model = self.model.to(config.device)
        self.model.train()

        self.config =config

        print("loading dataset")
        tokenizer=_tokenizer()
        self.train_dataset = MazeDataset(tokenizer=tokenizer, 
                                         json_file_path="data/train.json")
        
        self.val_dataset = MazeDataset(tokenizer=tokenizer, 
                                       json_file_path="data/val.json")
        
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=config.batch_size, 
                                           shuffle=True,
                                           collate_fn=partial(collate_fn, pad_token_id=tokenizer.token_to_id("<pad>"))
        )
        
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         collate_fn=partial(collate_fn, pad_token_id=tokenizer.token_to_id("<pad>"))
 )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)



    def train(self):
        pbar = tqdm(range(1500), desc="Training")
        for steps  in pbar:
            input_ids, labels = next(iter(self.train_dataloader))
            input_ids, labels = input_ids.to(self.config.device), labels.to(self.config.device)

            self.optimizer.zero_grad()
            logits, loss = self.model(input_ids, labels)
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        torch.save(self.model.state_dict(), self.config.checkpoint_path)

            



if __name__ == "__main__":
    trainer = MazeSFTTrainer(NanoQwenConfig())
    trainer.train()