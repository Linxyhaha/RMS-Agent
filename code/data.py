import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import ipdb

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    
class CHFSDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0):
        
        data = pd.read_csv(train_file)
        self.data = data["prompt"].tolist()

        random.seed(seed)
        
        if sample > 0:
            random.shuffle(self.data)
            self.data = self.data[:sample]
            
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len

        self.get_inputs()  

    def get_inputs(self):
        inputs = []
        for sample in tqdm(self.data):
            inputs.append(self.pre(sample))
        
        self.inputs = inputs

    def __len__(self):
        return len(self.data)

    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response: 
{data_point["output"]}"""

    
    def pre(self, sample):
        
        prompt = sample

        # Chat format prompt
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(text, bos=True, eos=False)
        
        attention_mask = [1] * len(tokens)
        
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
        }    

    def __getitem__(self, idx):
        return self.inputs[idx]