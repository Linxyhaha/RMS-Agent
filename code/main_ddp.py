import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pandas as pd
import random
from functools import partial
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import *
from transformers.models.qwen2 import modeling_qwen2

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    PeftModel,

)

# ==== 数据列定义 ====
numerical_cols = [
    'age', 'education', 'wage', 'hh_size', 'count_car',
    'total_asset', 'total_income', 'total_debt', 'total_consumption', 'hospitalization_expenses'
]
categorical_cols = [
    'gender', 'hukou', 'marriage', 'physical_condition', 'work',
    'old_age_insurance', 'medical_insurance', 'unemployment_insurance',
    'housing_fund', 'poor_hh', 'house_type', 'phone_type', 
    'hhid', 'remark', 'pline', 'track', 'hhead', 'respond', 'year'
]

# ==== preprocess ====
def preprocess(df, fit=True):
    global num_imputer, num_scaler, cat_imputer, cat_encoder
    if fit:
        num_imputer = SimpleImputer(strategy='median')
        num_scaler = StandardScaler()
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        num_data = num_scaler.fit_transform(num_imputer.fit_transform(df[numerical_cols]))
        cat_data = cat_encoder.fit_transform(cat_imputer.fit_transform(df[categorical_cols]))
    else:
        num_data = num_scaler.transform(num_imputer.transform(df[numerical_cols]))
        cat_data = cat_encoder.transform(cat_imputer.transform(df[categorical_cols]))
    X = np.hstack([num_data, cat_data])
    text_input = df["prompt"]
    return X, text_input

# ==== Dataset 封装 ====
from torch.utils.data import Dataset
class TabularDataset(Dataset):
    def __init__(self, X, text_input, y):
        self.X = X
        self.y = y
        self.text_input = text_input
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.text_input[idx], torch.tensor(self.y[idx], dtype=torch.long)

# ==== collator function ====
def collator_fn(batch, tokenizer, max_len=1024, n_query=None):
    x_tabular, x_text_list, y = zip(*batch)
    x_tabular = torch.stack(x_tabular)
    y = torch.stack(y)
    input_ids_list, attention_mask_list = [], []

    for text in x_text_list:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt_text)
        if len(tokens) > max_len:
            tokens = tokens[-max_len:]
        input_ids_list.append(tokens)
        attention_mask_list.append([1] * len(tokens))

    max_batch_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids, padded_attention_mask = [], []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        pad_len = max_batch_len - len(input_ids)
        padded_ids = [tokenizer.pad_token_id] * pad_len + input_ids
        padded_mask = [0] * pad_len + attention_mask
        padded_mask = padded_mask + [1] + [1] * n_query
        padded_input_ids.append(padded_ids)
        padded_attention_mask.append(padded_mask)

    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
    return {
        'x_tabular': x_tabular,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': y
    }

# ==== Qwen 模型加载 ====
def load_LLM(ckpt_path, tokenizer_path, device, lora, reasoning_steps):

    modeling_qwen2.Qwen2Model = Qwen2Model_Query

    config = AutoConfig.from_pretrained(ckpt_path)
    config.reasoning_steps = reasoning_steps
    config.train_use_cache = False

    model = QwenWithReasoning.from_pretrained(ckpt_path, config=config, torch_dtype="float32", device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(f"/storage_fast/xylin/Qwen/{tokenizer_path}")
    tokenizer.padding_side = "left"

    

    if lora:
        # === Begin LoRA integration ===
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',  # Changed from FEATURE_EXTRACTION to CAUSAL_LM
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            target_modules=["q_proj", "v_proj"]  # or other target modules specific to your architecture
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # === End LoRA integration ===


    return model, tokenizer

# ==== train & eval ====
def train(model, loader, optimizer, criterion, device):
    model.train()
    for batch in tqdm(loader, total=len(loader)):
        x_tabular = batch['x_tabular'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(x_tabular, input_ids, attention_mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_tabular = batch['x_tabular'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            logits = model(x_tabular, input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc

# ==== main entry ====
def main_ddp(
            train_data: str = "../../../data/CHFS/fusion/train_2015.csv",
            valid_data: str = "../../../data/CHFS/fusion/val_2015.csv",
            test_data: str = "../../../data/CHFS/fusion/test_2015.csv",
            model_class: str = "Fusion_model_late",
            learning_rate: float = 1e-3,
            num_epoch: int = 20,
            batch_size: int = 8,
            mlp_dim1: int = 128,
            mlp_dim2: int = 64,
            seed: int = 2025,
            ckpt_path: str = "",
            tokenizer_path: str = "",
            llm_proj_dim: int = 64,
            debug: bool = False,
            lora: bool = False,
            skip_layer: int = 0,
            n_query: int = 1,
         ):

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    Model = eval(model_class)

    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(valid_data)
    test_df = pd.read_csv(test_data)

    if debug:
        train_df = train_df[:10]
        val_df = val_df[:10]
        test_df = test_df[:10]

    for df in [train_df, val_df, test_df]:
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        df['returnee'] = df['returnee'].astype('category')

    X_train, X_text_tr = preprocess(train_df.drop(columns=['returnee']), fit=True)
    X_val, X_text_val = preprocess(val_df.drop(columns=['returnee']), fit=False)
    X_test, X_text_test = preprocess(test_df.drop(columns=['returnee']), fit=False)
    y_train = train_df['returnee'].astype(int).values
    y_val = val_df['returnee'].astype(int).values
    y_test = test_df['returnee'].astype(int).values

    llm, tokenizer = load_LLM(ckpt_path, tokenizer_path, f"cuda:{local_rank}", lora, n_query)
    collator = partial(collator_fn, tokenizer=tokenizer, max_len=1024, n_query=n_query)

    train_dataset = TabularDataset(X_train, X_text_tr, y_train)
    val_dataset = TabularDataset(X_val, X_text_val, y_val)
    test_dataset = TabularDataset(X_test, X_text_test, y_test)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = Model(X_train.shape[1], mlp_dim1, llm, skip_layer).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.module.apply(lambda m: init_weights(m, method='xavier'))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        train(model, train_loader, optimizer, criterion, device)

        if dist.get_rank() == 0:
            acc, prec, rec, f1, auc = evaluate(model, val_loader, device)
            if auc > best_auc:
                best_auc = auc
                print(f"Valid Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
                acc, prec, rec, f1, auc = evaluate(model, test_loader, device)
                # print(f"Test AUC: {auc:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
                print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main_ddp)