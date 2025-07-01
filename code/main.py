import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import fire
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from data import CHFSDataset
from model import *
import transformers
import ipdb
from functools import partial
from transformers.models.qwen2 import modeling_qwen2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== 数据处理模块 ====
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
    
    # id_data = df["data_id"].values.reshape(-1, 1)  # shape: (N, 1)
    X = np.hstack([num_data, cat_data])
    text_input = df["prompt"]
    # ipdb.set_trace()
    return X, text_input

# ==== Dataset 封装 ====
class TabularDataset(Dataset):
    def __init__(self, X, text_input, y):
        self.X = X
        self.y = y
        self.text_input = text_input
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx,:-1], self.text_input[idx], self.y[idx]  # -1 is the data_id

def collator_fn(batch, tokenizer, max_len=1024, n_query=None):
    """
    Collator function for batching tabular and text data
    Args:
        batch: list of tuples (x_tabular, x_text, y)
        tokenizer: tokenizer for text processing
        max_len: maximum sequence length
    Returns:
        dict with batched tensors
    """
    x_tabular, x_text_list, y = zip(*batch)
    
    # Stack tabular data and labels (they are already tensors)
    x_tabular = torch.stack(x_tabular)
    y = torch.stack(y)

    # Process text data
    input_ids_list = []
    attention_mask_list = []
    
    for text in x_text_list:
        # Generate prompt using the same logic as in data.py
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        tokens = tokenizer.encode(prompt_text)
        # ipdb.set_trace()
        # Truncate if too long
        if len(tokens) > max_len:
            tokens = tokens[-max_len:]
        
        input_ids_list.append(tokens)
        attention_mask_list.append([1] * len(tokens))
    
    # Find max length in this batch
    max_batch_len = max(len(ids) for ids in input_ids_list)
    
    # Pad sequences to max_batch_len with left padding
    padded_input_ids = []
    padded_attention_mask = []
    
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        # Calculate padding length
        pad_len = max_batch_len - len(input_ids)
        
        # Left padding
        padded_ids = [tokenizer.pad_token_id] * pad_len + input_ids
        padded_mask = [0] * pad_len + attention_mask
        padded_mask = padded_mask + [1] + [1] * n_query
        padded_input_ids.append(padded_ids)
        padded_attention_mask.append(padded_mask)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
    
    return {
        'x_tabular': x_tabular,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': y
    }

# ==== load Qwen 模型 ==== 
def load_LLM(ckpt_path, tokenizer_path, reasoning_steps):
    model_name = ckpt_path

    config = AutoConfig.from_pretrained(model_name)
    config.reasoning_steps = reasoning_steps
    config.train_use_cache = False
    
    model = QwenWithReasoning.from_pretrained(model_name, config=config, torch_dtype="float32", device_map={"": "cuda:0"})
    tokenizer = AutoTokenizer.from_pretrained(f"/storage_fast/xylin/Qwen/{tokenizer_path}")
    tokenizer.padding_side = "left"
    return model, tokenizer
    

# ==== 训练与评估函数 ====
def train(model, loader, optimizer, criterion):
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

def evaluate(model, loader):
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
            all_probs.extend(probs[:,1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc

def main(train_data: str = "../../../data/CHFS/fusion/train_2015.csv",
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
         skip_layer: int = 0, 
         n_query: int = 1, 
         ):

    # ===== 设置随机种子 =====
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Model = eval(model_class)

    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(valid_data)
    test_df = pd.read_csv(test_data)

    if debug:
        ###### debug only
        train_df = train_df[:10]
        val_df = val_df[:10]
        test_df = test_df[:10]
        ###### debug only

    for col in categorical_cols:
        print(f"{col}: {train_df[col].nunique()}")

    for df in [train_df, val_df, test_df]:
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        df['returnee'] = df['returnee'].astype('category')

    X_train = train_df.drop(columns=['returnee'])
    y_train = train_df['returnee'].astype(int)
    X_val = val_df.drop(columns=['returnee'])
    y_val = val_df['returnee'].astype(int)
    X_test = test_df.drop(columns=['returnee'])
    y_test = test_df['returnee'].astype(int)

    # 1. preprocess 这里，改成每个数据有tabular data, 有 text data，这样就可以去掉data id了 - ok
    # 2. 额外定义一个collator function, 把input id给左pad上 - ok
    # 3. 在model forward里面用llm来去提取feature
    X_tabular_tr, X_text_tr = preprocess(X_train, fit=True)
    X_train_tensor = torch.tensor(X_tabular_tr, dtype=torch.float32)
    
    X_tabular_val, X_text_val = preprocess(X_val, fit=False)
    X_val_tensor = torch.tensor(X_tabular_val, dtype=torch.float32)

    X_tabular_test, X_text_test = preprocess(X_test, fit=False)
    X_test_tensor = torch.tensor(X_tabular_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # ==== load llm and tokenizer ====
    modeling_qwen2.Qwen2Model = Qwen2Model_Query
    # ipdb.set_trace()
    llm, tokenizer = load_LLM(ckpt_path, tokenizer_path, n_query)
    
    collator = partial(collator_fn, tokenizer=tokenizer, max_len=1024, n_query=n_query)

    train_loader = DataLoader(TabularDataset(X_train_tensor, X_text_tr, y_train_tensor), batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(TabularDataset(X_val_tensor, X_text_val, y_val_tensor), batch_size=batch_size, collate_fn=collator)
    test_loader = DataLoader(TabularDataset(X_test_tensor, X_text_test, y_test_tensor), batch_size=batch_size, collate_fn=collator)

    model = Model(X_train_tensor.shape[1]-1, mlp_dim1=mlp_dim1, llm=llm, skip_layer=skip_layer).to(device)
    model.apply(lambda m: init_weights(m, method='xavier'))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    for epoch in tqdm(range(num_epoch), total=num_epoch):
        train(model, train_loader, optimizer, criterion)
        acc, prec, rec, f1, auc = evaluate(model, val_loader)
        if auc > best_auc:
            print(f"=== Epoch {epoch}")
            print(f"Valid Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            best_auc = auc

            acc, prec, rec, f1, auc = evaluate(model, test_loader)
            print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    fire.Fire(main)
