# RMS-Agent

This repository contains the implementation of **RMS-Agent**.

## Data 

Preprocessed tabular data is stored in "./data/preprocessed/" folder and the instruction data for LLM is stored in "./data/instruction_data/" folder. 
 

## 🚀 Training

To train the model using Distributed Data Parallel (DDP), run the following command (Take the 2015 data as an example):

```
sh run_ddp.sh LLM_with_mlp_query 1e-3 128 0 1
```

The training log and the test results will be saved under "./log/" folder. 