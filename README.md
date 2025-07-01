# RMS-Agent

This repository contains the implementation of **RMS-Agent**.

## Data 

Preprocessed data is stored in:

code/data/preprocessed/
├── cleaned_data_2015.csv
├── train_2015.csv
├── val_2015.csv
├── test_2015.csv
├── train_2017.csv
├── val_2017.csv
├── test_2017.csv
├── train_2019.csv
└── test_2019.csv

## 🚀 Training

To train the model using Distributed Data Parallel (DDP), run the following command (Take the 2015 data as an example):

```
sh run_ddp.sh LLM_with_mlp_query 1e-3 128 0 1
```

The training log and the test results will be saved under "./log/" folder. 