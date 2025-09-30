# Hawkes Attention on Time Series

Part of the paper 
**"From Hawkes Processes to Attention: Time-Modulated Mechanisms for Event Sequences"**

This repository implements the time series variant of the **Hawkes Attention** model.

This project is implemented on top of the **iTransformer** framework.

- iTransformer (recommended): **[https://github.com/thuml/iTransformer]** — _please refer to the official iTransformer repo for usage and other details._

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments
Here is an example of running the experiments:
```bash
python -u run.py --is_training 1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model hawkes --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 0 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 8 --d_ff 64 --itr 1 --label_len 0 --num_new_layers 2 --n_new_heads 2 --tmlp_width 4 --tmlp_depth 2 --new_d_model 8 --patience 2 --d_layers 0 --train_epochs 10 --learning_rate 0.001 --batch_size 256
```



---




