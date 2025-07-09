# llamafactory
0523训练模型使用的脚本，有待优化

# 当前我们主要修改的脚本
## 训练脚本
code/LLaMA/LLaMA-Factory-0523/LLaMA-Factory/examples/extras/fsdp_qlora/train.sh
code/LLaMA/LLaMA-Factory-0523/LLaMA-Factory/examples/accelerate/fsdp_config.yaml
code/LLaMA/LLaMA-Factory-0523/LLaMA-Factory/examples/extras/fsdp_qlora/qwen3_lora_sft.yaml

## 遭遇的问题


## 试训练200条数据的速度也很慢
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [01:00<00:00,  4.66s/it]
[INFO|2025-07-08 01:13:52] llamafactory.model.model_utils.checkpointing:143 >> Gradient checkpointing enabled.
[INFO|2025-07-08 01:13:52] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.
[INFO|2025-07-08 01:13:52] llamafactory.model.adapter:143 >> Upcasting trainable params to float32.
[INFO|2025-07-08 01:13:52] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA
[INFO|2025-07-08 01:13:52] llamafactory.model.model_utils.misc:143 >> Found linear modules: q_proj,o_proj,k_proj,gate_proj,down_proj,up_proj,v_proj,gate
[INFO|2025-07-08 01:14:11] llamafactory.model.loader:143 >> trainable params: 211,378,176 || all params: 30,743,500,800 || trainable%: 0.6876
[INFO|trainer.py:756] 2025-07-08 01:14:11,903 >> Using auto half precision backend
[INFO|trainer.py:2409] 2025-07-08 01:15:56,358 >> ***** Running training *****
[INFO|trainer.py:2410] 2025-07-08 01:15:56,359 >>   Num examples = 200
[INFO|trainer.py:2411] 2025-07-08 01:15:56,359 >>   Num Epochs = 10
[INFO|trainer.py:2412] 2025-07-08 01:15:56,359 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2415] 2025-07-08 01:15:56,359 >>   Total train batch size (w. parallel, distributed & accumulation) = 32
[INFO|trainer.py:2416] 2025-07-08 01:15:56,359 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2417] 2025-07-08 01:15:56,359 >>   Total optimization steps = 70
[INFO|trainer.py:2418] 2025-07-08 01:15:56,901 >>   Number of trainable parameters = 52,844,544
  0%|                                                                                                                                                             1%|█▋                                                                                                                     | 1/70 [29:59<34:28:58, 1799.11s/it]  3%|███▍                                                                                                                   | 2/70 [48:43<26:29:09, 1402.20s/it]

## code/LLaMA/LLaMA-Factory-0523/LLaMA-Factory/saves/Qwen3-30B-A3B_abliterated/lora/PSQO_100k_e5_0707/trainer_log.jsonl
{"current_steps": 60, "total_steps": 15625, "loss": 1.1856, "lr": 1.5089514066496162e-05, "epoch": 0.0192, "percentage": 0.38, "elapsed_time": "18:45:38", "remaining_time": "202 days, 18:49:31"}
{"current_steps": 70, "total_steps": 15625, "loss": 1.1217, "lr": 1.7647058823529414e-05, "epoch": 0.0224, "percentage": 0.45, "elapsed_time": "21:51:21", "remaining_time": "202 days, 8:42:06"}

## 训不动，训练速度过于缓慢
1. 基本参数设置：100k数据集，由于内存溢出报错，只能将batchsize设置为1。明显不科学。
2. 实际训练log：可以看出，训完得要4800多小时，即超200天。显然是不合理的，根据之前训练deepseek32B的结果，应当至多训练20天才合理。
   {'loss': 1.2133, 'grad_norm': 0.5021839141845703, 'learning_rate': 1.2531969309462916e-05, 'epoch': 0.02}                                     
  0%|▎                                                                                           | 52/15625 [16:16:21<48
  0%|▏                                                                     | 53/15625 [16:35:16<4861:41:45, 1123.95s/it]
3. GPU负载并不均衡，GPU5负载远大于4。在加载模型的步骤就有这种现象。
╒═════════════════════════════════════════════════════════════════════════════╕
│ NVITOP 1.3.2        Driver Version: 570.144       CUDA Driver Version: 12.8 │
├───────────────────────────────┬──────────────────────┬──────────────────────┤
│ GPU Fan Temp Perf Pwr:Usg/Cap │         Memory-Usage │ GPU-Util  Compute M. │
╞═══════════════════════════════╪══════════════════════╪══════════════════════╪════════════════════════════════════════╕
│   0 30%  35C  P8   14W / 300W │  22.69MiB / 47.99GiB │      0%      Default │ MEM: ▏ 0.0%                            │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   1 30%  36C  P8   11W / 300W │  17.38MiB / 47.99GiB │      0%      Default │ MEM: ▏ 0.0%                            │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   2 30%  33C  P8    7W / 300W │  17.38MiB / 47.99GiB │      0%      Default │ MEM: ▏ 0.0%                            │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   3 30%  35C  P8   12W / 300W │  17.38MiB / 47.99GiB │      0%      Default │ MEM: ▏ 0.0%                            │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   4 33%  62C  P0  111W / 300W │  28.89GiB / 47.99GiB │     99%      Default │ MEM: █████████████████▌ 60.2%          │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   5 45%  72C  P0  132W / 300W │  43.68GiB / 47.99GiB │     89%      Default │ MEM: ██████████████████████████▍ 91.0% │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   6 40%  68C  P0  122W / 300W │  34.74GiB / 47.99GiB │      5%      Default │ MEM: █████████████████████ 72.4%       │
├───────────────────────────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│   7 46%  73C  P0  110W / 300W │  36.66GiB / 47.99GiB │     64%      Default │ MEM: ██████████████████████▏ 76.4%     │
╘═══════════════════════════════╧══════════════════════╧══════════════════════╧════════════════════════════════════════╛
[ CPU: █████▍ 7.7%                                               UPTIME: 22:34:42 ]  ( Load Average:  4.81  5.02  5.07 )
[ MEM: █████▉ 8.4%                                                 USED: 159.4GiB ]  [ SWP: ▏ 0.0%                     ]

