#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/qwen3_lora_sft.yaml

# 如果GPU使用数量发生变化，需要修改fsdp_config.yaml中的num_processes参数。例如：如果使用8张GPU，则需要将num_processes: 4修改为num_processes: 8。当前如果使用8GPU会报错。

# 需要修改的是qwen3_lora_sft.yaml，主要修改dataset的路径和模型的路径和输出目录

执行合并，需要修改配置文件qwen3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
