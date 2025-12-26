accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py \
    --task i2v-A14B \
    --dit /workspace/models/dit_wan22/wan2.2_i2v_14b_low_noise_bf16.safetensors \
    --dit_high_noise /workspace/models/dit_wan22/wan2.2_i2v_14b_high_noise_bf16.safetensors \
    --dataset_config /path/to/dataset.toml \
    --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 \
    --network_module networks.lora_wan --network_dim 16 \
    --output_dir /workspace/output/customer_001 \
    --output_name my_wan22_lora
