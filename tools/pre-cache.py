# 1. Latent Caching (將影片轉為 Latent)
python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config /path/to/dataset.toml \
    --vae /workspace/models/vae/Wan2.1_VAE.pth \
    --batch_size 1 \
    --i2v # 如果是做 I2V

# 2. Text Encoder Caching (預先計算 T5 特徵)
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config /path/to/dataset.toml \
    --t5 /workspace/models/t5/models_t5_umt5-xxl-enc-bf16.pth \
    --batch_size 1 \
    --fp8_t5 # 如果 VRAM 小於 24GB，加上這個
