import os
import glob
import json
import shutil
import argparse
from pathlib import Path

def setup_dataset():
    # 1. 設定命令行參數解析
    parser = argparse.ArgumentParser(description="Initialize Wan2.1/2.2 LoRA Training Dataset")
    parser.add_argument("--dir", type=str, required=True, help="Target training directory (e.g., /workspace/abbie)")
    args = parser.parse_args()

    # 2. 設定基礎路徑 (基於輸入參數)
    base_dir = Path(args.dir).resolve()
    
    # 定義子目錄與檔案路徑
    images_dir = base_dir / "images"
    cache_dir = base_dir / "cache"
    jsonl_path = base_dir / "metadata.jsonl"
    toml_path = base_dir / "dataset.toml"

    # 自動偵測 Trigger Word (取資料夾名稱)
    trigger_word = base_dir.name
    
    print(f"🚀 初始化開始")
    print(f"📂 目標工作目錄: {base_dir}")
    print(f"🔑 Trigger Word: {trigger_word}")

    # 3. 檢查並建立目錄結構 (如果不存就建立)
    if not base_dir.exists():
        print(f"🛠️  建立主目錄: {base_dir}")
        base_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"🛠️  建立圖片目錄: {images_dir}")
        images_dir.mkdir(parents=True, exist_ok=True)
    
    if not cache_dir.exists():
        print(f"🛠️  建立快取目錄: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)

    # 4. 搜尋所有圖片 (支援多種格式)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(list(images_dir.glob(ext)))
        image_files.extend(list(images_dir.glob(ext.upper())))
    
    image_files.sort()

    # 如果沒有圖片，僅建立結構後退出，提示用戶上傳
    if not image_files:
        print(f"⚠️  警告: 在 {images_dir} 中找不到任何圖片。")
        print(f"ℹ️  請將圖片上傳至該目錄後，再次執行此腳本以生成 metadata 與 toml。")
        
        # 即使沒圖片，我們也可以先生成一個基本的 TOML 模板，方便用戶查看
        create_toml(toml_path, jsonl_path, cache_dir)
        return

    print(f"📸 找到 {len(image_files)} 張圖片，開始標準化處理...")

    # 5. 重新命名並建立 JSONL 內容
    jsonl_data = []
    rename_map = []
    
    for idx, img_path in enumerate(image_files, start=1):
        ext = img_path.suffix.lower()
        new_filename = f"{idx}{ext}"
        new_path = images_dir / new_filename
        
        rename_map.append((img_path, new_path))
        
        # 準備 JSONL 條目 (使用絕對路徑)
        entry = {
            "image_path": str(new_path),
            "caption": f"A caption for {trigger_word}" 
        }
        jsonl_data.append(entry)

    # 執行安全重新命名 (Temp renaming logic)
    temp_map = []
    for old_p, new_p in rename_map:
        if old_p != new_p:
            temp_name = old_p.parent / f"temp_{old_p.name}"
            try:
                shutil.move(str(old_p), str(temp_name))
                temp_map.append((temp_name, new_p))
            except Exception as e:
                print(f"❌ 移動失敗 {old_p}: {e}")
        else:
            temp_map.append((old_p, new_p))
            
    for temp_p, new_p in temp_map:
        try:
            shutil.move(str(temp_p), str(new_p))
        except Exception as e:
            print(f"❌重新命名失敗 {temp_p}: {e}")

    # 6. 寫入 metadata.jsonl
    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry) + '\n')
        print(f"✅ 已建立 Metadata: {jsonl_path}")
    except Exception as e:
        print(f"❌ 寫入 JSONL 失敗: {e}")

    # 7. 生成 dataset.toml
    create_toml(toml_path, jsonl_path, cache_dir)

    print("🎉 初始化完成！")

def create_toml(toml_path, jsonl_path, cache_dir):
    """將 TOML 生成邏輯獨立出來"""
    toml_content = f"""[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_jsonl_file = "{str(jsonl_path)}"
cache_directory = "{str(cache_dir)}"
num_repeats = 10
"""
    try:
        with open(toml_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
        print(f"✅ 已更新設定檔: {toml_path}")
    except Exception as e:
        print(f"❌ 寫入 TOML 失敗: {e}")

if __name__ == "__main__":
    setup_dataset()
