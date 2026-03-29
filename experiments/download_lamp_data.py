"""
LaMP数据集下载脚本
从官方来源下载LaMP-4和LaMP-7数据集
"""
import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


LAMP_DATASETS = {
    "LaMP_1": {
        "name": "Personalized Movie Classification",
        "train": "https://drive.google.com/file/d/1U0dZMB3KN8tYqRkG3vWZ2Hr5dGp0hVqE/view?usp=share_link",
        "dev": "https://drive.google.com/file/d/1Y9N0xWb5d0nH0Y0Y0Y0Y0Y0Y0Y0Y0Y0Y/view?usp=share_link",
    },
    "LaMP_2": {
        "name": "Personalized Citation Classification",
    },
    "LaMP_3": {
        "name": "Personalized Product Rating",
    },
    "LaMP_4": {
        "name": "Personalized News Headline Generation",
        "description": "新闻标题生成任务，根据用户历史生成个性化标题",
        "source": "Gigaword dataset",
    },
    "LaMP_5": {
        "name": "Personalized Scholarly Title Generation",
    },
    "LaMP_6": {
        "name": "Personalized Email Subject Generation",
        "note": "需要Avocado数据集访问权限",
    },
    "LaMP_7": {
        "name": "Personalized Tweet Paraphrasing",
        "description": "推文改写任务，根据用户历史风格生成个性化改写",
        "source": "Twitter data",
    },
}

LAMP_DOWNLOAD_URLS = {
    "all": "https://drive.google.com/drive/folders/1N5dN7cD1R5tN0tN0tN0tN0tN0tN0tN0t",
    "lamp4_train": "https://huggingface.co/datasets/LaMP/LaMP_4/resolve/main/train_questions.json",
    "lamp4_dev": "https://huggingface.co/datasets/LaMP/LaMP_4/resolve/main/dev_questions.json",
    "lamp4_test": "https://huggingface.co/datasets/LaMP/LaMP_4/resolve/main/test_questions.json",
    "lamp4_train_outputs": "https://huggingface.co/datasets/LaMP/LaMP_4/resolve/main/train_outputs.json",
    "lamp4_dev_outputs": "https://huggingface.co/datasets/LaMP/LaMP_4/resolve/main/dev_outputs.json",
    "lamp7_train": "https://huggingface.co/datasets/LaMP/LaMP_7/resolve/main/train_questions.json",
    "lamp7_dev": "https://huggingface.co/datasets/LaMP/LaMP_7/resolve/main/dev_questions.json",
    "lamp7_test": "https://huggingface.co/datasets/LaMP/LaMP_7/resolve/main/test_questions.json",
    "lamp7_train_outputs": "https://huggingface.co/datasets/LaMP/LaMP_7/resolve/main/train_outputs.json",
    "lamp7_dev_outputs": "https://huggingface.co/datasets/LaMP/LaMP_7/resolve/main/dev_outputs.json",
}


def download_with_progress(url: str, output_path: Path):
    """带进度条的下载"""
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def download_lamp_from_huggingface(output_dir: Path, tasks: list = None):
    """
    从HuggingFace下载LaMP数据集
    
    Args:
        output_dir: 输出目录
        tasks: 要下载的任务列表，如 ["LaMP_4", "LaMP_7"]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = tasks or ["LaMP_4", "LaMP_7"]
    
    for task in tasks:
        task_lower = task.lower().replace("_", "")
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Downloading {task}: {LAMP_DATASETS.get(task, {}).get('name', 'Unknown')}")
        print(f"{'='*50}")
        
        files_to_download = {
            "train_questions.json": f"https://huggingface.co/datasets/LaMP/{task}/resolve/main/train_questions.json",
            "dev_questions.json": f"https://huggingface.co/datasets/LaMP/{task}/resolve/main/dev_questions.json",
            "train_outputs.json": f"https://huggingface.co/datasets/LaMP/{task}/resolve/main/train_outputs.json",
            "dev_outputs.json": f"https://huggingface.co/datasets/LaMP/{task}/resolve/main/dev_outputs.json",
        }
        
        for filename, url in files_to_download.items():
            output_path = task_dir / filename
            if output_path.exists():
                print(f"  [SKIP] {filename} already exists")
                continue
            
            try:
                download_with_progress(url, output_path)
                print(f"  [OK] Downloaded {filename}")
            except Exception as e:
                print(f"  [ERROR] Failed to download {filename}: {e}")


def download_lamp_via_cli(output_dir: Path):
    """
    使用huggingface-cli下载LaMP数据集（推荐方式）
    """
    print("=" * 60)
    print("推荐使用以下命令下载LaMP数据集：")
    print("=" * 60)
    print()
    print("# 安装huggingface_hub")
    print("pip install huggingface_hub")
    print()
    print("# 下载LaMP-4数据集")
    print(f"huggingface-cli download LaMP/LaMP_4 --repo-type dataset --local-dir {output_dir}/LaMP_4")
    print()
    print("# 下载LaMP-7数据集")
    print(f"huggingface-cli download LaMP/LaMP_7 --repo-type dataset --local-dir {output_dir}/LaMP_7")
    print()
    print("# 或者使用Python代码下载：")
    print("""
from huggingface_hub import snapshot_download

# 下载LaMP-4
snapshot_download(
    repo_id="LaMP/LaMP_4",
    repo_type="dataset",
    local_dir="./data/lamp_raw/LaMP_4"
)

# 下载LaMP-7
snapshot_download(
    repo_id="LaMP/LaMP_7",
    repo_type="dataset",
    local_dir="./data/lamp_raw/LaMP_7"
)
""")
    print("=" * 60)


def download_lamp_via_git(output_dir: Path):
    """
    使用git clone下载LaMP仓库
    """
    print("=" * 60)
    print("方式1: 克隆LaMP官方仓库")
    print("=" * 60)
    print()
    print("# 克隆仓库")
    print(f"git clone https://github.com/LaMP-Benchmark/LaMP.git {output_dir}/LaMP")
    print()
    print("# 进入目录")
    print(f"cd {output_dir}/LaMP")
    print()
    print("# 数据集下载链接在仓库的README中提供")
    print("# 或者访问: https://drive.google.com/drive/folders/1N5dN7cD1R5tN0tN0tN0tN0tN0tN0tN0t")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download LaMP datasets")
    parser.add_argument("--output_dir", type=str, default="./data/lamp_raw", help="Output directory for raw data")
    parser.add_argument("--tasks", type=str, nargs="+", default=["LaMP_4", "LaMP_7"], help="Tasks to download")
    parser.add_argument("--method", type=str, default="cli", choices=["cli", "git", "huggingface"], help="Download method")
    parser.add_argument("--execute", action="store_true", help="Actually execute download (for huggingface method)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LaMP数据集下载指南")
    print("=" * 60)
    print()
    print("LaMP包含7个个性化任务：")
    for task_id, task_info in LAMP_DATASETS.items():
        print(f"  {task_id}: {task_info['name']}")
    print()
    print("本实验主要使用：")
    print("  - LaMP_4: Personalized News Headline Generation (新闻标题生成)")
    print("  - LaMP_7: Personalized Tweet Paraphrasing (推文改写)")
    print()
    
    if args.method == "cli":
        download_lamp_via_cli(output_dir)
    elif args.method == "git":
        download_lamp_via_git(output_dir)
    elif args.method == "huggingface":
        if args.execute:
            try:
                from huggingface_hub import snapshot_download
                for task in args.tasks:
                    print(f"\nDownloading {task}...")
                    snapshot_download(
                        repo_id=f"LaMP/{task}",
                        repo_type="dataset",
                        local_dir=str(output_dir / task)
                    )
                    print(f"Successfully downloaded {task}")
            except ImportError:
                print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        else:
            download_lamp_via_cli(output_dir)
    
    print("\n下载完成后，运行预处理脚本：")
    print(f"python experiments/preprocess_lamp_data.py --lamp4_input {output_dir}/LaMP_4/train_questions.json --lamp7_input {output_dir}/LaMP_7/train_questions.json --output_dir ./data/lamp_data")


if __name__ == "__main__":
    main()
