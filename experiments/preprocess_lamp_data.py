"""
LaMP数据集预处理脚本
将LaMP-4和LaMP-7数据集转换为DUA-RAG实验所需的格式
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import random


def parse_lamp_timestamp(ts_str: str) -> str:
    """解析LaMP时间戳格式"""
    try:
        if isinstance(ts_str, (int, float)):
            dt = datetime.fromtimestamp(ts_str)
        else:
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ]:
                try:
                    dt = datetime.strptime(str(ts_str), fmt)
                    break
                except ValueError:
                    continue
            else:
                dt = datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def convert_lamp4_to_dua_format(
    lamp_data: List[Dict[str, Any]],
    split_mode: str = "user"
) -> List[Dict[str, Any]]:
    """
    将LaMP-4数据集转换为DUA-RAG格式
    
    LaMP-4: 电影推荐任务
    - 输入: 用户历史评分记录
    - 输出: 个性化电影推荐
    
    Args:
        lamp_data: LaMP-4原始数据
        split_mode: "user" 或 "time" 分割模式
    """
    converted = []
    
    for item in lamp_data:
        user_id = item.get("user_id", item.get("id", "unknown"))
        profile = item.get("profile", [])
        history = item.get("history", [])
        
        haystack_sessions = []
        haystack_session_ids = []
        haystack_dates = []
        
        for idx, hist_item in enumerate(history):
            session_id = f"{user_id}_session_{idx}"
            
            movie_title = hist_item.get("title", "")
            movie_rating = hist_item.get("rating", "")
            movie_review = hist_item.get("review", "")
            timestamp = hist_item.get("timestamp", "")
            
            session_content = []
            if movie_title:
                session_content.append({
                    "role": "user",
                    "content": f"我观看了电影《{movie_title}》，我的评分是{movie_rating}星。{movie_review}"
                })
            
            if session_content:
                haystack_sessions.append(session_content)
                haystack_session_ids.append(session_id)
                haystack_dates.append(parse_lamp_timestamp(timestamp))
        
        query_item = item.get("input", "")
        if isinstance(query_item, dict):
            query_text = query_item.get("title", "")
        else:
            query_text = str(query_item)
        
        target = item.get("output", "")
        
        converted_item = {
            "user_id": str(user_id),
            "profile_id": str(user_id),
            "question": f"请根据我的观影历史，为我推荐电影。{query_text}",
            "answer": str(target),
            "haystack_sessions": haystack_sessions,
            "haystack_session_ids": haystack_session_ids,
            "haystack_dates": haystack_dates,
            "question_date": haystack_dates[-1] if haystack_dates else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "profile": profile,
            "task_type": "movie_recommendation",
            "split_mode": split_mode
        }
        
        converted.append(converted_item)
    
    return converted


def convert_lamp7_to_dua_format(
    lamp_data: List[Dict[str, Any]],
    split_mode: str = "user"
) -> List[Dict[str, Any]]:
    """
    将LaMP-7数据集转换为DUA-RAG格式
    
    LaMP-7: 评论写作任务
    - 输入: 用户历史评论记录
    - 输出: 个性化评论生成
    
    Args:
        lamp_data: LaMP-7原始数据
        split_mode: "user" 或 "time" 分割模式
    """
    converted = []
    
    for item in lamp_data:
        user_id = item.get("user_id", item.get("id", "unknown"))
        profile = item.get("profile", [])
        history = item.get("history", [])
        
        haystack_sessions = []
        haystack_session_ids = []
        haystack_dates = []
        
        for idx, hist_item in enumerate(history):
            session_id = f"{user_id}_session_{idx}"
            
            product_name = hist_item.get("product_name", hist_item.get("title", ""))
            review_text = hist_item.get("review", hist_item.get("text", ""))
            rating = hist_item.get("rating", hist_item.get("score", ""))
            timestamp = hist_item.get("timestamp", "")
            
            session_content = []
            if product_name or review_text:
                content_parts = []
                if product_name:
                    content_parts.append(f"产品: {product_name}")
                if rating:
                    content_parts.append(f"评分: {rating}")
                if review_text:
                    content_parts.append(f"我的评论: {review_text}")
                
                session_content.append({
                    "role": "user",
                    "content": " | ".join(content_parts)
                })
            
            if session_content:
                haystack_sessions.append(session_content)
                haystack_session_ids.append(session_id)
                haystack_dates.append(parse_lamp_timestamp(timestamp))
        
        query_item = item.get("input", "")
        if isinstance(query_item, dict):
            query_product = query_item.get("product_name", query_item.get("title", ""))
            query_text = f"请为产品「{query_product}」写一条评论。"
        else:
            query_text = str(query_item)
        
        target = item.get("output", "")
        
        converted_item = {
            "user_id": str(user_id),
            "profile_id": str(user_id),
            "question": query_text,
            "answer": str(target),
            "haystack_sessions": haystack_sessions,
            "haystack_session_ids": haystack_session_ids,
            "haystack_dates": haystack_dates,
            "question_date": haystack_dates[-1] if haystack_dates else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "profile": profile,
            "task_type": "review_writing",
            "split_mode": split_mode
        }
        
        converted.append(converted_item)
    
    return converted


def split_by_user(data: List[Dict], train_ratio: float = 0.8, seed: int = 42):
    """按用户分割数据集"""
    random.seed(seed)
    user_ids = list(set(item["user_id"] for item in data))
    random.shuffle(user_ids)
    
    split_point = int(len(user_ids) * train_ratio)
    train_users = set(user_ids[:split_point])
    
    train_data = [item for item in data if item["user_id"] in train_users]
    test_data = [item for item in data if item["user_id"] not in train_users]
    
    return train_data, test_data


def split_by_time(data: List[Dict], train_ratio: float = 0.8):
    """按时间分割数据集"""
    for item in data:
        if "question_date" not in item:
            item["question_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    sorted_data = sorted(data, key=lambda x: x["question_date"])
    split_point = int(len(sorted_data) * train_ratio)
    
    return sorted_data[:split_point], sorted_data[split_point:]


def main():
    parser = argparse.ArgumentParser(description="Convert LaMP datasets to DUA-RAG format")
    parser.add_argument("--lamp4_input", type=str, default="", help="Path to LaMP-4 raw data")
    parser.add_argument("--lamp7_input", type=str, default="", help="Path to LaMP-7 raw data")
    parser.add_argument("--output_dir", type=str, default="./data/lamp_data", help="Output directory")
    parser.add_argument("--split_mode", type=str, default="both", choices=["user", "time", "both"], help="Split mode")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.lamp4_input:
        print(f"Processing LaMP-4 data from {args.lamp4_input}")
        with open(args.lamp4_input, "r", encoding="utf-8") as f:
            lamp4_data = json.load(f)
        
        converted_lamp4 = convert_lamp4_to_dua_format(lamp4_data)
        
        if args.split_mode in ["user", "both"]:
            train_data, test_data = split_by_user(converted_lamp4, args.train_ratio, args.seed)
            with open(output_dir / "lamp_4_dev_user.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"Saved LaMP-4 user split: {len(test_data)} test samples")
        
        if args.split_mode in ["time", "both"]:
            train_data, test_data = split_by_time(converted_lamp4, args.train_ratio)
            with open(output_dir / "lamp_4_dev_time.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"Saved LaMP-4 time split: {len(test_data)} test samples")
    
    if args.lamp7_input:
        print(f"Processing LaMP-7 data from {args.lamp7_input}")
        with open(args.lamp7_input, "r", encoding="utf-8") as f:
            lamp7_data = json.load(f)
        
        converted_lamp7 = convert_lamp7_to_dua_format(lamp7_data)
        
        if args.split_mode in ["user", "both"]:
            train_data, test_data = split_by_user(converted_lamp7, args.train_ratio, args.seed)
            with open(output_dir / "lamp_7_dev_user.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"Saved LaMP-7 user split: {len(test_data)} test samples")
        
        if args.split_mode in ["time", "both"]:
            train_data, test_data = split_by_time(converted_lamp7, args.train_ratio)
            with open(output_dir / "lamp_7_dev_time.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"Saved LaMP-7 time split: {len(test_data)} test samples")
    
    print("LaMP data preprocessing completed!")


if __name__ == "__main__":
    main()
