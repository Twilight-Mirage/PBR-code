"""
构建LaMP数据集的冷启动原型库
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def build_lamp_prototype_bank(
    data: List[Dict[str, Any]],
    n_clusters: int = 10,
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
    min_samples_per_cluster: int = 3
) -> Dict[str, Any]:
    """
    构建LaMP数据集的原型库
    
    Args:
        data: LaMP格式数据
        n_clusters: 聚类数量
        embedding_model: 嵌入模型名称
        min_samples_per_cluster: 每个聚类的最小样本数
    """
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    
    user_profiles = defaultdict(list)
    for item in data:
        user_id = item.get("user_id", item.get("profile_id", "unknown"))
        history = item.get("haystack_sessions", [])
        profile = item.get("profile", [])
        
        history_texts = []
        for session in history:
            for turn in session:
                if turn.get("role") == "user":
                    history_texts.append(turn.get("content", ""))
        
        if history_texts:
            user_profiles[user_id].extend(history_texts)
        if profile:
            user_profiles[user_id].extend(profile if isinstance(profile, list) else [str(profile)])
    
    all_texts = []
    user_to_texts = {}
    for user_id, texts in user_profiles.items():
        combined = " ".join(texts)
        all_texts.append(combined)
        user_to_texts[user_id] = combined
    
    if len(all_texts) < n_clusters:
        n_clusters = max(1, len(all_texts) // min_samples_per_cluster)
    
    print(f"Encoding {len(all_texts)} user profiles...")
    embeddings = model.encode(all_texts, show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    cluster_centers = kmeans.cluster_centers_
    
    cluster_to_users = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        user_id = list(user_profiles.keys())[idx]
        cluster_to_users[int(label)].append(user_id)
    
    prototype_bank = {
        "n_clusters": n_clusters,
        "embedding_model": embedding_model,
        "clusters": {}
    }
    
    for cluster_id, users in cluster_to_users.items():
        cluster_embeddings = embeddings[[list(user_profiles.keys()).index(u) for u in users]]
        centroid = cluster_centers[cluster_id]
        
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        representative_idx = np.argmin(distances)
        representative_user = users[representative_idx]
        
        prototype_bank["clusters"][str(cluster_id)] = {
            "centroid_embedding": centroid.tolist(),
            "member_users": users,
            "representative_user": representative_user,
            "representative_profile": user_profiles[representative_user],
            "size": len(users)
        }
    
    return prototype_bank


def main():
    parser = argparse.ArgumentParser(description="Build prototype bank for LaMP datasets")
    parser.add_argument("--input", type=str, required=True, help="Path to LaMP processed data")
    parser.add_argument("--output", type=str, required=True, help="Output path for prototype bank")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--embedding_model", type=str, default="multi-qa-MiniLM-L6-cos-v1", help="Embedding model")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prototype_bank = build_lamp_prototype_bank(
        data=data,
        n_clusters=args.n_clusters,
        embedding_model=args.embedding_model
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prototype_bank, f, ensure_ascii=False, indent=2)
    
    print(f"Prototype bank saved to {output_path}")
    print(f"Total clusters: {prototype_bank['n_clusters']}")
    for cluster_id, cluster_info in prototype_bank["clusters"].items():
        print(f"  Cluster {cluster_id}: {cluster_info['size']} users")


if __name__ == "__main__":
    main()
