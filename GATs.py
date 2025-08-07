import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from community import best_partition  # Louvain 社区检测
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities  # Greedy 和 LPA
from infomap import Infomap  # Infomap 社区检测


# 加载数据集
def load_data(content_path, cites_path):
    # 读取节点特征和标签
    node_ids = []  # 存储节点ID
    features = []  # 存储节点特征
    labels = []  # 存储节点标签
    labels_map = {}  # 用于存储标签字符串到整数的映射

    with open(content_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            node_id = int(parts[0])  # 节点ID
            node_ids.append(node_id)
            features.append([float(x) for x in parts[1:-1]])  # 节点特征
            label = parts[-1]  # 节点标签

            # 将文本标签映射为整数
            if label not in labels_map:
                labels_map[label] = len(labels_map)
            labels.append(labels_map[label])

    # 特征归一化
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    # 创建节点ID到索引的映射
    node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    # 读取边信息
    edge_index = []
    with open(cites_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 尝试按制表符分割，如果失败则按空格分割
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # 跳过无效行
            src, dst = map(int, parts[:2])  # 只取前两个部分作为源节点和目标节点
            # 将节点ID映射为索引
            if src in node_id_to_index and dst in node_id_to_index:
                src_index = node_id_to_index[src]
                dst_index = node_id_to_index[dst]
                edge_index.append([src_index, dst_index])
                # 如果是无向图，添加反向边
                edge_index.append([dst_index, src_index])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 创建图数据
    data = Data(x=features, edge_index=edge_index, y=labels)
    return data


# 定义GAT模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads)

    def forward(self, x, edge_index):
        # 计算节点表示和注意力权重
        x, att_weights = self.conv1(x, edge_index, return_attention_weights=True)
        return x, att_weights


# 根据注意力权重增删边
def compute_global_attention_weights(node_features):
    # 使用点积计算注意力权重
    attention_scores = torch.mm(node_features, node_features.t())  # (num_nodes, num_nodes)
    # 归一化注意力权重
    attention_weights = F.softmax(attention_scores, dim=1)
    return attention_weights


# 根据注意力权重增删边
def modify_edges(edge_index, attention_weights, add_ratio=0.001, remove_ratio=0.001):
    num_nodes = attention_weights.shape[0]
    num_edges = edge_index.shape[1]

    # 将 edge_index 转换为 NetworkX 图
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # 删除权重高的边
    num_remove = int(num_edges * remove_ratio)
    # print(f"删除 {num_remove} 条权重最高的边")

    # 获取所有边的注意力权重
    edge_attention_weights = []
    for src, dst in edge_index.t().tolist():
        edge_attention_weights.append(attention_weights[src, dst].item())

    # 按注意力权重排序边
    sorted_edge_indices = np.argsort(edge_attention_weights)[::-1]  # 从高到低排序
    removed_edges = []
    remove_count = 0
    for idx in sorted_edge_indices:
        if remove_count >= num_remove:
            break
        src, dst = edge_index[:, idx].tolist()
        if src == dst:  # 跳过自环边
            # print(f"跳过自环边 ({src}, {dst})")
            continue
        if not G.has_edge(src, dst):  # 确保边存在
            continue
        # 直接删除边，不检查图是否连通
        G.remove_edge(src, dst)
        removed_edges.append((src, dst))
        remove_count += 1
        # print(f"删除边 ({src}, {dst}), 权重: {attention_weights[src, dst]:.4f}")

    # 增加权重低的边（在所有节点中找到未连接的节点对）
    num_add = int(num_edges * add_ratio)
    # print(f"增加 {num_add} 条权重最低的边")

    # 获取所有未连接的节点对的注意力权重
    non_edge_attention_weights = []
    non_edge_pairs = []
    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):
            if not G.has_edge(src, dst):
                non_edge_attention_weights.append(attention_weights[src, dst].item())
                non_edge_pairs.append((src, dst))

    # 按注意力权重排序未连接的节点对
    sorted_non_edge_indices = np.argsort(non_edge_attention_weights)  # 从低到高排序
    added_edges = []
    for idx in sorted_non_edge_indices[:num_add]:
        src, dst = non_edge_pairs[idx]
        G.add_edge(src, dst)
        added_edges.append((src, dst))
        # print(f"添加边 ({src}, {dst}), 权重: {attention_weights[src, dst]:.4f}")

    # 打印删除和添加的边数量
    # print(f"实际删除的边数量: {len(removed_edges)}")
    # print(f"实际添加的边数量: {len(added_edges)}")

    # 将 NetworkX 图转换回 edge_index
    new_edge_index = torch.tensor(list(G.edges)).t().contiguous()

    return new_edge_index


# 使用 Louvain 算法进行社区检测
def detect_communities_louvain(edge_index, num_nodes):
    # 将 PyG 的 edge_index 转换为 NetworkX 图
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # 打印图的详细信息
    # print(f"图的节点数量: {G.number_of_nodes()}")
    # print(f"图的边数量: {G.number_of_edges()}")
    # print(f"图的连通组件数量: {nx.number_connected_components(G)}")

    # 检查图是否为空（没有边）
    if G.number_of_edges() == 0:
        # print("警告：图没有边！")
        return np.zeros(num_nodes, dtype=int)  # 返回一个全零的社区划分

    # 使用 Louvain 算法进行社区检测
    partition = best_partition(G)
    communities = np.zeros(num_nodes, dtype=int)
    for node, community in partition.items():
        communities[node] = community
    return communities


# 使用 Greedy Modularity 算法进行社区检测
def detect_communities_greedy(edge_index, num_nodes):
    # 将 PyG 的 edge_index 转换为 NetworkX 图
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # 使用 Greedy Modularity 算法进行社区检测
    communities = greedy_modularity_communities(G)
    # 将社区划分转换为节点标签
    community_labels = np.zeros(num_nodes, dtype=int)
    for i, community in enumerate(communities):
        for node in community:
            community_labels[node] = i
    return community_labels


# 使用 Label Propagation Algorithm (LPA) 进行社区检测
def detect_communities_lpa(edge_index, num_nodes):
    # 将 PyG 的 edge_index 转换为 NetworkX 图
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # 使用 LPA 算法进行社区检测
    communities = label_propagation_communities(G)
    # 将社区划分转换为节点标签
    community_labels = np.zeros(num_nodes, dtype=int)
    for i, community in enumerate(communities):
        for node in community:
            community_labels[node] = i
    return community_labels


# 使用 Infomap 算法进行社区检测
def detect_communities_infomap(edge_index, num_nodes):
    # 将 PyG 的 edge_index 转换为 NetworkX 图
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # 使用 Infomap 算法进行社区检测
    im = Infomap()
    for src, dst in G.edges():
        im.add_link(src, dst)
    im.run()

    # 将社区划分转换为节点标签
    community_labels = np.zeros(num_nodes, dtype=int)
    for node, community in im.get_modules().items():
        community_labels[node] = community
    return community_labels


# 对比社区检测结果
def compare_communities(original, new, algorithm_name):
    # 计算 NMI
    nmi = normalized_mutual_info_score(original, new)
    print(f"{algorithm_name} 标准化互信息 (NMI): {nmi:.4f}")

    # 计算 ARI
    ari = adjusted_rand_score(original, new)
    print(f"{algorithm_name} 调整兰德指数 (ARI): {ari:.4f}")


# 参数设置
out_channels = 16  # 输出特征维度
heads = 4  # 注意力头数
add_ratios = [0.01, 0.03, 0.05]  # 增加边的比例
remove_ratios = [0.01, 0.03, 0.05]  # 删除边的比例

# 数据集路径
datasets = {
    "washington": (r"washington1.content", r"washington1.cites")
}

# 遍历数据集
for dataset_name, (content_path, cites_path) in datasets.items():
    print(f"\n=== 正在处理数据集: {dataset_name} ===")

    # 加载数据集
    data = load_data(content_path, cites_path)

    # 打印数据信息
    # print(f"节点数量: {data.num_nodes}")
    print(f"边数量: {data.edge_index.shape[1] // 2}")  # 无向图，边数量需要除以2
    print(f"特征维度: {data.num_features}")
    print(f"标签数量: {len(set(data.y.tolist()))}")

    # 计算全局节点之间的注意力权重
    attention_weights = compute_global_attention_weights(data.x)

    # 第一次社区检测（使用 Louvain 算法）
    # print("\n=== 原始图 ===")
    original_communities_louvain = detect_communities_louvain(data.edge_index, num_nodes=data.num_nodes)
    # print(f"原始社区数量 (Louvain): {len(set(original_communities_louvain))}")

    # 第一次社区检测（使用 Greedy Modularity 算法）
    original_communities_greedy = detect_communities_greedy(data.edge_index, num_nodes=data.num_nodes)
    # print(f"原始社区数量 (Greedy Modularity): {len(set(original_communities_greedy))}")

    # 第一次社区检测（使用 LPA 算法）
    original_communities_lpa = detect_communities_lpa(data.edge_index, num_nodes=data.num_nodes)
    # print(f"原始社区数量 (LPA): {len(set(original_communities_lpa))}")

    # 第一次社区检测（使用 Infomap 算法）
    original_communities_infomap = detect_communities_infomap(data.edge_index, num_nodes=data.num_nodes)
    # print(f"原始社区数量 (Infomap): {len(set(original_communities_infomap))}")

    # 遍历不同的增删比例
    for add_ratio, remove_ratio in zip(add_ratios, remove_ratios):
        print(f"\n=== 增删边比例: 增加 {add_ratio * 100}%, 删除 {remove_ratio * 100}% ===")

        # 增删边
        print("\n=== 增删边操作 ===")
        new_edge_index = modify_edges(data.edge_index, attention_weights, add_ratio=add_ratio, remove_ratio=remove_ratio)

        # 第二次社区检测（使用 Louvain 算法）
        # print("\n=== 增删边后的图 ===")
        new_communities_louvain = detect_communities_louvain(new_edge_index, num_nodes=data.num_nodes)
        # print(f"新社区数量 (Louvain): {len(set(new_communities_louvain))}")

        # 第二次社区检测（使用 Greedy Modularity 算法）
        new_communities_greedy = detect_communities_greedy(new_edge_index, num_nodes=data.num_nodes)
        # print(f"新社区数量 (Greedy Modularity): {len(set(new_communities_greedy))}")

        # 第二次社区检测（使用 LPA 算法）
        new_communities_lpa = detect_communities_lpa(new_edge_index, num_nodes=data.num_nodes)
        # print(f"新社区数量 (LPA): {len(set(new_communities_lpa))}")

        # 第二次社区检测（使用 Infomap 算法）
        new_communities_infomap = detect_communities_infomap(new_edge_index, num_nodes=data.num_nodes)
        # print(f"新社区数量 (Infomap): {len(set(new_communities_infomap))}")

        # 对比社区检测结果
        print("\n=== 社区检测结果对比 ===")
        compare_communities(original_communities_louvain, new_communities_louvain, "Louvain")
        compare_communities(original_communities_greedy, new_communities_greedy, "Greedy Modularity")
        compare_communities(original_communities_lpa, new_communities_lpa, "LPA")
        compare_communities(original_communities_infomap, new_communities_infomap, "Infomap")