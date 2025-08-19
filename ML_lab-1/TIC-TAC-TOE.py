# tictactoe_id3.py
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# ID3 Core Functions
# -------------------------------

def get_entropy_of_dataset(tensor: torch.Tensor):
    target = tensor[:, -1]
    unique_classes, counts = torch.unique(target, return_counts=True)
    probs = counts.float() / target.shape[0]
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9))  # avoid log(0)
    entropy_val = float(entropy)
    print(f"dataset_entropy: {entropy_val:.4f}")
    return entropy_val


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    total_samples = tensor.shape[0]
    values, counts = torch.unique(tensor[:, attribute], return_counts=True)

    avg_info = 0.0
    for v, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = subset.shape[0] / total_samples
        avg_info += weight * subset_entropy

    print(f"avg_info: {avg_info:.4f}")
    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = dataset_entropy - avg_info
    info_gain_val = round(float(info_gain), 4)
    print(f"information_gain: {info_gain_val:.4f}")
    return info_gain_val


def get_selected_attribute(tensor: torch.Tensor):
    num_attributes = tensor.shape[1] - 1  # exclude target
    gains = {}
    for attr in range(num_attributes):
        gains[attr] = get_information_gain(tensor, attr)
    best_attr = max(gains, key=gains.get)
    result = (gains, best_attr)
    print(result)
    return result


# -------------------------------
# Decision Tree Construction
# -------------------------------

def construct_tree(data, cols, used_attributes=None, level=0, max_depth=5):
    if used_attributes is None:
        used_attributes = set()

    if len(data) == 0:
        return None

    entropy = get_entropy_of_dataset(data)

    # Base case: pure node
    if entropy == 0:
        return int(data[0, -1].item())

    # Base case: max depth reached
    if level >= max_depth or len(used_attributes) >= len(cols) - 1:
        target_vals, counts = torch.unique(data[:, -1], return_counts=True)
        majority_class = int(target_vals[torch.argmax(counts)].item())
        return majority_class

    gain_dict, selected_attribute = get_selected_attribute(data)
    available_gains = {a: g for a, g in gain_dict.items() if a not in used_attributes}

    if not available_gains or max(available_gains.values()) <= 0:
        target_vals, counts = torch.unique(data[:, -1], return_counts=True)
        majority_class = int(target_vals[torch.argmax(counts)].item())
        return majority_class

    selected_attribute = max(available_gains, key=available_gains.get)

    tree_node = {
        'attribute': selected_attribute,
        'attribute_name': cols[selected_attribute],
        'gain': available_gains[selected_attribute],
        'level': level,
        'branches': {}
    }

    unique_values = torch.unique(data[:, selected_attribute])
    new_used_attributes = used_attributes.copy()
    new_used_attributes.add(selected_attribute)

    for value in unique_values:
        subset = data[data[:, selected_attribute] == value]
        if len(subset) == 0:
            target_vals, counts = torch.unique(data[:, -1], return_counts=True)
            majority_class = int(target_vals[torch.argmax(counts)].item())
            tree_node['branches'][int(value.item())] = majority_class
        else:
            subtree = construct_tree(subset, cols, new_used_attributes, level + 1, max_depth)
            tree_node['branches'][int(value.item())] = subtree

    return tree_node


def predict_single_sample(tree, sample):
    if isinstance(tree, int):
        return tree
    attr = tree['attribute']
    value = int(sample[attr])
    if value not in tree['branches']:
        return None
    return predict_single_sample(tree['branches'][value], sample)


def predict_batch(tree, data):
    return [predict_single_sample(tree, sample) for sample in data]


def print_tree_structure(tree, cols, level=0, prefix=""):
    if isinstance(tree, int):
        print(f"{prefix}├── Class {tree}")
        return
    attr_name = tree['attribute_name']
    gain = tree.get('gain', 0)
    if level == 0:
        print(f"Root [{attr_name}] (gain: {gain:.4f})")

    branches = tree['branches']
    for i, (value, subtree) in enumerate(branches.items()):
        branch_symbol = "└──" if i == len(branches) - 1 else "├──"
        print(f"{prefix}{branch_symbol} = {value}:")
        new_prefix = prefix + ("    " if i == len(branches) - 1 else "│   ")
        if isinstance(subtree, int):
            print(f"{new_prefix}├── Class {subtree}")
        else:
            print(f"{new_prefix}├── [{subtree['attribute_name']}] (gain: {subtree.get('gain', 0):.4f})")
            print_tree_structure(subtree, cols, level + 1, new_prefix)


# -------------------------------
# MAIN: Run on TicTacToe Dataset
# -------------------------------
if __name__ == "__main__":
    # Load data
    df = pd.read_csv(r"C:\Users\Nikhil\\Downloads\all\tictactoe.csv")
    print("Dataset shape:", df.shape)

    # Encode categorical values
    df_processed = df.copy()
    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Convert to torch tensor
    dataset = torch.tensor(df_processed.values, dtype=torch.float32)
    cols = list(df_processed.columns)

    # Train/Test Split
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    train_data = dataset[indices[:train_size]]
    test_data = dataset[indices[train_size:]]

    # Build tree
    print("\n🌳 Constructing Decision Tree...")
    tree = construct_tree(train_data, cols, max_depth=5)
    print("\nDecision Tree Structure:")
    print_tree_structure(tree, cols)

    # Predictions
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    preds = predict_batch(tree, X_test)

    # Evaluation
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print("\n📊 Performance Metrics")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
