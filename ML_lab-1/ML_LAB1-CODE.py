import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------- Core Functions from lab_Sample_Solution.py -----------------

def get_entropy_of_dataset(tensor:torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    The last column is assumed to be the target variable.
    """
    if len(tensor) == 0:
        return 0.0

    label_column = [t[-1].item() for t in tensor]
    total_samples = len(label_column)
    unique_labels = list(set(label_column))
    
    counts = [label_column.count(x) for x in unique_labels]
    probs = torch.tensor(counts, dtype=torch.float32) / total_samples

    entropy = -torch.sum(probs * torch.log2(probs + 1e-9)).item()
    return entropy


def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
    """Return avg_info of the attribute provided as parameter"""
    if len(tensor) == 0:
        return 0.0
    
    attribute_column = tensor[:, attribute].tolist()
    unique_values = list(set(attribute_column))
    
    total_samples = len(attribute_column)
    weighted_entropy = 0.0
    
    for value in unique_values:
        subset_indices = [i for i, val in enumerate(attribute_column) if val == value]
        subset_tensor = tensor[subset_indices]
        
        weight = len(subset_tensor) / total_samples
        entropy = get_entropy_of_dataset(subset_tensor)
        
        weighted_entropy += weight * entropy
        
    return weighted_entropy


def get_information_gain(tensor:torch.Tensor, attribute:int):
    """Return Information Gain of the attribute provided as parameter"""
    entropy_dataset = get_entropy_of_dataset(tensor)
    avg_info_attribute = get_avg_info_of_attribute(tensor, attribute)
    
    information_gain = entropy_dataset - avg_info_attribute
    
    return round(information_gain, 4)


def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute
    """
    gainInfo_dictionary = {}
    
    for i in range(tensor.size(1) - 1):
        gainInfo_dictionary[i] = get_information_gain(tensor, i)
        
    if not gainInfo_dictionary:
        return ({}, -1)
    
    best_attribute = max(gainInfo_dictionary, key=gainInfo_dictionary.get)
    
    return (gainInfo_dictionary, best_attribute)

# ----------------- Decision Tree Construction and Evaluation -----------------

class DecisionTree:
    """A class to represent a Decision Tree node."""
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.children = {}

def construct_tree(data: torch.Tensor, cols: list, max_depth=10, print_construction=False, current_depth=0):
    """
    Recursively constructs the decision tree using the ID3 algorithm.
    """
    labels = data[:, -1]
    unique_labels, counts = torch.unique(labels, return_counts=True)

    if len(unique_labels) == 1:
        return DecisionTree(label=unique_labels[0].item())

    if len(cols) == 1 or current_depth >= max_depth:
        majority_label = unique_labels[torch.argmax(counts)].item()
        return DecisionTree(label=majority_label)
    
    _, best_attribute_index = get_selected_attribute(data)
    
    if best_attribute_index == -1:
        majority_label = unique_labels[torch.argmax(counts)].item()
        return DecisionTree(label=majority_label)

    best_attribute_name = cols[best_attribute_index]
    
    if print_construction:
        print(f"Depth {current_depth}: Selected attribute '{best_attribute_name}' (column {best_attribute_index})")
    
    root = DecisionTree(attribute=best_attribute_name)
    unique_values = torch.unique(data[:, best_attribute_index])

    for value in unique_values:
        subset_indices = (data[:, best_attribute_index] == value).nonzero(as_tuple=True)[0]
        subset_data = data[subset_indices]
        
        remaining_cols = cols[:best_attribute_index] + cols[best_attribute_index+1:]
        
        subset_data = torch.cat((subset_data[:, :best_attribute_index], subset_data[:, best_attribute_index+1:]), dim=1)
        
        if print_construction:
            print(f"  -> Branch for value '{value.item()}': {len(subset_data)} samples")
            
        child_node = construct_tree(subset_data, remaining_cols, max_depth, print_construction, current_depth + 1)
        root.children[value.item()] = child_node
    
    return root

def predict(tree, sample, cols):
    """
    Predicts the class of a single sample using the decision tree.
    """
    if tree.label is not None:
        return tree.label
    
    attribute_name = tree.attribute
    attribute_index = cols.index(attribute_name)
    sample_value = sample[attribute_index]
    
    if sample_value.item() in tree.children:
        return predict(tree.children[sample_value.item()], sample, cols)
    else:
        # If a value is not in the tree, return the majority class of the parent node's data.
        # This is a simple fallback.
        return list(tree.children.values())[0].label

def evaluate_decision_tree(tree, X_test, y_test, cols, class_names):
    """
    Evaluates the trained decision tree on the test data.
    """
    predictions = []
    
    # Temporarily remove the target column for prediction
    feature_cols = cols[:-1]
    
    for i in range(X_test.shape[0]):
        sample = X_test[i]
        predictions.append(predict(tree, torch.cat((sample, torch.tensor([-1.0]))), feature_cols + ['dummy_target']))
    
    # Convert predictions to a PyTorch tensor
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)

    # Convert y_test to the same type for comparison
    y_test_tensor = y_test.to(dtype=torch.float32)

    accuracy = accuracy_score(y_test_tensor.numpy(), predictions_tensor.numpy())
    
    print(f"\n✅ Evaluation Completed!")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_tensor.numpy(), predictions_tensor.numpy())
    print(cm)
    
    print("\nClassification Report:")
    report = classification_report(y_test_tensor.numpy(), predictions_tensor.numpy(), target_names=list(class_names.values()))
    print(report)
    
    return {"accuracy": accuracy, "confusion_matrix": cm.tolist(), "classification_report": report}

def print_tree_structure(tree, cols, indent=""):
    """
    Prints the structure of the decision tree.
    """
    if tree.label is not None:
        print(f"{indent}-> Leaf Node: Class {tree.label}")
        return
    
    print(f"{indent}Attribute: {tree.attribute}")
    for value, child in tree.children.items():
        print(f"{indent}  -> Value: {value}")
        print_tree_structure(child, cols, indent + "     ")

# ----------------- Main Execution Block -----------------

if __name__ == '__main__':
    # Define a simple argparse for demonstration purposes
    # In a real scenario, this would be handled by the test.py script
    class Args:
        def __init__(self, data, framework='pytorch', print_tree=True, print_construction=False):
            self.data = data
            self.framework = framework
            self.print_tree = print_tree
            self.print_construction = print_construction
    
    # Change the data file path to run on a different dataset
    DATA_FILE = r"C:\Users\nikhi\Downloads\UNIVERSITY\ML_lab\ML_lab-1\all\nursery.csv" 
    
    try:
        # Load the dataset
        df = pd.read_csv(DATA_FILE)
        
        # Preprocessing: Convert categorical data to numerical using LabelEncoder
        label_encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
        
        # Convert DataFrame to PyTorch tensor
        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        
        # Split data into training and testing sets
        train_data_tensor, test_data_tensor = train_test_split(data_tensor, test_size=0.3, random_state=42)
        
        cols = list(df.columns)
        
        # Construct the decision tree
        print("Starting decision tree construction...")
        tree = construct_tree(train_data_tensor, cols=cols, max_depth=7, print_construction=False)
        
        if tree is not None:
            print(f"🌳 Decision tree construction completed!")
            
            # Print tree structure
            print("\n🌲 DECISION TREE STRUCTURE")
            print("="*60)
            print_tree_structure(tree, cols)
            
            # Prepare test data
            X_test = test_data_tensor[:, :-1]
            y_test = test_data_tensor[:, -1]
            
            # Get class names for reporting
            target_col = cols[-1]
            le = label_encoders[target_col]
            class_names = {i: le.inverse_transform([i])[0] for i in range(len(le.classes_))}
            
            # Evaluate the tree
            evaluation_results = evaluate_decision_tree(tree, X_test, y_test, cols, class_names)
            
        else:
            print("❌ Failed to construct decision tree!")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")