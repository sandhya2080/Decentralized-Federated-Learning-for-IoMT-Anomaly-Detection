import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import List, Tuple
import logging

class IoMTDataLoader:
    def __init__(self, filepath: str, min_samples_per_class: int = 2):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.attack_labels = []
        self.min_samples_per_class = min_samples_per_class
        
    def load_data(self) -> pd.DataFrame:
        """Load the IoMT dataset from CSV"""
        try:
            df = pd.read_csv(self.filepath)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
    def handle_imbalanced_classes(self, X: np.ndarray, y: np.ndarray, attack_labels: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Handle imbalanced classes by either upsampling or removing rare classes"""
        class_counts = Counter(y)
        logging.info(f"Initial class distribution: {class_counts}")
        
        # Identify classes with enough samples
        valid_classes = [cls for cls, count in class_counts.items() 
                        if count >= self.min_samples_per_class]
        
        if len(valid_classes) < 2:
            raise ValueError("Not enough classes with sufficient samples")
            
        # Create masks for valid classes
        mask = np.isin(y, valid_classes)
        X_filtered = X[mask]
        y_filtered = y[mask]
        attack_labels_filtered = [label for i, label in enumerate(attack_labels) if mask[i]]
        
        # Update label encoder with valid classes only
        self.label_encoder.fit(y_filtered)
        y_encoded = self.label_encoder.transform(y_filtered)
        
        logging.info(f"Final class distribution: {Counter(y_encoded)}")
        
        return X_filtered, y_encoded, attack_labels_filtered
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the data and split features/labels"""
        try:
            # Separate features and labels
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            attack_labels = y.tolist()
            
            # Handle imbalanced classes
            X, y, attack_labels = self.handle_imbalanced_classes(X, y, attack_labels)
            
            # Standardize features
            X = self.scaler.fit_transform(X)
            
            logging.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            logging.info(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
            
            return X, y, attack_labels
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def split_data_for_nodes(self, X: np.ndarray, y: np.ndarray, attack_labels: List[str], num_nodes: int) -> List[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Split data for different nodes ensuring class balance"""
        try:
            node_data = []
            
            # First split data while preserving class distribution
            X_splits = np.array_split(X, num_nodes)
            y_splits = np.array_split(y, num_nodes)
            label_splits = np.array_split(attack_labels, num_nodes)
            
            # Create data for each node
            for i in range(num_nodes):
                node_data.append((
                    X_splits[i],
                    y_splits[i],
                    label_splits[i].tolist()
                ))
                
                logging.info(f"Node {i} data shape: X={X_splits[i].shape}, y={y_splits[i].shape}")
                logging.info(f"Node {i} class distribution: {Counter(y_splits[i])}")
                
            return node_data
            
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise



# data_loader.py
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from typing import List, Tuple
# import logging


# class IoMTDataLoader:
#     def __init__(self, filepath: str):
#         self.filepath = filepath
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.attack_labels = []  # Store original attack labels
    
#     def load_data(self) -> pd.DataFrame:
#         """Load the IoMT dataset from CSV"""
#         df = pd.read_csv(self.filepath)
#         # Store original attack labels before encoding
#         self.attack_labels = df.iloc[:, -1].values.tolist()
#         logging.info(f"Attack types found: {np.unique(self.attack_labels)}")
#         return df
    
#     def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """Preprocess the data and split features/labels"""
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values
        
#         # Store original labels before encoding
#         original_labels = y.copy()
        
#         # Encode labels
#         y = self.label_encoder.fit_transform(y)
        
#         # Standardize features
#         X = self.scaler.fit_transform(X)
        
#         logging.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
#         return X, y, original_labels.tolist()
    
#     def split_data_for_nodes(self, X: np.ndarray, y: np.ndarray, attack_labels: List[str], num_nodes: int) -> List[Tuple[np.ndarray, np.ndarray, List[str]]]:
#         """Split data for different nodes"""
#         samples_per_node = len(X) // num_nodes
#         node_data = []
        
#         for i in range(num_nodes):
#             start_idx = i * samples_per_node
#             end_idx = start_idx + samples_per_node if i < num_nodes - 1 else len(X)
            
#             node_data.append((
#                 X[start_idx:end_idx],
#                 y[start_idx:end_idx],
#                 attack_labels[start_idx:end_idx]
#             ))
        
#         return node_data