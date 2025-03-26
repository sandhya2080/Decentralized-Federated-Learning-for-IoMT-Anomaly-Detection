import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import threading
import queue
import random
from cryptography.fernet import Fernet
from collections import defaultdict
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CDAE(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(CDAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class QDNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super(QDNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, parameters: np.ndarray) -> np.ndarray:
        sensitivity = 1.0
        noise_scale = np.sqrt(2 * np.log(1.25/self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, noise_scale, parameters.shape)
        return parameters + noise

class Node:
    def __init__(self, node_id: int, num_nodes: int, input_dim: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        
        # Initialize models
        self.cdae = CDAE(input_dim)
        self.qdnn = QDNN(input_dim)
        
        # Initialize privacy mechanism
        self.dp = DifferentialPrivacy()
        
        # Communication queues
        self.incoming_queue = queue.Queue()
        self.outgoing_queues = {}
        
        # Initialize encryption
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
    def preprocess_data(self, data: np.ndarray) -> torch.Tensor:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        return torch.FloatTensor(normalized_data)
    
    def train_local(self, data: np.ndarray, labels: np.ndarray, epochs: int = 10):
        processed_data = self.preprocess_data(data)
        
        # Train CDAE
        criterion_cdae = nn.MSELoss()
        optimizer_cdae = torch.optim.Adam(self.cdae.parameters())
        
        for epoch in range(epochs):
            # CDAE training
            self.cdae.train()
            optimizer_cdae.zero_grad()
            reconstructed, encoded = self.cdae(processed_data)
            loss_cdae = criterion_cdae(reconstructed, processed_data)
            loss_cdae.backward()
            optimizer_cdae.step()
            
            # Use encoded features for QDNN
            criterion_qdnn = nn.CrossEntropyLoss()
            optimizer_qdnn = torch.optim.Adam(self.qdnn.parameters())
            
            # QDNN training
            self.qdnn.train()
            optimizer_qdnn.zero_grad()
            outputs = self.qdnn(encoded.detach())
            loss_qdnn = criterion_qdnn(outputs, torch.LongTensor(labels))
            loss_qdnn.backward()
            optimizer_qdnn.step()
            
            self.metrics['cdae_loss'].append(loss_cdae.item())
            self.metrics['qdnn_loss'].append(loss_qdnn.item())
            
            logging.info(f"Node {self.node_id} - Epoch {epoch}: CDAE Loss = {loss_cdae.item():.4f}, QDNN Loss = {loss_qdnn.item():.4f}")
    
    def get_model_parameters(self) -> Dict[str, np.ndarray]:
        parameters = {
            'cdae': {name: param.data.numpy() for name, param in self.cdae.named_parameters()},
            'qdnn': {name: param.data.numpy() for name, param in self.qdnn.named_parameters()}
        }
        return parameters
    
    def update_model_parameters(self, parameters: Dict[str, Dict[str, np.ndarray]]):
        # Update CDAE parameters
        for name, param in self.cdae.named_parameters():
            if name in parameters['cdae']:
                param.data = torch.from_numpy(parameters['cdae'][name])
        
        # Update QDNN parameters
        for name, param in self.qdnn.named_parameters():
            if name in parameters['qdnn']:
                param.data = torch.from_numpy(parameters['qdnn'][name])
    
    def detect_attacks(self, data: np.ndarray) -> np.ndarray:
        self.cdae.eval()
        self.qdnn.eval()
        
        processed_data = self.preprocess_data(data)
        
        with torch.no_grad():
            _, encoded = self.cdae(processed_data)
            outputs = self.qdnn(encoded)
            predictions = torch.argmax(outputs, dim=1).numpy()
        
        return predictions
    
    def run(self):
        while True:
            try:
                # Get incoming messages
                message = self.incoming_queue.get(timeout=1)
                
                if message['type'] == 'parameters':
                    # Apply differential privacy
                    noisy_parameters = {
                        model_type: {
                            name: self.dp.add_noise(param) 
                            for name, param in model_params.items()
                        }
                        for model_type, model_params in message['parameters'].items()
                    }
                    
                    # Update local models
                    self.update_model_parameters(noisy_parameters)
                    
                    # Send updated parameters to other nodes
                    self.broadcast_parameters()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Node {self.node_id} error: {str(e)}")
    
    def broadcast_parameters(self):
        parameters = self.get_model_parameters()
        
        # Encrypt parameters
        encrypted_params = {
            model_type: {
                name: self.cipher.encrypt(param.tobytes())
                for name, param in model_params.items()
            }
            for model_type, model_params in parameters.items()
        }
        
        # Send to all other nodes
        message = {
            'type': 'parameters',
            'source_node': self.node_id,
            'parameters': encrypted_params
        }
        
        for queue in self.outgoing_queues.values():
            queue.put(message)

class FederatedSystem:
    def __init__(self, num_nodes: int, input_dim: int):
        self.nodes = []
        self.num_nodes = num_nodes
        
        # Create nodes
        for i in range(num_nodes):
            node = Node(i, num_nodes, input_dim)
            self.nodes.append(node)
        
        # Set up communication channels
        self._setup_communication()
    
    def _setup_communication(self):
        # Create fully connected topology
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    self.nodes[i].outgoing_queues[j] = self.nodes[j].incoming_queue
    
    def start(self):
        # Start all nodes in separate threads
        threads = []
        for node in self.nodes:
            thread = threading.Thread(target=node.run)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        return threads

# main.py
import logging
from data_loader import IoMTDataLoader
# from federated_system import FederatedSystem  # Previous code goes in federated_system.py

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('federated_learning.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configuration
    DATA_PATH = "your_dataset.csv"  # Put your dataset path here
    NUM_NODES = 5
    
    try:
        # Initialize data loader
        logging.info("Loading and preprocessing data...")
        data_loader = IoMTDataLoader(DATA_PATH)
        df = data_loader.load_data()
        
        # Get input dimension from data
        input_dim = df.shape[1] - 1  # Excluding label column
        
        # Preprocess data
        X, y = data_loader.preprocess_data(df)
        
        # Split data for nodes
        node_data = data_loader.split_data_for_nodes(X, y, NUM_NODES)
        
        # Initialize federated system
        logging.info("Initializing federated learning system...")
        system = FederatedSystem(NUM_NODES, input_dim)
        
        # Start the system
        threads = system.start()
        
        # Train nodes with their respective data
        for i, node in enumerate(system.nodes):
            node_X, node_y = node_data[i]
            logging.info(f"Training Node {i} with {len(node_X)} samples...")
            node.train_local(node_X, node_y)
        
        # Keep the system running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutting down the system...")
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()















    import logging
from data_loader import IoMTDataLoader
from federated_system import FederatedSystem  # Previous code goes in federated_system.py
import time

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('federated_learning.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configuration
    DATA_PATH = "iomt_dataset.csv"  # Put your dataset path here
    NUM_NODES = 5
    
    try:
        # Initialize data loader
        logging.info("Loading and preprocessing data...")
        data_loader = IoMTDataLoader(DATA_PATH)
        df = data_loader.load_data()
        
        # Get input dimension from data
        input_dim = df.shape[1] - 1  # Excluding label column
        
        # Preprocess data
        X, y = data_loader.preprocess_data(df)
        
        # Split data for nodes
        node_data = data_loader.split_data_for_nodes(X, y, NUM_NODES)
        
        # Initialize federated system
        logging.info("Initializing federated learning system...")
        system = FederatedSystem(NUM_NODES, input_dim)
        
        # Start the system
        threads = system.start()
        
        # Train nodes with their respective data
        for i, node in enumerate(system.nodes):
            node_X, node_y = node_data[i]
            logging.info(f"Training Node {i} with {len(node_X)} samples...")
            node.train_local(node_X, node_y)
        
        # Keep the system running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutting down the system...")
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()