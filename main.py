import logging
from data_loader import IoMTDataLoader
from federated_system import FederatedSystem  # Previous code goes in federated_system.py
import time
import numpy as np



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
    DATA_PATH = "iomt_dataset.csv"
    NUM_NODES = 5
    MIN_SAMPLES_PER_CLASS = 2  # New parameter
    
    try:
        
        data_loader = IoMTDataLoader(DATA_PATH, min_samples_per_class=MIN_SAMPLES_PER_CLASS)
        df = data_loader.load_data()
        # Initialize data loader
        logging.info("Loading and preprocessing data...")
        data_loader = IoMTDataLoader(DATA_PATH)
        df = data_loader.load_data()
        
        # Log data dimensions
        logging.info(f"Dataset shape: {df.shape}")
        
        # Get input dimension from data
        input_dim = df.shape[1] - 1  # Excluding label column
        
        # Preprocess data
        X, y, attack_labels = data_loader.preprocess_data(df)
        num_classes = len(np.unique(y))
        logging.info(f"Number of classes: {num_classes}")
        
        # Split data for nodes
        node_data = data_loader.split_data_for_nodes(X, y, attack_labels, NUM_NODES)
        
        # Initialize federated system
        system = FederatedSystem(NUM_NODES, input_dim, num_classes)
        
        # Start the system
        threads = system.start()
        
        # Train nodes and collect results
        for i, node in enumerate(system.nodes):
            node_X, node_y, node_attack_labels = node_data[i]
            logging.info(f"\nTraining Node {i} with {len(node_X)} samples...")
            
            # Train the node
            node.train_local(node_X, node_y)
            
            # Detect attacks
            predictions, attack_counts = node.detect_attacks(node_X, node_attack_labels)
            
            # Get metrics summary
            metrics = node.get_metrics_summary()
            
            # Log comprehensive results
            logging.info(f"\nNode {i} Results Summary:")
            logging.info(f"Final Accuracy: {metrics['final_accuracy']:.2f}%")
            logging.info(f"Average Accuracy: {metrics['avg_accuracy']:.2f}%")
            logging.info(f"Maximum Accuracy: {metrics['max_accuracy']:.2f}%")
            logging.info(f"Total Attacks Detected: {metrics['total_attacks_detected']}")
            logging.info("\nAttack Distribution:")
            for attack_type, count in metrics['attack_distribution'].items():
                logging.info(f"  {attack_type}: {count}")
        
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