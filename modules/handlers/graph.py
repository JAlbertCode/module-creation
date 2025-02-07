"""Handler for graph neural network models"""

import os
from typing import List, Any, Dict, Optional
import torch
import numpy as np
from .base import BaseHandler

class GraphHandler(BaseHandler):
    """Handler for graph neural network models"""
    
    TASK_TO_MODEL_CLASS = {
        "graph-classification": "AutoModelForGraphClassification",
        "node-classification": "AutoModelForNodeClassification",
        "link-prediction": "AutoModelForLinkPrediction",
        "graph-generation": "AutoModelForGraphGeneration"
    }
    
    def __init__(self, model_id: str, task: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, task, config)
        
    def generate_imports(self) -> str:
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task)
        return """
import os
import json
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from transformers import AutoProcessor, {model_class}
""".format(model_class=model_class)

    def generate_inference(self) -> str:
        if self.task == "graph-classification":
            return self._generate_classification_code()
        elif self.task == "node-classification":
            return self._generate_node_classification_code()
        elif self.task == "link-prediction":
            return self._generate_link_prediction_code()
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _generate_classification_code(self) -> str:
        return '''
def load_model():
    """Load graph classification model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForGraphClassification.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def load_graph(file_path: str) -> Data:
    """Load graph from file"""
    data = torch.load(file_path)
    if not isinstance(data, Data):
        raise ValueError("Input must be a PyG Data object")
    return data

def run_classification(
    file_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run graph classification"""
    # Load graph
    graph_data = load_graph(file_path)
    
    # Process inputs
    inputs = processor(
        graph_data,
        return_tensors="pt"
    ).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
    pred_idx = torch.argmax(probs).item()
    
    # Convert to networkx for visualization
    nx_graph = to_networkx(graph_data)
    
    # Save visualization
    viz_path = os.path.join("/outputs", "graph.png")
    plt.figure(figsize=(10, 10))
    nx.draw(nx_graph, with_labels=True)
    plt.savefig(viz_path)
    plt.close()
    
    return {
        "input_file": file_path,
        "prediction": {
            "label": model.config.id2label[pred_idx],
            "confidence": float(probs[pred_idx])
        },
        "graph_info": {
            "num_nodes": graph_data.num_nodes,
            "num_edges": graph_data.num_edges,
            "features": graph_data.num_features
        },
        "visualization": viz_path
    }

def main():
    """Main inference function"""
    file_path = os.getenv("MODEL_INPUT", "/inputs/graph.pt")
    
    model, processor = load_model()
    results = run_classification(file_path, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_node_classification_code(self) -> str:
        return '''
def load_model():
    """Load node classification model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForNodeClassification.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def run_node_classification(
    file_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run node classification"""
    # Load graph
    graph_data = load_graph(file_path)
    
    # Process inputs
    inputs = processor(
        graph_data,
        return_tensors="pt"
    ).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for each node
    node_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    node_preds = torch.argmax(node_probs, dim=-1)
    
    # Convert predictions to list
    predictions = []
    for node_idx, (pred_idx, probs) in enumerate(zip(node_preds, node_probs)):
        predictions.append({
            "node_id": int(node_idx),
            "label": model.config.id2label[int(pred_idx)],
            "confidence": float(probs[pred_idx])
        })
    
    # Create colored visualization
    nx_graph = to_networkx(graph_data)
    node_colors = [model.config.id2label[int(p)] for p in node_preds]
    
    viz_path = os.path.join("/outputs", "node_classes.png")
    plt.figure(figsize=(10, 10))
    nx.draw(nx_graph, node_color=node_colors, with_labels=True)
    plt.savefig(viz_path)
    plt.close()
    
    return {
        "input_file": file_path,
        "node_predictions": predictions,
        "graph_info": {
            "num_nodes": graph_data.num_nodes,
            "num_edges": graph_data.num_edges,
            "features": graph_data.num_features
        },
        "visualization": viz_path
    }

def main():
    """Main inference function"""
    file_path = os.getenv("MODEL_INPUT", "/inputs/graph.pt")
    
    model, processor = load_model()
    results = run_node_classification(file_path, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_link_prediction_code(self) -> str:
        return '''
def load_model():
    """Load link prediction model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForLinkPrediction.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def run_link_prediction(
    file_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run link prediction"""
    # Load graph
    graph_data = load_graph(file_path)
    
    # Process inputs
    inputs = processor(
        graph_data,
        return_tensors="pt"
    ).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted links
    edge_probs = torch.sigmoid(outputs.logits)
    pred_edges = edge_probs > 0.5
    
    # Format predictions
    predictions = []
    for i, j in zip(*pred_edges.nonzero(as_tuple=True)):
        predictions.append({
            "source": int(i),
            "target": int(j),
            "probability": float(edge_probs[i, j])
        })
    
    # Visualize predicted links
    nx_graph = to_networkx(graph_data)
    pred_edges = [(p["source"], p["target"]) for p in predictions]
    
    viz_path = os.path.join("/outputs", "predicted_links.png")
    plt.figure(figsize=(10, 10))
    nx.draw(nx_graph, with_labels=True)
    nx.draw_networkx_edges(
        nx_graph,
        pos=nx.spring_layout(nx_graph),
        edgelist=pred_edges,
        edge_color='r'
    )
    plt.savefig(viz_path)
    plt.close()
    
    return {
        "input_file": file_path,
        "link_predictions": predictions,
        "graph_info": {
            "num_nodes": graph_data.num_nodes,
            "num_edges": graph_data.num_edges,
            "num_predicted_links": len(predictions)
        },
        "visualization": viz_path
    }

def main():
    """Main inference function"""
    file_path = os.getenv("MODEL_INPUT", "/inputs/graph.pt")
    
    model, processor = load_model()
    results = run_link_prediction(file_path, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def get_requirements(self) -> List[str]:
        return [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "numpy>=1.24.0",
            "torch-geometric>=2.4.0",
            "networkx>=3.2.1",
            "matplotlib>=3.8.0"
        ]
        
    def requires_gpu(self) -> bool:
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, str):
            return False
        if not os.path.exists(input_data):
            return False
        try:
            data = torch.load(input_data)
            return isinstance(data, torch_geometric.data.Data)
        except:
            return False