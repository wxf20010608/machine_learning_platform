import numpy as np
import torch

class KNNClassifier:
    def __init__(self):
        self.examples = {"A": [], "B": [], "C": []}

    def add_example(self, activation, class_id):
        # 确保activation是numpy数组
        if isinstance(activation, torch.Tensor):
            activation = activation.detach().cpu().numpy()
        self.examples[class_id].append(activation)
        return len(self.examples[class_id])

    def predict_class(self, activation):
        # 确保activation是numpy数组
        if isinstance(activation, torch.Tensor):
            activation = activation.detach().cpu().numpy()
        
        distances = {}
        for class_id, examples in self.examples.items():
            if not examples:
                continue
            distances[class_id] = np.mean([
                np.linalg.norm(np.array(act).flatten() - np.array(activation).flatten())
                for act in examples
            ])

        if not distances:
            return None, 0.0

        predicted_class = min(distances, key=distances.get)
        confidence = 1 / (1 + distances[predicted_class])
        return predicted_class, confidence 