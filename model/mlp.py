import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Dropout(dropout))
            in_size = hidden_size
            
        layers.append(nn.Linear(in_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
class MultiTaskMLP(nn.Module):
    def __init__(self, input_size: int, shared_hidden_sizes: List[int], task_hidden_sizes, task_output_sizes: List[int], dropout: float = 0.5):
        super(MultiTaskMLP, self).__init__()
        self.shared_mlp = MLP(input_size, shared_hidden_sizes, shared_hidden_sizes[-1], dropout)
        self.task_mlps = nn.ModuleList([
            MLP(shared_hidden_sizes[-1], task_hidden_sizes[i], output_size, dropout) for i, output_size in enumerate(task_output_sizes)
        ])
        
    def forward(self, x):
        shared_representation = self.shared_mlp(x)
        task_outputs = [task_mlp(shared_representation) for task_mlp in self.task_mlps]
        return task_outputs
    
if __name__ == "__main__":
    model = MultiTaskMLP(
        input_size=512,
        shared_hidden_sizes=[256, 128],
        task_hidden_sizes=[[], [64], [64]],
        task_output_sizes=[10, 20, 15],
        dropout=0.3
    )
    print(model)
    sample_input = torch.randn(4, 512)
    outputs = model(sample_input)
    for i, output in enumerate(outputs):
        print(f"Task {i+1} output shape: {output.shape}")