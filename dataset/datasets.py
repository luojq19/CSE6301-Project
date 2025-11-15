import torch
from torch.utils.data import Dataset, DataLoader
import json

class MultiTaskDataset(Dataset):
    def __init__(self, data, label_list, task_list):
        super().__init__()
        self.data = data
        self.label_list_per_task = label_list
        self.task_list = task_list
        self.label2idx_per_task = {
            task: {label: idx for idx, label in enumerate(self.label_list_per_task[task])}
            for task in self.task_list
        }
        self.task2idx = {task: idx for idx, task in enumerate(self.task_list)}
        self.entries = list(self.data.keys())

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        embedding = self.data[entry]['embedding']
        label_tensor_list = []
        for task in self.task_list:
            labels = self.data[entry][task]
            label_tensor = torch.zeros(len(self.label_list_per_task[task]), dtype=torch.float32)
            for label in labels:
                label_tensor[self.label2idx_per_task[task][label]] = 1
            label_tensor_list.append(label_tensor)
        return embedding, label_tensor_list
    

if __name__ == "__main__":
    data = torch.load('../data/test_merged_with_labels_filtered.pt')
    with open('../data/label_list_per_task.json', 'r') as f:
        label_list = json.load(f)
    dataset = MultiTaskDataset(
        data=data,
        label_list=label_list,
        task_list=['EC number', 'Gene3D', 'Pfam']
    )
    emb, (task1_labels, task2_labels, task3_labels) = dataset[0]
    print("Embedding shape:", emb.shape)
    print("Task 1 labels shape:", task1_labels)
    print("Task 2 labels shape:", task2_labels)
    print("Task 3 labels shape:", task3_labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        embeddings, label_tensors = batch
        print("Batch embeddings shape:", embeddings.shape)
        for i, labels in enumerate(label_tensors):
            print(f"Batch labels for task {i+1} shape:", labels.shape)
        break