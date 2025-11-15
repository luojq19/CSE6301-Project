import torch
from torch.utils.data import Dataset, DataLoader
import json

class MultiTaskDataset(Dataset):
    def __init__(self, data_file, label_list_file, task_list):
        super().__init__()
        self.data = torch.load(data_file)
        with open(label_list_file, 'r') as f:
            self.label_list_per_task = json.load(f)
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
        return embedding, *label_tensor_list
    

if __name__ == "__main__":
    dataset = MultiTaskDataset(
        data_file='../data/test_merged_with_labels_filtered.pt',
        label_list_file='../data/label_list_per_task.json',
        task_list=['EC number', 'Gene3D', 'Pfam']
    )
    emb, task1_labels, task2_labels, task3_labels = dataset[0]
    print("Embedding shape:", emb.shape)
    print("Task 1 labels shape:", task1_labels)
    print("Task 2 labels shape:", task2_labels)
    print("Task 3 labels shape:", task3_labels)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # for batch in dataloader:
    #     embedding = batch[0]
    #     task1_labels = batch[1]
    #     task2_labels = batch[2]
    #     print("Embedding shape:", embedding.shape)
    #     print("Task 1 labels shape:", task1_labels.shape)
    #     print("Task 2 labels shape:", task2_labels.shape)
    #     break