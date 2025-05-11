import torch
from torch.utils.data import Dataset
import json

class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.path = f"{args.path}/fnd/dataset/{args.dataset}"
        with open(f"{self.path}/text_entity_relation.txt", 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
        self.rationale_yes_no=args.rationale_yes_no
        self.graph_name=args.graph_name
        if not self.rationale_yes_no:
            self.prompt = '【Question】: Is this news real or fake? Answer in one word in the form of \'real\' or \'fake\'.\n\n【Answer】:'
        else:
            # generate rationale
            self.prompt = '【Task】Is this news real or fake? According to the external information provided, analyze the veracity of the given news. This is important for me. Thank you! Do not generalize that there is no hard evidence or sources of facts, do your best to find relevant evidence and facts, and then list your reasons in points.\n\n'

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        item_json = json.loads(line)
        full_text=item_json['text']
        graph = torch.load(f'{self.path}/{self.graph_name}/graphs/{index}.pt')
        assert graph.x.shape[0]==graph.num_nodes
        if not self.rationale_yes_no:
            question = f'【News】:{full_text[0:3800]}\n{self.prompt}'
        else:
            # generate rationale
            question = f'{self.prompt}【News】:{full_text[0:3600]}\n\n【Answer】:'

        return {
            'id': index,
            'label': item_json['label'],
            'graph': graph,
            'question': question,
            'len': len(question),
        }

    def get_idx_split(self):

        with open(f'{self.path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{self.path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{self.path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}