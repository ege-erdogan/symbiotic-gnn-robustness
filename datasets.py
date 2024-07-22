import os.path as osp
import pandas as pd
import datatable as dt
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


class PygOgbnArxiv(PygNodePropPredDataset):
    def __init__(self):
        root, name, transform = "../data", "ogbn-arxiv", T.ToSparseTensor()
        master = pd.read_csv(osp.join(root, name, "ogbn-master.csv"), index_col=0)
        meta_dict = master[name]
        meta_dict["dir_path"] = osp.join(root, name)
        super().__init__(name=name, root=root, transform=transform, meta_dict=meta_dict)

    def get_idx_split(self):
        split_type = self.meta_info["split"]
        path = osp.join(self.root, "split", split_type)
        train_idx = dt.fread(osp.join(path, "train.csv"), header=False).to_numpy().T[0]
        train_idx = torch.from_numpy(train_idx).to(torch.long)
        valid_idx = dt.fread(osp.join(path, "valid.csv"), header=False).to_numpy().T[0]
        valid_idx = torch.from_numpy(valid_idx).to(torch.long)
        test_idx = dt.fread(osp.join(path, "test.csv"), header=False).to_numpy().T[0]
        test_idx = torch.from_numpy(test_idx).to(torch.long)
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}
