from torch.utils.data import Dataset, DataLoader
from packages.data.data_gen import feature_scaling, data_chunker
import numpy as np
import os

class LorenzDataSet(Dataset):
    def __init__(self, parent_folder, config, scalar_dict=None, transform=None):
        self.transform = transform
        self.raw_data = self.load_data(parent_folder)
        self.config = config
        self.scalar_dict = scalar_dict
        
        if self.scalar_dict == None:
            self.scaled_data, self.scalar_dict = feature_scaling(self.raw_data)
        else:
            self.scaled_data = feature_scaling(self.raw_data, self.scalar_dict)
        
        self.features, self.targets = data_chunker(self.raw_data,
                                                   self.scaled_data,
                                                   self.config)

    def load_data(self, raw_data_path):
        data = np.load(os.path.join(raw_data_path, "raw.npy"))
        return data
    
    def __len__(self):
        return len(self.features[:, 0, 0])

    def __getitem__(self, index) -> tuple:
        sample_features = self.features[index, :, :]
        sample_targets = self.targets[index, :, :]
        sample = {"features": sample_features, "targets": sample_targets}
        return sample

