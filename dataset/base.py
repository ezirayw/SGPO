from abc import abstractmethod
from torch.utils.data import Dataset

class DiscreteData(Dataset):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def get_dim(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass