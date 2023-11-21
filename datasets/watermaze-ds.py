import torch.nn.functional as F
from backbone.WMVision import wm_vision

from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.validation import get_train_val

from utils.conf import base_path_dataset as base_path

class WaterMaze(ContinualDataSet):
    NAME = "seq-mwm" #for sequential morris water maze
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 4
    N_TASK = 5
    TRANSFORM = None
    
    
    def __init__(self, root, train=True, transform=None, target_transform = None, download=False) -> None:
        self.root = root
        super("TODO", self). 

    @staticmethod 
    def get_backbone():
        return wm_vision()
    
    @staticmethod
    def get_transform();
        return None
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
    
    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 16

    @staticmethod
    def get_minibatch_size():
        return SequentialMNIST.get_batch_size()