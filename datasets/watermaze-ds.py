import torch.nn.functional as F
from backbone.WMVision import wm_vision

from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val

from utils.conf import base_path_dataset as base_path


import sys
from watermaze.environment import *

def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    start_idx = setting.i * 200
    end_idx = start_idx + 200

    # Create masks based on the indices
    train_mask = (np.arange(len(train_dataset)) >= start_idx) & (np.arange(len(train_dataset)) < end_idx)
    test_mask = (np.arange(len(test_dataset)) >= start_idx) & (np.arange(len(test_dataset)) < end_idx)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


class WaterMazeDS(Dataset):
    #200 images from each environment,  
    def __init__(self, envs, N_TASK = 5, seed = 27):
        self.num_trajectories = 200 
        np.random.seed(27)
        self.data = []
        self.labels = []
        for i in np.arange(N_TASK):
            for j in np.arange(self.num_trajectories):
                d, t = envs[str(i)].get_random_observation()
                self.data.append(d)
                self.labels.append(t)

    def __getitem__(self, index:int):
        dat = self.data[index]
        lab = self.labels[index]

        return dat, lab, dat


class SequentialMWM(ContinualDataSet):
    NAME = "seq-mwm" #for sequential morris water maze
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 1 #one environment per segment of training
    N_TASK = 5
    TRANSFORM = None
    
    
    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        self.envs = {}
        for i in np.arange(self.N_TASK):
            env_name = f"{i}" #to keep 1 index names
            self.envs[env_name] = SquareMaze(size = 15, observation_size = 1000, name = f"{i}")

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

    def get_data_loaders(self):
        train_dataset = WaterMazeDS(self.envs) # add code for watermaze, perhaps need to add seed to this #TODO
        test_dataset = WaterMazeDS(self.envs)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

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
        return SequentialMWM.get_batch_size()