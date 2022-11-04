import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

class AimDataset(Dataset):

    def __init__(self, data_dir: str, ground_truth_df: pd.DataFrame, label_mapping: dict, transforms: List = []) -> None:
        self._data_dir = data_dir
        self._gt_df = ground_truth_df
        self._label_mapping = label_mapping
        self._transforms = transforms

    def get_class_weights(self) -> torch.Tensor:

        class_weights = compute_class_weight(
            'balanced', classes=list(self._label_mapping.keys()), y = self._gt_df['class'].tolist()
        )

        return torch.Tensor(class_weights)

    def __len__(self):
        return self._gt_df.shape[0]

    def __getitem__(self, idx):
        # Generate image path
        sample = self._gt_df.iloc[idx]
        file_name = sample.image_name + '.jpg'
        data_path = os.path.join(self._data_dir, file_name)

        # Load image
        image = Image.open(data_path)

        # Apply transforms
        for transform in self._transforms:
            image = transform(image)

        # Generate label
        label = self._label_mapping[sample['class']]

        return image, label

    
class AimDatasetConstructor:

    def __init__(
        self, 
        root_dir: str, 
        csv_path: str, 
        data_subdir: str, 
        transforms: List,
        train_percentage: int = 75, 
        val_percentage: int = 15, 
        test_percentage: int = 10
    ) -> None:
        self._root_dir = root_dir
        self._transforms = transforms
        self._csv_path = os.path.join(self._root_dir, csv_path)
        self._data_path = os.path.join(self._root_dir, data_subdir)

        # Check paths exists
        if not os.path.exists(self._csv_path):
            raise Exception(f'Ground truth csv path {self._csv_path} does not exist.')

        if not os.path.exists(self._data_path):
            raise Exception(f'Data path {self._data_path} does not exist.')

        # Check the distributions of data
        assert train_percentage + val_percentage + test_percentage == 100, 'Train, val and test percentages do not equal 100%!'

        # Load ground truth dataframe
        self._df = pd.read_csv(self._csv_path, names=['image_name', 'class'], dtype='str')

        # Generate split dataframes 
        self.train_df, self.val_df, self.test_df = self._get_split_dfs(
            self._df,
            val_percentage=val_percentage,
            test_percentage=test_percentage
        )

        # Create class mapping for labels
        self._label_mapping = self._get_class_mapping(self._df)

    def _get_split_dfs(
        self, 
        dataframe: pd.DataFrame, 
        val_percentage: int, 
        test_percentage: int,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame]:

        
        # Get number of samples per split
        num_samples = dataframe.shape[0]
        test_samples = int(num_samples * test_percentage / 100)
        val_samples = int(num_samples * val_percentage / 100)

        # Get index and shuffle with a preset random seed, which is hardcoded for consistency
        index_array = [i for i in range(num_samples)]
        random.Random(7).shuffle(index_array)

        # Get split indices
        test_indices = index_array[:test_samples]
        val_indices = index_array[test_samples:test_samples+val_samples]
        train_indices = index_array[test_samples+val_samples:]

        # Parse out split datframes
        train_df = dataframe.iloc[train_indices, :]
        val_df = dataframe.iloc[val_indices, :]
        test_df = dataframe.iloc[test_indices, :]

        if verbose:
            print(f'Train percent: {len(train_indices) / len(index_array) * 100}')
            print(f'Val percent: {len(val_indices) / len(index_array) * 100}')
            print(f'Test percent: {len(test_indices) / len(index_array) * 100}')

        return train_df, val_df, test_df

    def _get_class_mapping(self, dataframe: pd.DataFrame) -> dict:
        
        # Instantiate mapping
        mapping = {}

        # Get unique classes
        unique_classes = sorted(dataframe['class'].unique())
        
        for i, class_name in enumerate(unique_classes):
            
            # Generate label, set value to 1
            label = torch.zeros(len(unique_classes))
            label[i] = 1

            mapping[class_name] = label
        
        return mapping

    def _get_dataset(self, dataset_df: pd.DataFrame) -> AimDataset:
        return AimDataset(data_dir=self._data_path, ground_truth_df=dataset_df, transforms=self._transforms, label_mapping=self._label_mapping)

    def get_train(self) -> AimDataset:
        return self._get_dataset(self.train_df)

    def get_val(self) -> AimDataset:
        return self._get_dataset(self.val_df)

    def get_test(self) -> AimDataset:
        return self._get_dataset(self.test_df)

    def get_all_datasets(self) -> Tuple[AimDataset]:
        train_df = self.get_train()
        val_df = self.get_val()
        test_df = self.get_test()

        return train_df, val_df, test_df








