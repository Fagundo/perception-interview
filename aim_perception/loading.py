import os
import torch
import random
import pandas as pd
from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class AimDataset(Dataset):
    ''' Dataset for the AIM Perception Vehicle Dataset 

    Attributes:
        _data_dir (str): Root directory for data, of structure:
            /
            - data/
                - <image_name>.jpg
            - ground_truth.csv
        _gt_df (pd.DataFrame): Pandas dataframe for this dataset, containing a mapping of image name and classification label
        _label_mapping (dict): A mapping of class labels (str) to one hot encoded tensor (torch.Tensor)
        _transforms (List[torchvision.transforms]): List of transforms to apply to the images
    '''
    def __init__(self, data_dir: str, ground_truth_df: pd.DataFrame, label_mapping: dict, transforms: List = []) -> None:
        '''
        Args:
            data_dir (str): Root directory for data, of structure:
                /
                - data/
                    - <image_name>.jpg
                - ground_truth.csv
            gt_df (pd.DataFrame): Pandas dataframe for this dataset, containing a mapping of image name and classification label
            label_mapping (dict): A mapping of class labels (str) to one hot encoded tensor (torch.Tensor)
            transforms (List[torchvision.transforms]): List of transforms to apply to the images        
        '''
        self._data_dir = data_dir
        self._gt_df = ground_truth_df
        self._label_mapping = label_mapping
        self._transforms = transforms

    @property
    def reverse_mapping(self) -> dict:
        ''' Property to generate reverse class mappings. Useful when mapping from predicted labels to class names.

        Returns:
            dict: Mapping of index of label in one hot encoded self._label_mapping to class name
        '''
        reverse_mapping = {}
        for k, v in self._label_mapping.items():
            class_id = torch.argmax(v).item()
            reverse_mapping[class_id] = k

        return reverse_mapping

    def get_class_weights(self) -> torch.Tensor:
        ''' Util to compute class weights. We use this in the training dataset to 
            help the model learn given the unbalanced class distribution.

        Returns:
            torch.Tensor: Torch tensor with class weights
        '''
        class_weights = compute_class_weight(
            'balanced', classes=list(self._label_mapping.keys()), y = self._gt_df['class'].tolist()
        )

        return torch.Tensor(class_weights)

    def __len__(self) -> int:
        '''Returns length of dataset'''

        return self._gt_df.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        ''' Returns an image as a tensor with transforms applied and one hot encoded label.

        Args:
            idx (int): Index of datapoint in self._gt_df
        
        Returns:
            image (torch.Tensor): Image as tensor with transforms applied
            label (torch.Tensor): One hot encoded label for the class
        '''
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

    ''' Class to construct AimDataset train, val and test splits.
    
    Attributes:
        _root_dir (str): Root directory for data, of structure:
            /
            - data/
                - <image_name>.jpg
            - ground_truth.csv
        _transforms (List[torchvision.transforms]): List of transforms to apply to the images
        _csv_path (str): Name of ground truth csv within root dir
        _data_path (str): Name of data path within root dir
        _df (pd.DataFrame): Pandas dataframe loaded from csv path
        _train_df (pd.DataFrame): Pandas dataframe with data for training split
        _val_df (pd.DataFrame): Pandas dataframe with data for validation split
        _test_df (pd.DataFrame): Pandas dataframe with data for test split
        _label_mapping (dict): A mapping of class labels (str) to one hot encoded tensor (torch.Tensor)
    '''
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
        '''
        Args:
            root_dir (str): Root directory for data, of structure:
                /
                - data/
                    - <image_name>.jpg
                - ground_truth.csv
            csv_path (str): Name of ground truth csv within root dir
            data_path (str): Name of data path within root dir
            transforms (List[torchvision.transforms]): List of transforms to apply to the images
            train_percentage (int): Percentage of data to use for training as a whole number. Default, 75.
            val_percentage (int): Percentage of data to use for validation as a whole number. Default, 15.
            test_percentage (int): Percentage of data to use for testing as a whole number. Default, 10.
        '''
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
        ''' Generate a split dataframe for train, val and test 
            given the respective data percentages

        Args:   
            dataframe (pd.DataFrame): Pandas dataframe with all ground truth file names and labels
            val_percentage (int): Percentage of data to use for validation as a whole number
            test_percentage (int): Percentage of data to use for testing as a whole number
            verbose (bool): Whether to report the actual computed split percentages. Default, True.

        Returns:
            train_df (pd.DataFrame): Pandas dataframe with data for training split
            val_df (pd.DataFrame): Pandas dataframe with data for validation split
            test_df (pd.DataFrame): Pandas dataframe with data for test split
        '''
        
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
        ''' Generates mappings from ground truth dataframe class column (str) to one hot encoded tensors.

        Args:
            dataframe (pd.DataFrame): Dataframe with class labels (class) and file names (image_name)

        Returns:
            mapping (dict): A mapping of class labels (str) to one hot encoded tensor (torch.Tensor)
        '''
        
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
        ''' Method to generalize the creation of an AimDataset split.

        Args:
            dataset_df (pd.DataFrame): Dataframe for a split
        
        Returns:
            AimDataset: AimDataset for the split
        '''
        return AimDataset(data_dir=self._data_path, ground_truth_df=dataset_df, transforms=self._transforms, label_mapping=self._label_mapping)

    def get_train(self) -> AimDataset:
        ''' Generate training dataset '''
        return self._get_dataset(self.train_df)

    def get_val(self) -> AimDataset:
        ''' Generate validation dataset '''
        return self._get_dataset(self.val_df)

    def get_test(self) -> AimDataset:
        ''' Generate test dataset '''
        return self._get_dataset(self.test_df)

    def get_all_datasets(self) -> Tuple[AimDataset]:
        ''' Method to return train, val and test AimDatasets.

        Returns:
            train (AimDataset): Training split AimDataset
            val (AimDataset): Validation split AimDataset
            test (AimDataset): Test split AimDataset
        '''
        train = self.get_train()
        val = self.get_val()
        test = self.get_test()

        return train, val, test








