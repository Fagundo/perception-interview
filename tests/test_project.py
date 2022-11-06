import os
import sys
TESTS_LOCATION = os.path.dirname(os.path.abspath(__file__))
REPO_LOCATION = os.path.dirname(TESTS_LOCATION)
sys.path.append(REPO_LOCATION)

import pandas as pd
from aim_perception import run_training, run_inference


def test_training():
    # Configure paths
    dataset_path = os.path.join(TESTS_LOCATION, 'dataset')
    results_path = os.path.join(TESTS_LOCATION, 'model_checkpoint_hold')
    model_path = os.path.join(results_path, 'model.pt')

    # Run training
    run_training.run(root_data_path=dataset_path, model_path=model_path, epochs=1, batch_size=5)

    # Get Test result path
    label_df_path = os.path.join(results_path, 'test_results.csv')
    
    # Check that paths exists
    assert os.path.exists(label_df_path)==True, 'Label results not written!'
    assert os.path.exists(model_path)==True, 'Model not generated'

    # Clean up
    os.remove(label_df_path)
    os.remove(model_path)


def test_inference():

    # Configure paths
    dataset_path = os.path.join(TESTS_LOCATION, 'dataset')
    data_path = os.path.join(dataset_path, 'data')
    ground_truth_path = os.path.join(dataset_path, 'ground_truth.csv')
    model_path = os.path.join(REPO_LOCATION, 'checkpoints', 'resnet_50', 'final.pt')
    output_path = os.path.join(data_path, 'results.csv')

    # Run training
    run_inference.infer(data_path=data_path, model_path=model_path)

    # Get results
    results_df = pd.read_csv(output_path, dtype=str).sort_values('image_name')

    # Get ground truth
    ground_truth_df = pd.read_csv(ground_truth_path, names=['image_name', 'class'], dtype=str)
    ground_truth_df = ground_truth_df.drop_duplicates().sort_values('image_name')

    # Drop indices, this causes issues
    ground_truth_df.reset_index(drop=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    # Test dataframes are the same
    assert ground_truth_df.equals(results_df)==True, 'Predicted and ground truth dataframes dont match!'

    # Clean up
    os.remove(output_path)