import os
import torch
import argparse
from aim_perception import pipeline
from aim_perception.evaluation import InferenceEvaluation


# Parameters
ORIG_DATA_PATH = '/home/ubuntu/aim/vehicle_dataset'
ORIG_SAVE_PATH = '/home/ubuntu/aim/aim-perception/checkpoints/resnet_18/model.pt'

EPOCHS = 55
FINE_TUNE_EPOCHS=15
IMAGE_SIZE = 128
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-5
MODEL_NAME = 'resnet_50'
MODEL_KWARGS = dict(dropout=0.05)
SWA_KWARGS = dict(swa_start=45, swa_freq=1)


def run(root_data_path: str, model_path: str, epochs: int) -> None:
    ''' Function for train model. 

    Args:
        root_data_path (str): Path to dataset
        model_path (str): Path to either load saved weights from or save weights to
        epochs (int): Number of epochs to run
    '''

    # Assert paths exist
    assert os.path.exists(os.path.abspath(os.path.dirname(model_path))), f'Model path {model_path} does not exist!'
    assert os.path.exists(root_data_path), f'Data path {model_path} does not exist!'    

    # Load data loaders
    train_loader, val_loader, test_loader = pipeline.create_data_loaders(
        root_data_path=root_data_path, batch_size=BATCH_SIZE, image_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    # Load model and optimizer
    model, optimizer = pipeline.create_model_and_optimimzer(
        MODEL_NAME, MODEL_KWARGS, weight_decay=WEIGHT_DECAY, swa_kwargs=SWA_KWARGS
    )

    # Get device and put model on it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Train model
    print('Training model ...')
    model = pipeline.train_model(
        model=model,
        optimizer=optimizer, 
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=model_path,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
    )

    # Run inference on test
    test_eval = InferenceEvaluation(model, test_loader)

    print(f'-------- TEST SET EVALUATION --------')
    print(f'  Balanced Accuracy: {test_eval.balanced_accuracy}')
    print(f'  Classification Report: ')
    print(test_eval.classification_report)

    # Write results to csv
    csv_write_path = os.path.join(os.path.dirname(model_path), 'test_results.csv')
    df = test_eval.get_label_df()
    df.to_csv(csv_write_path, index=False)

if __name__=='__main__':

    # Parse Args
    parser = argparse.ArgumentParser(prog = 'Final Model Run')
    parser.add_argument('-d','--data_path', help='Data path', required=True)
    parser.add_argument('-e','--epochs', help='Number of epochs to run', default=EPOCHS)
    parser.add_argument(
        '-m','--model_path', help='Absolute path to save model to or load from', required=True
    )

    args = parser.parse_args()

    # Run model
    run(args.data_path, args.model_path, int(args.epochs))

