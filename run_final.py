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


def run(root_data_path: str, model_path: str, retrain: bool) -> None:
    ''' Function for final run of model. 
        Will either retrain or load final model with params as outlined above.
        The function will run evaluation on the test set!

    Args:
        root_data_path (str): Path to dataset
        model_path (str): Path to either load saved weights from or save weights to
        retrain (bool): If retrain, the model will train. Otherwise we will
            attempt to load the model from model_path
    '''

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

    if retrain:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    else:
        print('Retraining model ...')
        
        # Train model
        model = pipeline.train_model(
            model=model,
            optimizer=optimizer, 
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            save_path=model_path,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
        )

    # Run inference on test
    test_eval = InferenceEvaluation(model, test_loader)

    print(f'-------- TEST RESULTS --------')
    print(f'  Balanced Accuracy: {test_eval.balanced_accuracy}')
    print(f'  Classification Report: ')
    print(test_eval.classification_report)


if __name__=='__main__':

    # Parse Args
    parser = argparse.ArgumentParser(prog = 'Final Model Run')
    parser.add_argument('-d','--data_path', help='Data path', required=True)
    parser.add_argument(
        '-m','--model_path', help='Absolute path to save model to or load from', required=True
    )
    parser.add_argument('--retrain', action='store_false')

    args = parser.parse_args()

    # Run model
    run(args.data_path, args.model_path, args.retrain)

