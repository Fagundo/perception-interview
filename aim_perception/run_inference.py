import os
import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
from aim_perception import ModelFactory
from aim_perception.pipeline import IMAGE_MEAN, IMAGE_STD

# Creating class mapping here, doesnt get used anywhere else
CLASS_MAPPING = {
    0: 'articulated_truck',
    1: 'bicycle',
    2: 'bus',
    3: 'car',
    4: 'motorcycle',
    5: 'non-motorized_vehicle',
    6: 'pedestrian',
    7: 'pickup_truck',
    8: 'single_unit_truck',
    9: 'work_van'
}

def infer(data_path: str):

    # Assert data path exists
    assert os.path.exists(data_path), f'Data path {data_path} does not exist!'

    # Lets get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create data transforms
    data_transforms = [
        transforms.ToTensor(),
        transforms.Resize(size=(128, 128)),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ]

    # Load model and weighs
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'checkpoints', 'resnet_50', 'final.pt'
    )
    model = ModelFactory.get_model('resnet_50', dropout=0.05)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )

    # Configure for evaluation
    model.eval()
    model.to(device)

    # Iterate over images and run inference
    output = []
    for file_name in os.listdir(data_path):
        try:
            # Transform input
            file_path = os.path.join(data_path, file_name)
            image_name, _ = os.path.splitext(file_name)

            input = Image.open(file_path)
            for transform in data_transforms:
                input = transform(input)

            # Unsqueeze and send to device
            input = input.unsqueeze(0).to(device)

            # Get probabilities
            prob = model(input)

            # Get class
            image_class = torch.argmax(prob)
            image_label = CLASS_MAPPING[int(image_class)]

            output.append([image_name, image_label])

        except Exception as e:
            print(f'Encountered error: {e}')

    # Create csv
    output = pd.DataFrame(output, columns=['image_name', 'class'], dtype=str)
    
    # Write csv 
    output.to_csv(os.path.join(data_path, 'results.csv'), index=False)


if __name__=='__main__':

    # Parse Args
    parser = argparse.ArgumentParser(prog = 'Model inference')
    parser.add_argument('-d','--data_path', help='Data path', required=True)
    args = parser.parse_args()

    # Run model
    infer(args.data_path)
