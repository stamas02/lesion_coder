from lesion_coder import utils
from lesion_coder.dataset import ImageData
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
from torchvision.utils import save_image
from lesion_coder.model import BaseAutoEncoder


IMG_DIR = "ISIC_2019_Training_Input"
CSV_FILE = "ISIC_2019_Training_GroundTruth.csv"

def test(model_path, dataset_dir, image_x, image_y, dim, test_split, val_split):
    log_name = os.path.basename(model_path).split("-")[0]
    log_dir = os.path.dirname(model_path)
    device = torch.device("cuda")

    model = BaseAutoEncoder(dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    df = pd.read_csv(os.path.join(dataset_dir, CSV_FILE))
    _, test_df, _ = utils.slit_data(df, test_split, val_split)

    test_files = [os.path.join(dataset_dir, IMG_DIR, f + ".jpg") for f in test_df.image]

    test_dataset = ImageData(test_files, None, transform=utils.get_test_transform((image_x, image_y)))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    criterion = nn.MSELoss()

    worst_top_5 = utils.TopTracker(5, extrema="max")
    best_top_5 = utils.TopTracker(5, extrema="min")

    denormalize = utils.get_denormalize_transform()

    with torch.no_grad():
        for images, _ in tqdm(test_data_loader, desc="Predicting on test set"):
            images = images.to(device)
            outputs = model(images)

            loss = criterion(outputs, images)
            images = [denormalize(i) for i in images]
            outputs = [denormalize(i) for i in outputs]
            worst_top_5.add(item=torch.stack([images[0], outputs[0]]), score=loss.cpu())
            best_top_5.add(item=torch.stack([images[0], outputs[0]]), score=loss.cpu())

    viz_images = torch.cat(worst_top_5.get_top_items(), axis=0).cpu()
    viz_file = os.path.join(log_dir, log_name + "worst_top5.png")
    save_image(viz_images, viz_file, nrow=2, normalize=False)

    viz_images = torch.cat(best_top_5.get_top_items(), axis=0).cpu()
    viz_file = os.path.join(log_dir, log_name + "best_top5.png")
    save_image(viz_images, viz_file, nrow=2, normalize=False)



def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate model.')
    # Dataset Arguments
    parser.add_argument("--model-path", "-m",
                        type=str,
                        help='String Value - path to the model file.',
                        )
    parser.add_argument("--dataset-dir", "-d",
                        type=str,
                        help='String Value - The folder where the dataset is downloaded using get_dataset.py',
                        )
    parser.add_argument("--image_x", type=int,
                        default=128,
                        help="Integer Value - Width of the image that should be resized to.")
    parser.add_argument("--image_y", type=int,
                        default=128,
                        help="Integer Value - Height of the image that should be resized to.")
    parser.add_argument("--val-split", type=float,
                        default=0.1,
                        help="Floating Point Value - The percentage of data to be used for validation.")
    parser.add_argument("--test-split", type=float,
                        default=0.1,
                        help="Floating Point Value - The percentage of data to be used for test.")
    parser.add_argument("--dim", type=float,
                        default=512,
                        help="Floating Point Value - Dimensionality of the feature space.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    test(**args.__dict__)
