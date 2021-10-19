from lesion_coder import utils
from lesion_coder.dataset import ImageData
from lesion_coder.model import BaseAutoEncoder
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import time
from torchvision.utils import save_image
from test import test

IMG_DIR = "ISIC_2019_Training_Input"
CSV_FILE = "ISIC_2019_Training_GroundTruth.csv"


def read_datasets(dataset_files):
    df = pd.DataFrame()
    for dataset_file in dataset_files:
        _df = pd.read_csv(dataset_file)
        df = pd.concat([df, _df], ignore_index=True)
    return df


def train(dataset_dir, image_x, image_y, lr, batch_size, epoch, log_dir, log_name, do_test, val_split, test_split, net):
    df = pd.read_csv(os.path.join(dataset_dir, CSV_FILE))
    train_df, _, val_df = utils.slit_data(df, test_split, val_split)

    train_files = [os.path.join(dataset_dir, IMG_DIR, f + ".jpg") for f in train_df.image]
    val_files = [os.path.join(dataset_dir, IMG_DIR, f + ".jpg") for f in val_df.image]

    train_dataset = ImageData(train_files, None, transform=utils.get_train_transform((image_x, image_y)))
    val_dataset = ImageData(val_files, None, transform=utils.get_test_transform((image_x, image_y)))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda")
    model = BaseAutoEncoder().to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    df_train_log = pd.DataFrame(columns=['epoch', 'train-loss', 'val-loss'])

    for _epoch in range(epoch):

        model.train()
        train_loss = 0
        p_bar = tqdm(train_data_loader, desc=f"Training epoch {_epoch}")
        for i, (images, _) in enumerate(p_bar):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            train_loss = train_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))
            p_bar.set_postfix({'loss': "{:.9f}".format(train_loss)})
            optimizer.step()

        model.eval()
        val_loss = 0
        p_bar = tqdm(val_data_loader, desc=f"Validation epoch {_epoch}")
        with torch.no_grad():
            for i, (images, _) in enumerate(p_bar):
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                if i == 0:
                    viz_images = torch.cat([outputs, images], axis=0).cpu()
                    viz_file = os.path.join(log_dir, log_name + f"{_epoch:03d}_viz.png")
                    save_image(viz_images, viz_file, nrow=viz_images.shape[0]//2, normalize = True)

                val_loss = val_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))

        df_train_log = df_train_log.append({'epoch': _epoch,
                                            'train-loss': train_loss,
                                            'val-loss': val_loss}, ignore_index=True)

    df_train_log.to_csv(os.path.join(log_dir, log_name + "-train_log.csv"), index=False, header=True)
    torch.save(model.state_dict(), os.path.join(log_dir, log_name + "-model.pt"))


    if do_test:
        test(model_path=os.path.join(log_dir, log_name + "-model.pt"),
            dataset_dir=dataset_dir,
            image_x=image_x,
            image_y=image_y,
            test_split = test_split,
            val_split = val_split)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate different thresholds')
    # Dataset Arguments
    parser.add_argument("--dataset-dir", "-d",
                        type=str,
                        help='String Value - The folder where the dataset is downloaded using get_dataset.py',
                        )
    parser.add_argument("--image_x", type=int,
                        default=224,
                        help="Integer Value - Width of the image that should be resized to.")
    parser.add_argument("--image_y", type=int,
                        default=224,
                        help="Integer Value - Height of the image that should be resized to.")

    # Training Arguments
    parser.add_argument("--lr", type=float,
                        default=6e-4,
                        help="Floating Point Value - Starting learning rate.")
    parser.add_argument("--batch_size", type=int,
                        default=2,
                        help="Integer Value - The sizes of the batches during training.")
    parser.add_argument("--epoch", type=int,
                        default=1,
                        help="Integer Value - Number of epoch.")
    parser.add_argument('--do-test', action='store_true',
                        help="Flag Boolean Value - If set testing will be carried out after training")
    parser.add_argument('--net', type=str,
                        help="String Value - Either test or gtp")
    parser.add_argument("--val-split", type=float,
                        default=0.2,
                        help="Floating Point Value - The percentage of data to be used for validation.")
    parser.add_argument("--test-split", type=float,
                        default=0.2,
                        help="Floating Point Value - The percentage of data to be used for test.")

    # Logging Arguments
    parser.add_argument("--log-dir", type=str,
                        help="String Value - Path to the folder the log is to be saved.")
    parser.add_argument("--log-name", type=str,
                        default="deflog",
                        help="String Value - This is a descriptive name of the method. "
                             "Will be used in legends e.g. ROC curve")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, time_str)
    os.makedirs(args.log_dir)
    train(**args.__dict__)
