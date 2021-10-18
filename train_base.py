from src import utils
from src.dataset import ImageData
from src.model import GTPClassifier, TESTClassifier
from test import test
import pandas as pd
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
from src.utils import class_decoder_loss, decoder_loss
from sklearn.metrics import accuracy_score
from torchvision.utils import save_image
from einops import rearrange
from torchvision import transforms



DIR_TRAINING_DATA = "ISIC-2017_Training_Data"
FILE_TRAINING_LABELS = "ISIC-2017_Training_Part3_GroundTruth.csv"
DIR_VALIDATION_DATA = "ISIC-2017_Validation_Data"
FILE_VALIDATION_LABELS = "ISIC-2017_Validation_Part3_GroundTruth.csv"


def read_datasets(dataset_files):
    df = pd.DataFrame()
    for dataset_file in dataset_files:
        _df = pd.read_csv(dataset_file)
        df = pd.concat([df, _df], ignore_index=True)
    return df



def train(dataset_dir, image_x, image_y, lr, batch_size, epoch, log_dir, log_name, do_test, net):
    train_df = pd.read_csv(os.path.join(dataset_dir, FILE_TRAINING_LABELS))
    val_df = pd.read_csv(os.path.join(dataset_dir, FILE_VALIDATION_LABELS))

    train_files = [os.path.join(dataset_dir, DIR_TRAINING_DATA, f + ".jpg") for f in train_df.image_id]
    val_files = [os.path.join(dataset_dir, DIR_VALIDATION_DATA, f + ".jpg") for f in val_df.image_id]

    train_labels = np.array((-train_df.melanoma * 2) - (train_df.seborrheic_keratosis) + 2, dtype=int)
    val_labels = np.array((-val_df.melanoma * 2) - (val_df.seborrheic_keratosis) + 2, dtype=int)

    train_dataset = ImageData(train_files, train_labels, transform=utils.get_train_transform((image_x, image_y)))
    val_dataset = ImageData(val_files, val_labels, transform=utils.get_test_transform((image_x, image_y)))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda")
    if net == "test":
        model = TESTClassifier(num_classes=1, num_channels=1).to(device)
    else:
        model = GTPClassifier(num_classes=1).to(device)


    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)



    df_train_log = pd.DataFrame(columns=['epoch', 'train-loss', 'val-loss'])
    to_grey = transforms.Grayscale(num_output_channels=1)
    for _epoch in range(epoch):


        model.train()
        train_loss = 0
        train_acc = 0
        p_bar = tqdm(train_data_loader, desc=f"Training epoch {_epoch}")
        for i, (images, labels, _) in enumerate(p_bar):
            images = images.to(device)
            images = torch.stack([to_grey(i) for i in images])
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, predictions = decoder_loss(outputs, images, labels)
            acc = accuracy_score(predictions, labels.detach().cpu().numpy())
            loss.backward()
            train_loss = train_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))
            train_acc = train_acc * (1 - (1 / (i + 1))) + acc * (1 / (i + 1))
            p_bar.set_postfix({'loss': train_loss, 'acc': train_acc})
            optimizer.step()

        model.eval()
        val_loss = 0
        train_acc = 0
        p_bar = tqdm(val_data_loader, desc=f"Validation epoch {_epoch}")
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(p_bar):
                images = images.to(device)
                images = torch.stack([to_grey(i) for i in images])
                labels = labels.to(device)
                outputs = model(images)
                loss, predictions = decoder_loss(outputs, images, labels)
                if i == 0:
                    outputs = rearrange(outputs, 'k b c h w -> b k c h w')
                    outputs = torch.cat((images[:, None, :], outputs), 1)
                    outputs = outputs.reshape(-1, *outputs.size()[2::])
                    transform = transforms.Normalize(mean=[-0.5/0.5],
                                                     std=[1/0.5])
                    outputs = ([transform(t).cpu() for t in outputs])
                    save_image(outputs, os.path.join(log_dir, log_name + f"{_epoch}_rec.png"), nrow=4)

                acc = accuracy_score(predictions, labels.detach().cpu().numpy())
                val_loss = val_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))
                train_acc = train_acc * (1 - (1 / (i + 1))) + acc * (1 / (i + 1))



        df_train_log = df_train_log.append({'epoch': _epoch,
                                            'train-loss': train_loss,
                                            'val-loss': val_loss}, ignore_index=True)

    df_train_log.to_csv(os.path.join(log_dir, log_name + "-train_log.csv"), index=False, header=True)
    torch.save(model, os.path.join(log_dir, log_name + "-model.pt"))

    if do_test:
        test(model_path=os.path.join(log_dir, log_name + "-model.pt"),
             dataset_dir=dataset_dir,
             batch_size=batch_size,
             image_x=image_x,
             image_y=image_y)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate different thresholds')
    # Dataset Arguments
    parser.add_argument("--dataset-dir", "-d",
                        type=str,
                        help='String Value - The folder where the dataset is downloaded using get_dataset.py',
                        )
    parser.add_argument("--image_x", type=int,
                        default=256,
                        help="Integer Value - Width of the image that should be resized to.")
    parser.add_argument("--image_y", type=int,
                        default=256,
                        help="Integer Value - Height of the image that should be resized to.")

    # Training Arguments
    parser.add_argument("--lr", type=float,
                        default=0.002,
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
