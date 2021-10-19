import torch
import argparse
import os
from lesion_coder.web_helper import download_url
from lesion_coder.utils import get_test_transform
from PIL import Image
import numpy as np

MODEL_FILE = "vgg19_skin_auto_encoder.pt"
MODEL_LINK = "https://huggingface.co/stamas01/vgg19_skin_auto_encoder/resolve/main/vgg19_skin_auto_encoder.pt"


def get_model():
    return torch.load("../log/20211019-142923/fist_test-model.pt")
    if not os.path.isfile(MODEL_FILE):
        print("Downloading pretrained model file. This will only happen at first use!")
        download_url(MODEL_LINK, MODEL_FILE)
    return torch.load(MODEL_FILE)


def main(image):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = get_test_transform((224,224))
    img = Image.open(image)
    img = transform(img).to(device)

    model = get_model().to(device)
    model.eval()
    outputs = model(img.unsqueeze(0))[0].detach().cpu().numpy()
    outputs = (outputs+1)//2
    reconstruction = Image.fromarray(np.uint8(np.rollaxis(outputs, 0,3)))
    reconstruction.show()

    d  =7






def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate model.')
    # Dataset Arguments
    parser.add_argument("--image", "-i",
                        type=str,
                        help='String Value - Image file to be reconstructed.',
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    main(**args.__dict__)
