import torch
import argparse
import os
from lesion_coder.web_helper import download_url
from lesion_coder.utils import get_test_transform, get_denormalize_transform
from PIL import Image
from lesion_coder.model import BaseAutoEncoder
import numpy as np

MODEL_FILE = "vgg19_skin_auto_encoder.pt"
MODEL_LINK = "https://huggingface.co/stamas01/vgg19_skin_auto_encoder/resolve/main/vgg19_skin_auto_encoder.pt"
IMAGE_X = 128
IMAGE_Y = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.isfile(MODEL_FILE):
    print("Downloading pretrained model file. This will only happen at first use!")
    download_url(MODEL_LINK, MODEL_FILE)

model = BaseAutoEncoder().to(device)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()
model.to(device)
transform = get_test_transform((IMAGE_X, IMAGE_Y))
denormalize = get_denormalize_transform()


def encode(image):
    img = transform(image).to(device)
    code = model.encode(img.unsqueeze(0))[0]
    return code


def decode(code):
    reconstruction = model.decode(code.unsqueeze(0))[0]
    reconstruction = denormalize(reconstruction)
    reconstruction = torch.clamp(reconstruction, 0.0, 1.0)
    return np.rollaxis(np.uint8(reconstruction.cpu().detach().numpy()*256), 0,3)


def reconstruct(image):
    code = encode(image)
    reconstruction = decode(code)
    return reconstruction


def main(image):
    img = Image.open(image)
    rec_img = reconstruct(img)
    rec_img = Image.fromarray(rec_img)
    rec_img.save("output.jpg")
    print("result is saved to output.jpg")


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
