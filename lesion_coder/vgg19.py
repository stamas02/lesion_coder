import torch
import argparse
import os
from lesion_coder.web_helper import download_url
from lesion_coder.utils import get_test_transform
from PIL import Image
from torchvision.utils import save_image
from lesion_coder.model import BaseAutoEncoder

MODEL_FILE = "vgg19_skin_auto_encoder.pt"
MODEL_LINK = "https://huggingface.co/stamas01/vgg19_skin_auto_encoder/resolve/main/vgg19_skin_auto_encoder.pt"


def get_model():
    if not os.path.isfile(MODEL_FILE):
        print("Downloading pretrained model file. This will only happen at first use!")
        download_url(MODEL_LINK, MODEL_FILE)
    return torch.load(MODEL_FILE)


def main(image):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = get_test_transform((224,224))
    img = Image.open(image)
    img = transform(img).to(device)

    model = BaseAutoEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    outputs = model(img.unsqueeze(0))
    save_image(outputs, "output.jpg", nrow=1, normalize=True)
    print ("result is saved to output.jpg")




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
