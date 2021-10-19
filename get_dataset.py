import argparse
import os
import zipfile
from src import web_helper

LINK_DATA = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
LINK_LABELS = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"


def main(output):
    output_file = os.path.join(output, "data.zip")

    print("Downloading data. Might take a while.")

    web_helper.download_url(LINK_DATA, output_file)
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        print("Extracting zip file...")
        zip_ref.extractall(output)
    os.remove(output_file)

    print("Downloading ground truth.")
    web_helper.download_url(LINK_LABELS, os.path.join(output, os.path.basename(LINK_LABELS)))


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Downloads the ISIC 2019 Challange data.')
    # Dataset Arguments
    parser.add_argument("--output", "-o",
                        type=str,
                        help='String Value - Destination path',
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    main(**args.__dict__)
