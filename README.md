# lesion_coder
This is an Auto Encoder trained using torch trained to reconstruct skin lesion images.
The model is based on the vgg19 architecture and accepts 224x224 images.

##Install
Install package using the following command:

```
pip install git+https://github.com/stamas02/lesion_coder        
```

You will need to install pytorch by yourself. You can install it with the 
following command: 

```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Notice! The above command might be obsolete or may not work for CUDA versions 
other than 11.1. Please visit the official pytorch website to find command that
is specific to your system. https://pytorch.org/get-started/locally/

##Usage
Example usage:

```
import lesion_coder.vgg19 as auto_encoder
import requests

# Get a sample image from the ISIC archive
url = "blob:https://www.isic-archive.com/c0949edb-c104-4b74-b13f-80c3cb72d945"
im = Image.open(requests.get(url, stream=True).raw)

# Map the image to the feature space using the autoencoder
code = auto_encoder.encode(image)

# Map the feature back to the input image space
reconstruction = auto_encoder.decode(code)

#Show the result
reconstruction = Image.fromarray(reconstruction)
im.show(title="Original")
reconstruction.show(title="Reconstruction")
```
    
    
##Training

### Downloading the dataset
Download the ISIC 2019 Challange data:

```
get_dataset.py -o data/
```

### Model training

```
train.py --dataset-dir data/ISIC_2019/
```

Further Parameters:

option name |       Description
--- | --- 
--dataset-dir, -d |     String Value - The folder where the dataset is downloaded using get_dataset.py.
--image_x |             Integer Value - Width of the image that should be resized to.
--image_y |             Integer Value - Height of the image that should be resized to.
--lr |                  Floating Point Value - Starting learning rate.
--batch_size |          Integer Value - The sizes of the batches during training.
--epoch |               Integer Value - Number of epoch.
--val-split |           Floating Point Value - The percentage of data to be used for validation.
--test-split |          Floating Point Value - The percentage of data to be used for test.
--log-dir |             String Value - Path to the folder the log is to be saved.
--log-name |            String Value - This is a descriptive name of the current test to help you distinguish the results.

    
    