from setuptools import setup, find_packages


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='lesion_coder',
    version='0.1.0',
    license="MIT",
    description='Reconstruct skin lesion images with a pre-trained auto encoder based on vgg19',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tamas Suveges',
    packages=find_packages(include=['lesion_coder']),
    install_requires = [
        'pandas',
        'tqdm',
        'matplotlib'
    ]
)
