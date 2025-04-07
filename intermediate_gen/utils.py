import os 
import requests
import zipfile
import tarfile
import urllib.request
import gdown

def download_celeba():
    ''' Download CelebA dataset '''
    # Create directory if it does not exist
    if os.path.exists('data/celeba'):
        print('CelebA dataset already exists')
        return
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download CelebA dataset
    url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
    response = requests.get(url)
    with open('data/celeba.zip', 'wb') as file:
        file.write(response.content)
    print(f'Downloaded CelebA dataset to data/celeba.zip')

    # Extract CelebA dataset
    with zipfile.ZipFile('data/celeba.zip', 'r') as file:
        file.extractall('data/celeba')

    # Download metadata from the GDrive (https://drive.google.com/drive/u/2/folders/1VyAoL6KhEnL7Wf08dmYfuB-8-taUq_dB)
    import gdown

    # Define the URLs of the files
    metadata_url = "https://drive.google.com/uc?export=download&id=1pLqOPwNQk_IjY5y9rY6MSJYbfsjdUOiF"
    list_attr_celeba_url = "https://drive.google.com/uc?export=download&id=1Fwp8oMlD89V9JhcH0DMD4vZ1X1PFFwT4"

    # Define the local paths where the files will be saved
    metadata_path = "./data/celeba/metadata.csv"
    list_attr_celeba_path = "./data/celeba/list_attr_celeba.csv"

    # Download the files
    gdown.download(metadata_url, metadata_path, quiet=False)
    gdown.download(list_attr_celeba_url, list_attr_celeba_path, quiet=False)
    # Remove CelebA zip file
    os.remove('data/celeba.zip')

def download_waterbirds():
    ''' Download Waterbirds dataset using WILDs '''
    # Create directory if it does not exist
    if os.path.exists('data/waterbirds'):
        print('Waterbirds dataset already exists')
        return
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download Waterbirds dataset
    # reference from https://github.com/kohpangwei/group_DRO?tab=readme-ov-file#waterbirds
    url = 'https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz'
    response = requests.get(url)
    with open('data/waterbirds.tar.gz', 'wb') as file:
        file.write(response.content)

    # Extract Waterbirds dataset
    with tarfile.open('data/waterbirds.tar.gz', 'r:gz') as file:
        file.extractall('data/waterbirds')

    # Remove Waterbirds tar file
    os.remove('data/waterbirds.tar.gz')

def download_cifar10c():
    # Directory to save CIFAR-10C
    cifar10c_dir = './data/CIFAR-10C'

    # Create directory if not exists
    if not os.path.exists(cifar10c_dir):
        os.makedirs(cifar10c_dir)

    # Download CIFAR-10C
    url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
    if not os.path.exists(os.path.join(cifar10c_dir, 'CIFAR-10-C.tar')):
        urllib.request.urlretrieve(url, os.path.join(cifar10c_dir, 'CIFAR-10-C.tar'))

        # Extract the dataset
        with tarfile.open(os.path.join(cifar10c_dir, 'CIFAR-10-C.tar'), 'r') as tar:
            tar.extractall(path=cifar10c_dir)
    print(f'CIFAR-10C dataset is downloaded and extracted to {cifar10c_dir}')

    return cifar10c_dir

def download_cifar10():
    # Directory to save CIFAR-10
    cifar10_dir = './data/CIFAR-10'

    # Create directory if not exists
    if not os.path.exists(cifar10_dir):
        os.makedirs(cifar10_dir)

    # Download CIFAR-10
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    if not os.path.exists(os.path.join(cifar10_dir, 'CIFAR-10.tar.gz')):
        urllib.request.urlretrieve(url, os.path.join(cifar10_dir, 'CIFAR-10.tar.gz'))

        # Extract the dataset
        with tarfile.open(os.path.join(cifar10_dir, 'CIFAR-10.tar.gz'), 'r:gz') as tar:
            tar.extractall(path=cifar10_dir)
    print(f'CIFAR-10 dataset is downloaded and extracted to {cifar10_dir}')