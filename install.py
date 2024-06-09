import os
import subprocess
import urllib.request
import shutil


def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_file(url, output_path):
    if not os.path.exists(output_path):
        try:
            with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print(f"Failed to download {output_path}. Error: {e}")
            raise


def main():
    run_command('pip install -r requirements.txt')
    models_dir = os.path.abspath(os.path.join('..', '..', 'models'))

    directories = {
        'flashface': ['flashface.ckpt'],
        'vae': ['sd-v1-vae.pth'],
        'clip': ['openai-clip-vit-large-14.pth', 'bpe_simple_vocab_16e6.txt.gz'],
        'face_detection': ['retinaface_resnet50.pth']
    }

    urls = {
        'flashface.ckpt': 'https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/flashface.ckpt?download=true',
        'sd-v1-vae.pth': 'https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/sd-v1-vae.pth?download=true',
        'openai-clip-vit-large-14.pth': 'https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/openai-clip-vit-large-14.pth?download=true',
        'bpe_simple_vocab_16e6.txt.gz': 'https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/bpe_simple_vocab_16e6.txt.gz?download=true',
        'retinaface_resnet50.pth': 'https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/retinaface_resnet50.pth?download=true'
    }

    for dir_name, files in directories.items():
        dir_path = os.path.join(models_dir, dir_name)
        create_directory(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            download_file(urls[file], file_path)


if __name__ == '__main__':
    main()
