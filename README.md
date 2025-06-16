# Image classification on custom Fashion MNIST dataset

This project is specifically designed for NCCU AI Business Task 3. Below is a quickstart guide:

1. Download datasets into `data` folder, which includes `fashion-mnist_test.csv` and `fashion-mnist_train.csv`.
2. Install Python dependencies: `uv sync --frozen` or `pip install -r requirements.txt`. The uv way is recommended, while the pip way may need custom installations regarding to PyTorch like `pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`
3. Use `src/train.py` to train the model or `src/generate.py` to generate predictions for the test set.

---

The whole project is a Python PyTorch uv project. It is configured to use cuda 12.2.2, python >=3.10, ubuntu 22.04 and PyTorch 2.5.1. Coding with devcontainers is supported.

## Features

- Python >=3.10
- uv latest
- CUDA 12.2.2
- Ubuntu 22.04
- Git, vim, wget, curl
- Timezone in Asia/Taipei
- Support for OpenCV windows (xcb)

## Usage

### Using VSCode for development

1. Install [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
2. Open this project in VSCode.
3. Click the `Reopen in Container` button.

### Build and run for production

Just run:

```sh
docker compose up --build
```

