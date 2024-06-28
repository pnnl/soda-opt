# Tensorflow Tutorial - SODA-OPT

In this section you will learn how to use our compiler frontend to perform
hardware software partitioning of high-level applications and automatic
optimization of selected kernels.


# Instructions for docker users

Download this project on a machine **capable of running docker containers**

```
git clone https://gitlab.pnnl.gov/sodalite/soda-opt
```

Navigate to the turorial folder

```
cd soda-opt/docs/tutorials/tensorflow/docker-version
```

Update/download the `soda` docker image

```
docker pull agostini01/soda
```

Create a python environment with needed dependencies

```bash
# Option 1 - using virtualenv
#    with virtualenv, vscode current_project_root_folder must contain the .venv_soda folder
virtualenv .venv_soda
source .venv_soda/bin/activate
pip install tensorflow-cpu pandas lxml ipykernel

# Option 2 - using conda
conda create --name soda tensorflow-cpu pandas lxml
conda activate soda
```

Enter the HPCA tutorial directory with vscode or another tool capable of rendering jupyter notebooks.

```bash
# Open this folder with vscode
code soda-opt/docs/tutorials/tensorflow/docker-version
```

Open the tutorial file `tutorial.ipynb` and select the correct virtual env.
If required, **grant permissions for vscode to install any missing dependencies** and extensions.

- Python vscode extension: `ms-python.python`
- Jupyter vscode extension: `ms-toolsai.jupyter`
- MLIR language extension: `llvm-vs-code-extensions.vscode-mlir`
- Dot graphviz extension: `tintinweb.graphviz-interactive-preview`
    - Open `.dot` file and click on "dot" at the top right of the editor.


## Notes

- Tested with python==3.8 and tensorflow-cpu==2.13.1


# Instructions for users without docker access

Please follow the tutorial using the files and folders available [here](docs/tutorials/tensorflow/docker-version-executed).
