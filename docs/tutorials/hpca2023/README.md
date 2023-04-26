# HPCA-2023 Tutorial - SODA-OPT

Part of the *SODA Synthesizer: Accelerating Data Science Applications with an end-to-end Silicon Compiler* toturial in the HPCA2023 conference: 

* [Tutorial website](https://hpc.pnl.gov/SODA/tutorials/2023/HPCA.html)


* Presentation date: Sunday, Feb 26, 2023
* Time: 8:15 AM to 12:20 PM (EST)

In this section you will learn how to use our compiler frontend to perform hardware software partitioning of high-level applications and automatic optimization of selected kernels.

# Instructions for docker users

Download this project on a machine **capable of running docker containers**

```
git clone https://gitlab.pnnl.gov/sodalite/soda-opt
```

Navigate to the turorial folder

```
cd soda-opt/docs/tutorials/hpca2023/docker-version
```

Update/download the `soda` docker image

```
docker pull agostini01/soda
```

Create a python environment with needed dependencies

```
# Option 1 - using virtualenv
#    with virtualenv, vscode current_project_root_folder must contain the .venv_hpca2023 folder
virtualenv .venv_hpca2023
source .venv_hpca2023/bin/activate
pip install tensorflow pandas lxml

# Option 2 - using conda
conda create --name hpca2023 tensorflow-cpu pandas lxml
conda activate hpca2023
```

Enter the HPCA tutorial directory with vscode or another tool capable of rendering jupyter notebooks.

```
vscode soda-opt/docs/tutorials/hpca2023/docker-version
# Open this folder with vscode
```

Open the tutorial file `tutorial.ipynb` and select the correct virtual env.
If required, **grant permissions for vscode to install any missing dependencies**.

# Instructions for users without docker access

Please follow the tutorial using the files and folders available [here](docs/tutorials/hpca2023/docker-version-executed).
<<<<<<< HEAD
=======

>>>>>>> origin/main
