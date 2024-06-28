# PyTorch to LLVM IR Lowering Example

In this example, we will show how to lower a simple PyTorch model to LLVM IR
using scripts from the SODA [docker image](https://hub.docker.com/r/agostini01/soda).

## Instructions for docker users

Download this project on a machine **capable of running docker containers** and
update the docker image to the latest version.


```
git clone https://github.com/pnnl/soda-opt
docker pull agostini01/soda
```

Navigate to the turorial folder and run `make`. This will trigger execution of
scripts that will generate a simple PyTorch model, lower it TOSA dialect, and
then to LLVM IR.

```
cd soda-opt/docs/tutorials/torch/lowering/docker-version
docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --user $(id -u):$(id -g) agostini01/soda make
```

### Accessing the docker container shell

If you want to access the docker container shell so that you don't have to
prefix every command with 
`docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --user $(id -u):$(id -g) agostini01/soda`, 
you can run the following command:

```bash 
docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --user $(id -u):$(id -g) agostini01/soda /bin/bash 
```

This will mount the current folder into the docker container. Once inside the
container, you can run `make` to compile the tutorial.


## Artifacts

```
docker-version/
└── output
    ├── 01_tosa.mlir
    ├── 02_linalg.mlir
    ├── 02_linalg_on_tensors.mlir
    ├── 03_llvm.mlir
    └── 04_llvm.ll
```