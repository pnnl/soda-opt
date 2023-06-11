# PyTorch to LLVM IR Lowering Example

In this example, we will show how to lower a simple PyTorch model to LLVM IR
using scripts from the SODA [docker image](https://hub.docker.com/r/agostini01/soda).

## Instructions for docker users

Download this project on a machine **capable of running docker containers** and
update the docker image to the latest version.


```
git clone https://gitlab.pnnl.gov/sodalite/soda-opt
docker pull agostini01/soda
```

Navigate to the turorial folder and run `make`. This will trigger execution of
scripts that will generate a simple PyTorch model, lower it TOSA dialect, and
then to LLVM IR.

```
cd soda-opt/docs/tutorials/torch/lowering/docker-version
make
```

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