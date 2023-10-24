# PyTorch MatMul to Bambu Accelerator Generation Example

In this example, we will show how transform the linalg matmul inside a simple PyTorch
model to an accelerator generated with Bambu using scripts from the SODA [docker image](https://hub.docker.com/r/agostini01/soda).

## Instructions for docker users

Download this project on a machine **capable of running docker containers** and
update the docker image to the latest version.

```
git clone https://gitlab.pnnl.gov/sodalite/soda-opt
docker pull agostini01/soda
```

Navigate to the turorial folder and run the `make` commands below. This will
trigger execution of scripts that will generate a simple PyTorch model, lower it
TOSA dialect, and then generate accelerators and simulation results.

```bash
cd soda-opt/docs/tutorials/torch/lowering/docker-version
# Currently, commands inside make are not wrapped with a docker prefix, thus they must be executed inside docker image

# Generate the source files
docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --user $(id -u):$(id -g) agostini01/soda make

# Generate baseline verilog and simulation results (may take 30sec)
docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --user $(id -u):$(id -g) agostini01/soda make synth-baseline

# Generate optimized verilog and simulation results (may take 5min)
docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) --user $(id -u):$(id -g) agostini01/soda make synth-optimized
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
container, you can run the `make` comamnds above to compile the tutorial.

## Artifacts

```
docker-version/
├── Makefile (Can be edited to generate accelerators for different targets, frequencies, etc.)
└── output
   ├── 01_tosa.mlir
   ├── 02_linalg.mlir
   ├── 02_linalg_on_tensors.mlir
   ├── 03-01_linalg_searched.mlir
   ├── 03-02_linalg_outlined.mlir
   ├── 03-03_linalg_isolated.mlir
   ├── 04_llvm_baseline.mlir
   ├── 04_llvm_optimized.mlir
   ├── 05_llvm_baseline.ll
   ├── 05_llvm_optimized.ll
   ├── bambu-baseline-synth-log (synthesis log)
   ├── bambu-optimized-synth-log (synthesis log)
   └── bambu
       ├── baseline
       |   ├── ... many files and folders ...
       |   ├── forward_kernel.v
       |   ├── results.txt
       |   ├── simulate_forward_kernel.sh
       |   └── synthesize_Synthesis_forward_kernel.sh (requires mounting vivado inside docker)
       └── optimized
           ├── ... many files and folders ...
           ├── forward_kernel.v
           ├── results.txt
           ├── simulate_forward_kernel.sh
           └── synthesize_Synthesis_forward_kernel.sh (requires mounting vivado inside docker)
```