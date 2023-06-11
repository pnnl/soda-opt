# Compile Torch model to LLVM

FILE_PATH=torchscript.py
SCRIPTS_DIR=/opt/soda/scripts
# Make sure the output directory is not empty, many files will be generated.
# ODIR=.
ODIR=output

# Edit to change the target
TARGET=$(ODIR)/04_llvm.ll

all: $(TARGET)

# MLIR file is generated with the torch_mlir python libraries. The file can be
# generated at the tosa dialect level. However, empirical observations show that
# generating the model at the linalg dialect level yields less instructions in
# the final LLVM IR.  
$(ODIR)/01_tosa.mlir: $(FILE_PATH) 
	python $(FILE_PATH) $@ --dialect=tosa

# Include the rules to generate linalg from a torch model translated to linalg
include $(SCRIPTS_DIR)/mkinc/tosa_to_llvm.mk
