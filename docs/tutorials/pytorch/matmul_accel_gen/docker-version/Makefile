# Compile Torch model to LLVM

FILE_PATH=torchscript.py
SCRIPTS_DIR=/opt/soda/scripts
# SCRIPTS_DIR=../../../../../../scripts
# Make sure the output directory is not empty, many files will be generated.
# ODIR=.
ODIR=output

# Edit to change the target
# TARGET=$(ODIR)/04_llvm.ll
TARGET=$(ODIR)/04_llvm_baseline.mlir $(ODIR)/04_llvm_optimized.mlir

all: $(TARGET)

$(ODIR)/01_tosa.mlir: $(FILE_PATH) 
	python $(FILE_PATH) $@ --dialect=tosa

$(ODIR)/02_linalg.mlir: $(ODIR)/01_tosa.mlir $(SCRIPTS_DIR)/tosa_to_linalg.sh
	$(SCRIPTS_DIR)/tosa_to_linalg.sh $< $@

# Define here which operation to outline
$(ODIR)/03-01_linalg_searched.mlir: $(ODIR)/02_linalg.mlir 
	soda-opt $< -o $@ -convert-operation-to-soda="anchor-op=linalg.batch_matmul" 

$(ODIR)/03-02_linalg_outlined.mlir: $(ODIR)/03-01_linalg_searched.mlir
	soda-opt $< -o $@ -soda-outline-bambu-code -soda-extract-arguments-to-xml=using-bare-ptr
	mv forward_kernel_interface.xml  $(ODIR)/forward_kernel_interface.xml
	mv forward_kernel_test.xml $(ODIR)/forward_kernel_test.xml

$(ODIR)/03-03_linalg_isolated.mlir: $(ODIR)/03-02_linalg_outlined.mlir
	soda-opt $< -o $@ -soda-generate-bambu-accelcode=no-aa 

# =============================================================================
# Rules specific to non-optimized Baseline
$(ODIR)/04_llvm_baseline.mlir: $(ODIR)/03-03_linalg_isolated.mlir
	soda-opt $< -o $@ -lower-all-to-llvm=use-bare-ptr-memref-call-conv

$(ODIR)/05_llvm_baseline.ll: $(ODIR)/04_llvm_baseline.mlir
	mlir-translate $< -o $@ --mlir-to-llvmir -opaque-pointers=0

$(ODIR)/bambu/baseline/forward_kernel.v $(ODIR)/bambu/baseline/simulate_forward_kernel.sh : $(ODIR)/05_llvm_baseline.ll
	test -d $(ODIR)/bambu/baseline || mkdir -p $(ODIR)/bambu/baseline; \
	cd $(ODIR)/bambu/baseline; \
	bambu \
		-v3 --print-dot \
		-lm --soft-float \
		--compiler=I386_CLANG12 \
		--device=xc7z020-1clg484-VVD \
		--clock-period=5 --no-iob \
		--experimental-setup=BAMBU-BALANCED-MP \
		--channels-number=2 \
		--memory-allocation-policy=ALL_BRAM \
		--disable-function-proxy \
		--generate-tb=../../forward_kernel_test.xml \
  	--simulate --simulator=VERILATOR \
		--top-fname=forward_kernel \
		../../../$< 2>&1 | tee ../../bambu-baseline-synth-log

.PHONY: synth-baseline
synth-baseline : $(ODIR)/bambu/baseline/forward_kernel.v

# =============================================================================
# Rules specific soda-opt optimized
$(ODIR)/04_llvm_optimized.mlir: $(ODIR)/03-03_linalg_isolated.mlir
	soda-opt $< -o $@ -soda-opt-pipeline-for-bambu=use-bare-ptr-memref-call-conv

$(ODIR)/05_llvm_optimized.ll: $(ODIR)/04_llvm_optimized.mlir
	mlir-translate $< -o $@ --mlir-to-llvmir -opaque-pointers=0

$(ODIR)/bambu/optimized/forward_kernel.v $(ODIR)/bambu/optimized/simulate_forward_kernel.sh : $(ODIR)/05_llvm_optimized.ll
	test -d $(ODIR)/bambu/optimized || mkdir -p $(ODIR)/bambu/optimized; \
	cd $(ODIR)/bambu/optimized; \
	bambu \
		-v3 --print-dot \
		-lm --soft-float \
		--compiler=I386_CLANG12 \
		--device=xc7z020-1clg484-VVD \
		--clock-period=5 --no-iob \
		--experimental-setup=BAMBU-BALANCED-MP \
		--channels-number=2 \
		--memory-allocation-policy=ALL_BRAM \
		--disable-function-proxy \
		--generate-tb=../../forward_kernel_test.xml \
		--simulate --simulator=VERILATOR \
		--top-fname=forward_kernel \
		../../../$< 2>&1 | tee ../../bambu-optimized-synth-log

.PHONY: synth-optimized
synth-optimized : $(ODIR)/bambu/optimized/forward_kernel.v

.PHONY: clean
clean:
	test -f $(ODIR)/01_tosa.mlir && \
	rm -f $(ODIR)/*.mlir && \
	rm -f $(ODIR)/*.ll && \
	rm -r $(ODIR) && \
	rm -f core

.PHONY: clean-bambu
clean-bambu:
	test -f $(ODIR)/bambu/baseline/forward_kernel.v && \
	rm -rf $(ODIR)/bambu