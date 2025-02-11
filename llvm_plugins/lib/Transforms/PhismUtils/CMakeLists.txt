# TODO: It appears that packing passes in a single library is not supported
# by the building infrastructure and the passes from the second file do not
# get exposed to the commandline.
# This must be investigated further.

add_llvm_library(VhlsLLVMRewriter MODULE
  VhlsLLVMRewriter.cc
  Utils.cc

  PLUGIN_TOOL
  opt
)

add_llvm_library(MemRefToArray MODULE
  MemRefToArray.cc
  Utils.cc

  PLUGIN_TOOL
  opt
)
  