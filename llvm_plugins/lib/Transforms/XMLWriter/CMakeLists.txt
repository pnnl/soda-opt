# add_llvm_library(XMLWriter MODULE
#   XMLWriter.cpp

#   PLUGIN_TOOL
#   opt
#   )

add_llvm_pass_plugin(XMLWriter
  XMLWriter.cpp
)