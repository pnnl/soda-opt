#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
bool runAllocaNamerPass(Function &F) {
  if (!F.isDeclaration()) {
    int alloca_counter = 0;
    for (auto I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (isa<AllocaInst>(*I)) {
        std::string my_base = "alloca_";
        std::string my_function = std::string(F.getName());
        std::string my_ID = std::to_string(alloca_counter);
        std::string my_name = my_base + my_function + my_ID;
        I->addAnnotationMetadata(my_name);
        alloca_counter++;
        // further information can be appended with further calls to
        // addAnnotationMetadata
      }
    }
  }
  return true;
}

// struct LegacyAllocaNamerPass : public FunctionPass {
//   static char ID;
//   LegacyAllocaNamerPass() : FunctionPass(ID) {}
//   bool runOnFunction(Function &F) override { return runAllocaNamerPass(F); }
// };

struct AllocaNamerPass : public PassInfoMixin<AllocaNamerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    runAllocaNamerPass(F);
    return PreservedAnalyses::all();
  }
};
} // end of anonymous namespace

/* Legacy PM Registration */
// char LegacyAllocaNamerPass::ID = 0;
// static RegisterPass<LegacyAllocaNamerPass>
//   X("name-allocas-for-xml-gen",
//     "Assign metadata representing names for memory allocation operations.",
//     false /* Only looks at CFG */, false /* Analysis Pass */);

/* New PM Registration */
llvm::PassPluginLibraryInfo getAllocaNamerPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AllocaNamer", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "name-allocas-for-xml-gen") {
                    PM.addPass(AllocaNamerPass());
                    return true;
                  }
                  return false;
                });
          }};
}

#ifndef LLVM_ALLOCANAMER_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAllocaNamerPluginInfo();
}
#endif
