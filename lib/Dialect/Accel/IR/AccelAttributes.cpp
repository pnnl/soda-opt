//===- AccelAttributes.cpp - Accel attribute definitions ------------------===//


#include "soda/Dialect/Accel/IR/AccelAttributes.h"
#include "soda/Dialect/Accel/IR/Accel.h"
#include "OpcodeMapDetail.h"

// Includes for storage
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

using namespace mlir;
using namespace mlir::accel;

//===----------------------------------------------------------------------===//
// Attribute storage classes
//===----------------------------------------------------------------------===//

// namespace mlir {
// namespace accel {
// namespace detail {

// //===----------------------------------------------------------------------===//
// // OpcodeExprStorage
// //===----------------------------------------------------------------------===//

// struct OpcodeExprStorage : public StorageUniquer::BaseStorage {
//   MLIRContext *context;
//   OpcodeExprKind kind;
// };

// /// A send(id) or recv(id) expression appearing in an opcode expression.
// struct OpcodeSendRecvIdExprStorage : public OpcodeExprStorage {
//   using KeyTy = std::tuple<unsigned, unsigned>;

//   bool operator==(const KeyTy &key) const {
//     return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
//            std::get<1>(key) == id;
//   }

//   static OpcodeSendRecvIdExprStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto *result = allocator.allocate<OpcodeSendRecvIdExprStorage>();
//     result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
//     result->id = std::get<1>(key);
//     return result;
//   }

//   unsigned id;
// };

// /// A send_*(id, pos) expression appearing in an opcode expression.
// /// This supports: send_idx(id, pos), send_dim(id, pos)
// struct OpcodeSendIdPosExprStorage : public OpcodeExprStorage {
//   using KeyTy = std::tuple<unsigned, unsigned, unsigned>;

//   bool operator==(const KeyTy &key) const {
//     return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
//            std::get<1>(key) == id && std::get<2>(key) == pos;
//   }

//   static OpcodeSendIdPosExprStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto *result = allocator.allocate<OpcodeSendIdPosExprStorage>();
//     result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
//     result->id = std::get<1>(key);
//     result->pos = std::get<2>(key);
//     return result;
//   }

//   unsigned id;
//   unsigned pos;
// };

// /// A send_literal(integer_literal) expression appearing in an opcode
// /// expression. This supports: send_literal(v: i32)
// /// TODO: Make it support literals of different bitwidths
// struct OpcodeSendLiteralExprStorage : public OpcodeExprStorage {
//   using KeyTy = std::tuple<unsigned, int>;

//   bool operator==(const KeyTy &key) const {
//     return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
//            std::get<1>(key) == value;
//   }

//   static OpcodeSendLiteralExprStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto *result = allocator.allocate<OpcodeSendLiteralExprStorage>();
//     result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
//     result->value = std::get<1>(key);
//     return result;
//   }

//   int value;
// };

// /// A binary operation appearing in an affine expression.
// struct OpcodeBinaryOpExprStorage : public OpcodeExprStorage {
//   using KeyTy = std::tuple<unsigned, OpcodeExpr, OpcodeExpr>;

//   bool operator==(const KeyTy &key) const {
//     return static_cast<OpcodeExprKind>(std::get<0>(key)) == kind &&
//            std::get<1>(key) == lhs && std::get<2>(key) == rhs;
//   }

//   static OpcodeBinaryOpExprStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto *result = allocator.allocate<OpcodeBinaryOpExprStorage>();
//     result->kind = static_cast<OpcodeExprKind>(std::get<0>(key));
//     result->lhs = std::get<1>(key);
//     result->rhs = std::get<2>(key);
//     result->context = result->lhs.getContext();
//     return result;
//   }

//   OpcodeExpr lhs;
//   OpcodeExpr rhs;
// };

// /// A dimensional or symbolic identifier appearing in an affine expression.
// struct OpcodeDimExprStorage : public OpcodeExprStorage {
//   using KeyTy = std::pair<unsigned, unsigned>;

//   bool operator==(const KeyTy &key) const {
//     return kind == static_cast<OpcodeExprKind>(key.first) &&
//            position == key.second;
//   }

//   static OpcodeDimExprStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto *result = allocator.allocate<OpcodeDimExprStorage>();
//     result->kind = static_cast<OpcodeExprKind>(key.first);
//     result->position = key.second;
//     return result;
//   }

//   /// Position of this identifier in the argument list.
//   unsigned position;
// };

// /// An integer constant appearing in affine expression.
// struct OpcodeConstantExprStorage : public OpcodeExprStorage {
//   using KeyTy = int64_t;

//   bool operator==(const KeyTy &key) const { return constant == key; }

//   static OpcodeConstantExprStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto *result = allocator.allocate<OpcodeConstantExprStorage>();
//     result->kind = OpcodeExprKind::Constant;
//     result->constant = key;
//     return result;
//   }

//   // The constant.
//   int64_t constant;
// };

// //===----------------------------------------------------------------------===//
// // OpcodeListStorage
// //===----------------------------------------------------------------------===//

// using OpcodeKV = std::tuple<StringRef, OpcodeList>;
// using OpcodeDictArrayRef = ArrayRef<OpcodeKV>;

// struct OpcodeListStorage final
//     : public StorageUniquer::BaseStorage,
//       public llvm::TrailingObjects<OpcodeListStorage, OpcodeExpr> {
//   /// The hash key used for uniquing.
//   using KeyTy = std::tuple<unsigned, ArrayRef<OpcodeExpr>>;

//   unsigned numActions;

//   MLIRContext *context;

//   /// The opcode expressions for this opcode list.
//   ArrayRef<OpcodeExpr> results() const {
//     return {getTrailingObjects<OpcodeExpr>(), numActions};
//   }

//   bool operator==(const KeyTy &key) const {
//     return std::get<0>(key) == numActions && std::get<1>(key) == results();
//   }

//   // Constructs an OpcodeListStorage from a key. The context must be set by the
//   // caller.
//   static OpcodeListStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto results = std::get<1>(key);
//     auto byteSize =
//         OpcodeListStorage::totalSizeToAlloc<OpcodeExpr>(results.size());
//     auto *rawMem = allocator.allocate(byteSize, alignof(OpcodeListStorage));
//     auto *res = new (rawMem) OpcodeListStorage();
//     res->numActions = results.size();
//     std::uninitialized_copy(results.begin(), results.end(),
//                             res->getTrailingObjects<OpcodeExpr>());
//     return res;
//   }
// };

// //===----------------------------------------------------------------------===//
// // OpcodeMapStorage
// //===----------------------------------------------------------------------===//

// using OpcodeDictArrayRef = ArrayRef<OpcodeKV>;

// struct OpcodeMapStorage final
//     : public StorageUniquer::BaseStorage,
//       public llvm::TrailingObjects<OpcodeMapStorage, OpcodeKV> {
//   /// The hash key used for uniquing.
//   using KeyTy = std::tuple<unsigned, OpcodeDictArrayRef>;

//   unsigned numOpcodes; // TODO:: Using numDims as numOpcodes

//   MLIRContext *context;

//   /// The opcode expressions for this opcode map.
//   OpcodeDictArrayRef opcodes() const {
//     return {getTrailingObjects<OpcodeKV>(), numOpcodes};
//   }

//   bool operator==(const KeyTy &key) const {
//     return std::get<0>(key) == numOpcodes && std::get<1>(key) == opcodes();
//   }

//   // Constructs an OpcodeMapStorage from a key. The context must be set by the
//   // caller.
//   static OpcodeMapStorage *
//   construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
//     auto opcodes = std::get<1>(key);
//     auto byteSize =
//         OpcodeMapStorage::totalSizeToAlloc<OpcodeKV>(opcodes.size());
//     auto *rawMem = allocator.allocate(byteSize, alignof(OpcodeMapStorage));
//     auto *res = new (rawMem) OpcodeMapStorage();
//     res->numOpcodes = opcodes.size();
//     std::uninitialized_copy(opcodes.begin(), opcodes.end(),
//                             res->getTrailingObjects<OpcodeKV>());
//     return res;
//   }
// };

// } // namespace detail
// } // namespace accel
// } // namespace soda


//===----------------------------------------------------------------------===//
// OpcodeMap uniquing
//===----------------------------------------------------------------------===//


// TODO, implement ::get methods for the new attributes

// OpcodeMap OpcodeMap::get(MLIRContext *context) {
//   StorageUniquer uniquer;
//   auto *storage = uniquer.get<detail::OpcodeMapStorage>(
//       [&](detail::OpcodeMapStorage *storage) { storage->context = context; },
//       /*opcodeCount=*/0, /*opcodes=*/{});
//   return OpcodeMap(storage);
// }

//===----------------------------------------------------------------------===//
// ODS Generated Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "soda/Dialect/Accel/IR/AccelAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute Parsing
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Accel Dialect
//===----------------------------------------------------------------------===//

// Register attributes
void AccelDialect::registerAttributes() {
  // addAttributes<OpcodeMap>();
  // addAttributes<OpcodeMap, OpcodeList, OpcodeExpr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "soda/Dialect/Accel/IR/AccelAttributes.cpp.inc"
      >();
}
