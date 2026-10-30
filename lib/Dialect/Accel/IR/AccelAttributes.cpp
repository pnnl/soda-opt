//===- AccelAttributes.cpp - Accel dialect attributes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "soda/Dialect/Accel/IR/Accel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::accel;

#include "soda/Dialect/Accel/IR/AccelEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "soda/Dialect/Accel/IR/AccelAttributes.cpp.inc"

void AccelDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "soda/Dialect/Accel/IR/AccelAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// OpcodeActionAttr
//===----------------------------------------------------------------------===//

OpcodeActionAttr OpcodeActionAttr::getSend(MLIRContext *ctx,
                                           int64_t operand) {
  return get(ctx, OpcodeActionKind::send, operand, 0, RecvMode::replace);
}

OpcodeActionAttr OpcodeActionAttr::getSendLiteral(MLIRContext *ctx,
                                                  int64_t literal) {
  return get(ctx, OpcodeActionKind::send_literal, 0, literal,
             RecvMode::replace);
}

OpcodeActionAttr OpcodeActionAttr::getSendDim(MLIRContext *ctx,
                                              int64_t operand, int64_t dim) {
  return get(ctx, OpcodeActionKind::send_dim, operand, dim, RecvMode::replace);
}

OpcodeActionAttr OpcodeActionAttr::getSendIdx(MLIRContext *ctx,
                                              int64_t operand, int64_t dim) {
  return get(ctx, OpcodeActionKind::send_idx, operand, dim, RecvMode::replace);
}

OpcodeActionAttr OpcodeActionAttr::getRecv(MLIRContext *ctx, int64_t operand,
                                           RecvMode mode) {
  return get(ctx, OpcodeActionKind::recv, operand, 0, mode);
}

LogicalResult
OpcodeActionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         OpcodeActionKind kind, int64_t operand, int64_t value,
                         RecvMode mode) {
  if (kind != OpcodeActionKind::send_literal && operand < 0)
    return emitError() << "operand id must be non-negative, got " << operand;
  switch (kind) {
  case OpcodeActionKind::send_literal:
    if (value < INT32_MIN || value > UINT32_MAX)
      return emitError() << "send_literal value " << value
                         << " does not fit in 32 bits";
    break;
  case OpcodeActionKind::send_dim:
  case OpcodeActionKind::send_idx:
    if (value < 0)
      return emitError() << "dimension must be non-negative, got " << value;
    break;
  case OpcodeActionKind::send:
  case OpcodeActionKind::recv:
    break;
  }
  if (kind != OpcodeActionKind::recv && mode != RecvMode::replace)
    return emitError() << "only recv actions can have a mode";
  return success();
}

FailureOr<OpcodeActionAttr> OpcodeActionAttr::parseBody(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  SMLoc loc = parser.getCurrentLocation();
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<OpcodeActionKind> kind = symbolizeOpcodeActionKind(keyword);
  if (!kind) {
    parser.emitError(loc) << "unknown opcode action '" << keyword
                          << "', expected one of: send, send_literal, "
                             "send_dim, send_idx, recv";
    return failure();
  }

  int64_t operand = 0, value = 0;
  RecvMode mode = RecvMode::replace;
  if (parser.parseLParen())
    return failure();
  switch (*kind) {
  case OpcodeActionKind::send_literal:
    if (parser.parseInteger(value))
      return failure();
    break;
  case OpcodeActionKind::send:
    if (parser.parseInteger(operand))
      return failure();
    break;
  case OpcodeActionKind::send_dim:
  case OpcodeActionKind::send_idx:
    if (parser.parseInteger(operand) || parser.parseComma() ||
        parser.parseInteger(value))
      return failure();
    break;
  case OpcodeActionKind::recv:
    if (parser.parseInteger(operand))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      SMLoc modeLoc = parser.getCurrentLocation();
      std::string modeStr;
      if (parser.parseKeywordOrString(&modeStr))
        return failure();
      std::optional<RecvMode> parsedMode = symbolizeRecvMode(modeStr);
      if (!parsedMode) {
        parser.emitError(modeLoc) << "unknown recv mode '" << modeStr
                                  << "', expected 'replace' or 'accumulate'";
        return failure();
      }
      mode = *parsedMode;
    }
    break;
  }
  if (parser.parseRParen())
    return failure();

  auto attr = getChecked([&] { return parser.emitError(loc); }, ctx, *kind,
                         operand, value, mode);
  if (!attr)
    return failure();
  return attr;
}

void OpcodeActionAttr::printBody(AsmPrinter &printer) const {
  printer << stringifyOpcodeActionKind(getKind()) << "(";
  switch (getKind()) {
  case OpcodeActionKind::send_literal:
    if (getLiteral() < 0)
      printer << getLiteral();
    else
      printer << "0x" << llvm::utohexstr(getLiteral(), /*LowerCase=*/false);
    break;
  case OpcodeActionKind::send:
    printer << getOperand();
    break;
  case OpcodeActionKind::send_dim:
  case OpcodeActionKind::send_idx:
    printer << getOperand() << ", " << getDim();
    break;
  case OpcodeActionKind::recv:
    printer << getOperand();
    if (getMode() != RecvMode::replace)
      printer << ", " << stringifyRecvMode(getMode());
    break;
  }
  printer << ")";
}

Attribute OpcodeActionAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};
  FailureOr<OpcodeActionAttr> attr = parseBody(parser);
  if (failed(attr) || parser.parseGreater())
    return {};
  return *attr;
}

void OpcodeActionAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printBody(printer);
  printer << ">";
}

//===----------------------------------------------------------------------===//
// OpcodeMapAttr
//===----------------------------------------------------------------------===//

LogicalResult
OpcodeMapAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<OpcodeMapEntry> entries) {
  if (entries.empty())
    return emitError() << "opcode map must define at least one opcode";
  llvm::SmallDenseSet<StringAttr> names;
  for (const OpcodeMapEntry &entry : entries) {
    if (!entry.name || entry.name.getValue().empty())
      return emitError() << "opcode names must be non-empty";
    if (!names.insert(entry.name).second)
      return emitError() << "opcode '" << entry.name.getValue()
                         << "' is defined more than once";
    if (!entry.actions || entry.actions.empty())
      return emitError() << "opcode '" << entry.name.getValue()
                         << "' must have at least one action";
    if (!llvm::all_of(entry.actions, llvm::IsaPred<OpcodeActionAttr>))
      return emitError() << "opcode '" << entry.name.getValue()
                         << "' actions must be #accel.action attributes";
  }
  return success();
}

ArrayAttr OpcodeMapAttr::lookup(StringRef name) const {
  for (const OpcodeMapEntry &entry : getEntries())
    if (entry.name.getValue() == name)
      return entry.actions;
  return nullptr;
}

Attribute OpcodeMapAttr::parse(AsmParser &parser, Type) {
  MLIRContext *ctx = parser.getContext();
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpcodeMapEntry> entries;

  auto parseAction = [&](SmallVectorImpl<Attribute> &actions) -> ParseResult {
    FailureOr<OpcodeActionAttr> action = OpcodeActionAttr::parseBody(parser);
    if (failed(action))
      return failure();
    actions.push_back(*action);
    return success();
  };

  auto parseEntry = [&]() -> ParseResult {
    std::string name;
    SmallVector<Attribute> actions;
    if (parser.parseKeywordOrString(&name) || parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       [&] { return parseAction(actions); }))
      return failure();
    entries.push_back(
        {StringAttr::get(ctx, name), ArrayAttr::get(ctx, actions)});
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater,
                                     parseEntry))
    return {};
  return getChecked([&] { return parser.emitError(loc); }, ctx, entries);
}

void OpcodeMapAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getEntries(), printer, [&](const OpcodeMapEntry &e) {
    printer.printKeywordOrString(e.name.getValue());
    printer << " = [";
    llvm::interleaveComma(e.actions, printer, [&](Attribute action) {
      llvm::cast<OpcodeActionAttr>(action).printBody(printer);
    });
    printer << "]";
  });
  printer << ">";
}

//===----------------------------------------------------------------------===//
// OpcodeFlowAttr
//===----------------------------------------------------------------------===//

LogicalResult
OpcodeFlowAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<OpcodeFlowLevel> levels) {
  if (levels.empty())
    return emitError() << "opcode flow must have at least one group";
  unsigned numOpcodes = 0;
  for (const OpcodeFlowLevel &level : levels) {
    if (!level.pre || !level.post)
      return emitError() << "opcode flow levels must be non-null";
    for (ArrayAttr names : {level.pre, level.post}) {
      if (!llvm::all_of(names, llvm::IsaPred<StringAttr>))
        return emitError() << "opcode flow entries must be strings";
      numOpcodes += names.size();
    }
  }
  if (!levels.back().post.empty())
    return emitError() << "the innermost flow group cannot have opcodes after "
                          "a nested group";
  if (numOpcodes == 0)
    return emitError() << "opcode flow must contain at least one opcode";
  return success();
}

void OpcodeFlowAttr::walkOpcodes(function_ref<void(StringAttr)> fn) const {
  for (const OpcodeFlowLevel &level : getLevels()) {
    for (Attribute name : level.pre)
      fn(llvm::cast<StringAttr>(name));
    for (Attribute name : level.post)
      fn(llvm::cast<StringAttr>(name));
  }
}

/// Parses the rest of a group after its `(` and appends it, then its nested
/// groups, to `levels`.
static ParseResult parseFlowGroup(AsmParser &parser,
                                  SmallVectorImpl<OpcodeFlowLevel> &levels) {
  MLIRContext *ctx = parser.getContext();
  size_t index = levels.size();
  levels.push_back({});
  SmallVector<Attribute> pre, post;
  bool hasNested = false;
  while (failed(parser.parseOptionalRParen())) {
    SMLoc loc = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalLParen())) {
      if (hasNested)
        return parser.emitError(loc)
               << "a flow group can contain at most one nested group";
      hasNested = true;
      if (parseFlowGroup(parser, levels))
        return failure();
      continue;
    }
    std::string name;
    if (parser.parseKeywordOrString(&name))
      return failure();
    (hasNested ? post : pre).push_back(StringAttr::get(ctx, name));
  }
  levels[index] = {ArrayAttr::get(ctx, pre), ArrayAttr::get(ctx, post)};
  return success();
}

Attribute OpcodeFlowAttr::parse(AsmParser &parser, Type) {
  MLIRContext *ctx = parser.getContext();
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpcodeFlowLevel> levels;
  if (parser.parseLess() || parser.parseLParen() ||
      parseFlowGroup(parser, levels) || parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, ctx, levels);
}

static void printFlowGroup(AsmPrinter &printer,
                           ArrayRef<OpcodeFlowLevel> levels) {
  auto printNames = [&](ArrayAttr names) {
    llvm::interleave(
        names,
        [&](Attribute name) {
          printer.printKeywordOrString(cast<StringAttr>(name).getValue());
        },
        [&] { printer << " "; });
  };
  const OpcodeFlowLevel &level = levels.front();
  printer << "(";
  printNames(level.pre);
  if (levels.size() > 1) {
    if (!level.pre.empty())
      printer << " ";
    printFlowGroup(printer, levels.drop_front());
    if (!level.post.empty())
      printer << " ";
    printNames(level.post);
  }
  printer << ")";
}

void OpcodeFlowAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printFlowGroup(printer, getLevels());
  printer << ">";
}
