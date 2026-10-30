# AXI4MLIR in soda-opt

soda-opt implements the AXI4MLIR flow (CGO 2024): host code for custom
accelerators connected through an AXI DMA engine is generated from attributes
on linalg operations. The attributes say which opcodes the accelerator
understands and in which loop of the tiled loop nest each one is sent.

```
linalg op + accel.* attributes
  --convert-linalg-to-accel        tiled loop nest of accel ops
  --convert-accel-to-axi-runtime   calls into the AXI runtime library
  (standard lowering to LLVM)
  mlir-runner -shared-libs=libsoda_axi_runtime_<backend>.so
```

## Attributes

They are attached to any linalg op (generic or named) as discardable
attributes, and verified by the accel dialect.

```mlir
linalg.matmul {
  // Accelerator tile size of each loop (m, n, k); 0 = not tiled.
  accel.accel_dim = affine_map<(m, n, k) -> (4, 4, 4)>,
  // Order of the tiled loops, outermost first.
  accel.permutation_map = affine_map<(m, n, k) -> (m, k, n)>,
  // The opcodes and the actions each one performs, in order.
  accel.opcode_map = #accel.opcode_map<
    sA     = [send_literal(0x22), send(0)],
    sBcCrC = [send_literal(0x25), send(1), recv(2, accumulate)],
    reset  = [send_literal(0xFF)]>,
  // Opcodes sent once, before the loop nest.
  accel.init_opcodes = #accel.opcode_flow<(reset)>,
  // Where each opcode runs in the loop nest.
  accel.opcode_flow = #accel.opcode_flow<(sA (sBcCrC))>,
  // Optional: DMA engine address and buffers (sizes in 32-bit words).
  accel.dma_init_config = #accel.dma_config<address = 0x40400000,
      input_address = 0x16000000, input_size = 65536,
      output_address = 0x16400000, output_size = 65536>
} ins(%A, %B : ...) outs(%C : ...)
```

### Actions

Operand ids are the operand positions of the linalg op (for a matmul,
0 = A, 1 = B, 2 = C).

| Action | Sends or receives |
|---|---|
| `send(i)` | the current tile of operand `i` |
| `send_literal(x)` | the 32-bit word `x` |
| `send_dim(i, d)` | the size of the current tile of operand `i` along dimension `d` |
| `send_idx(i, d)` | the index of the current tile of operand `i` along dimension `d` (tile offset / tile size) |
| `recv(i)` | the current tile of output `i`, replacing it |
| `recv(i, accumulate)` | the current tile of output `i`, added to it |

For example, with tiles of size (10, 20), the tile at offset (10, 80) is tile
(1, 4): `send_idx(0, 0)` sends 1, `send_idx(0, 1)` sends 4, and
`send_dim(0, 1)` sends 20.

### Opcode flow

A flow is a chain of nested groups, `group ::= ( name* group? name* )`. The
innermost group runs in the innermost tiled loop. Each enclosing group runs
one loop further out: the names before its nested group just before the
inner loop, the names after it just after the inner loop. Empty groups skip a
loop level.

For a matmul with loops (m, n, k):

| Flow | Loop order | Dataflow |
|---|---|---|
| `(sA sB cC rC)` | m, n, k | nothing stationary |
| `(sA (sBcCrC))` | m, k, n | A stationary: A is sent before the n loop |
| `((sA sB cC) rC)` | m, n, k | C stationary: C is received after the k loop |

A tile moved at some loop level covers the whole range of the loops nested
inside it: that is how a tile stays stationary. The `strict-placement` option
of `convert-linalg-to-accel` rejects tiles that span an inner loop that they
depend on.

Loops that are not tiled are handled by the accelerator as a whole. For
example, the conv_v3 tests tile only the batch, output channel and output
pixel loops of `linalg.conv_2d_nchw_fchw`, and send whole input windows.

## Passes

* `convert-linalg-to-accel`: builds the tiled loop nest and the accel ops
  (see `soda-opt --help`). Its options attach missing attributes to the ops
  named by `anchor-op`, e.g.
  `--convert-linalg-to-accel="anchor-op=linalg.matmul accel-dim=4,4,4
  opcode-map=opcode_map<s=[send_literal(7),send(0),send(1)],r=[recv(2,accumulate)]>
  opcode-flow=opcode_flow<(s r)>"`.
  To tile for CPU caches first, tile the op with upstream tiling (e.g.
  `transform.structured.tile_using_for`): the tiles keep their attributes.
* `convert-accel-to-axi-runtime`: lowers accel ops to runtime calls. The
  sends of one opcode, chained through their offsets, become one DMA
  transaction (`coalesce-sends=false` sends each one separately).

## Runtime library

`include/soda/ExecutionEngine/axi/AxiRuntime.h` declares the C interface.
There is one library per backend:

| Library | Backend |
|---|---|
| `libsoda_axi_runtime_mock` | loopback stream, for tests |
| `libsoda_axi_runtime_hw` | Xilinx AXI DMA through `/dev/mem` (e.g. PYNQ boards) |
| `libsoda_axi_runtime_sysc_<model>` | SystemC simulation of the DMA engine and an accelerator model from `include/soda/ExecutionEngine/axi/accelerators` |

Environment variables:

* `SODA_AXI_TRACE=1` prints every DMA transaction.
* `SODA_AXI_SYSC_MAX_NS` bounds the simulated time of one transfer (default
  1e9 ns). A transfer that does not complete in time, e.g. after a wrong
  opcode, stops the program with an error.
* `SYSC_DISABLE_COPYRIGHT_MESSAGE=1` hides the SystemC banner.

## SystemC

SystemC is found at configure time through a SystemC CMake package,
pkg-config or `SYSTEMC_HOME`. `-DSODA_ENABLE_SYSTEMC=OFF` disables it. When
it is found, the SystemC runtime libraries are built and the tests in
`test/Runner/axi/sysc` run: matmul on `mm_4x4_v1` to `mm_4x4_v4` with
several dataflows, and convolution on `conv_v3`, each checked against the
CPU.
