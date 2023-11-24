#ifndef ACC_H
#define ACC_H

#define PE_M 16
#define PE_N 16
#define PE_K 16

#include "../dma_engine.sc.h"
#define ACCNAME MM_4x4v4

// #define VERBOSE_ACC
#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif

// OP-Code Stuct
// 0000 : 0 = NOP;
// 0001 : 1 = read_A;
// 0010 : 2 = read_B;
// 0011 : 3 = read_A -> read_B;
// 0100 : 4 = compute_C;
// 0101 : 5 = read_A -> compute_C;
// 0110 : 6 = read_B -> compute_C;
// 0111 : 7 = read_A -> read_B -> compute_C;

// 1000 : 8 = send_C;
// 1001 : 9 = read_A -> send_C;
// 1010 : 10 = read_B -> send_C;
// 1011 : 11 = read_A -> read_B -> send_C;
// 1100 : 12 = compute_C -> send_C;
// 1101 : 13 = read_A -> compute_C -> send_C;
// 1110 : 14 = read_B -> compute_C -> send_C;
// 1111 : 15 = read_A -> read_B -> compute_C -> send_C;

#define su10 sc_uint<10>
#define su12 sc_uint<12>
// MAX M, N, K = 256
struct opcode {
  unsigned int packet;
  bool read_A;
  bool read_B;
  bool compute_C;
  bool send_C;

  opcode(sc_uint<32> _packet) {
    // ALOG("OPCODE: " << _packet);
    // ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    read_A = _packet.range(0, 0);
    read_B = _packet.range(1, 1);
    compute_C = _packet.range(2, 2);
    send_C = _packet.range(3, 3);
  }
};

struct code_extension {
  su10 N;
  su10 M;
  su10 K;
  su10 K16;
  su10 N16;

  code_extension(sc_uint<32> _packetA) {
    M = _packetA.range(9, 0);
    N = _packetA.range(19, 10);
    K = _packetA.range(29, 20);
    N16 = _packetA.range(19, 10) / PE_N;
    K16 = _packetA.range(29, 20) / PE_K;
    // ALOG("packetA: " << _packetA);
    // ALOG("Time: " << sc_time_stamp());
    // ALOG("N: " << N << ", M: " << M << ", K: " << K);
    // cin.ignore();
  }
};

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> A_buffer[256][16];
  sc_int<32> B_buffer[256][16];
  sc_int<32> C_buffer[256][16];
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

  // Debug variables
  int process_blocks;
  int read_A_len;
  int read_B_len;
  int compute_C_len;
  int send_C_len;
  bool verbose;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send;
#else
  sc_signal<bool> compute;
  sc_signal<bool> send;
#endif

  code_extension acc_args = code_extension(0);

  void Recv();

  void Compute(sc_int<32>[PE_M][PE_K], sc_int<32>[PE_K][PE_N],
               sc_int<32>[PE_M][PE_N]);

  void LoadA(sc_int<32>[PE_M][PE_K], su10, su10, su10);

  void LoadB(sc_int<32>[PE_K][PE_N], su10, su10, su10);

  void Store(sc_int<32>[PE_M][PE_N], su10, su10, su10);

  void Schedule_Compute();

  void Send();

  void print_profile();

  int mul_int32(int, int);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Schedule_Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    process_blocks = 0;
    read_A_len = 0;
    read_B_len = 0;
    compute_C_len = 0;
    send_C_len = 0;
    verbose = false;
  }
};

template <typename Integer>
void accelerator_dma_connect(ACCNAME *acc, DMA_DRIVER *dmad,
                             int _dma_input_buffer_size,
                             int _dma_output_buffer_size) {

  static sc_clock clk_fast("ClkFast", 1, SC_NS);
  static sc_signal<bool> sig_reset;
  static sc_fifo<DATA> din1("din1_fifo", _dma_input_buffer_size);
  static sc_fifo<DATA> dout1("dout1_fifo", _dma_output_buffer_size);

  acc->clock(clk_fast);
  acc->reset(sig_reset);
  acc->dout1(dout1);
  acc->din1(din1);

  dmad->clock(clk_fast);
  dmad->reset(sig_reset);
  dmad->dout1(dout1);
  dmad->din1(din1);
}

void ACCNAME::print_profile() {
  ALOG("++++++++++++++++++++++++++++++++++++++++");
  ALOG("Read A data_len: " << read_A_len);
  ALOG("Read B data_len: " << read_B_len);
  ALOG("MACs count: " << compute_C_len);
  ALOG("Send C data_len: " << send_C_len);
  ALOG("++++++++++++++++++++++++++++++++++++++++");
  ALOG("Executed with :" << __FILE__);
  ALOG("- - - - - - - - - - - - - - - - - - - - ");
}

void ACCNAME::Recv() {

  wait();
  while (1) {
    opcode packet(din1.read().data);
    code_extension op_args(din1.read().data);
    acc_args = op_args;

    if (packet.read_A) {
      int read_length = op_args.M * op_args.K16;
      for (int i = 0; i < read_length; i++) {
        for (int j = 0; j < 16; j++) {
          A_buffer[i][j] = din1.read().data;
          read_A_len++;
          DWAIT();
        }
      }
    }

    if (packet.read_B) {
      int read_length = op_args.K * op_args.N16;
      for (int i = 0; i < read_length; i++) {
        for (int j = 0; j < 16; j++) {
          B_buffer[i][j] = din1.read().data;
          read_B_len++;
          DWAIT();
        }
      }
    }

    // Computes C if true
    if (packet.compute_C) {
      compute.write(true);
      wait();
    }

    while (compute)
      wait();

    // Sends then clears C if true
    if (packet.send_C) {
      send.write(true);
      wait();
    }

    while (send)
      wait();

    wait();
  }
}

void ACCNAME::LoadA(sc_int<32> A[PE_M][PE_K], su10 M, su10 K, su10 in_stride) {
  su12 base = M * in_stride + K;
  su12 offset = 0;
  for (su10 m = 0; m < PE_M; m++) {
    for (su10 k = 0; k < PE_K; k++) {
      // #pragma HLS unroll
      A[m][k] = A_buffer[base + offset][k];
    }
    offset += in_stride;
  }
}

void ACCNAME::LoadB(sc_int<32> B[PE_K][PE_N], su10 K, su10 N, su10 in_stride) {
  su12 base = K * in_stride + N;
  su12 offset = 0;
  for (su10 k = 0; k < PE_K; k++) {
    for (su10 n = 0; n < PE_N; n++) {
      // #pragma HLS unroll
      B[k][n] = B_buffer[base + offset][n];
    }
    offset += in_stride;
  }
}

void ACCNAME::Compute(sc_int<32> A[PE_M][PE_K], sc_int<32> B[PE_K][PE_N],
                      sc_int<32> C[PE_M][PE_N]) {
  for (int m = 0; m < PE_M; m++) {
    for (int n = 0; n < PE_N; n++) {
      // #pragma HLS pipeline
      // #pragma HLS unroll factor 4
      int acc = 0;
      for (int k = 0; k < PE_K; k++) {
        int x = A[m][k];
        int y = B[k][n];
        acc += mul_int32(x, y);
        compute_C_len++;
      }
      C[m][n] = acc;
    }
  }
}

void ACCNAME::Store(sc_int<32> C[PE_M][PE_N], su10 M, su10 N, su10 out_stride) {
  su12 base = M * out_stride + N;
  su12 offset = 0;
  for (su10 m = 0; m < PE_M; m++) {
    // #pragma HLS pipeline
    for (su10 n = 0; n < PE_N; n++) {
      // #pragma HLS unroll
      C_buffer[base + offset][n] += C[m][n];
    }
    offset += out_stride;
  }
}

void ACCNAME::Schedule_Compute() {
  sc_int<32> A[PE_M][PE_K];
  sc_int<32> B[PE_K][PE_N];
  sc_int<32> C[PE_M][PE_N];
  // #pragma HLS array_partition variable = A complete dim = 2
  // #pragma HLS array_partition variable = B complete dim = 2
  // #pragma HLS array_partition variable = C complete dim = 2

  wait();
  while (1) {
    while (!compute)
      wait();

    unsigned int ks = 0;
    for (su10 k = 0; k < acc_args.K; k += PE_K) {
      for (su10 m = 0; m < acc_args.M; m += PE_M) {
        LoadA(A, m, ks, acc_args.K16);
        for (su10 n = 0; n < acc_args.N16; n++) {
          LoadB(B, k, n, acc_args.N16);
          Compute(A, B, C);
          Store(C, m, n, acc_args.N16);
        }
      }
      ks++;
    }

    wait();
    compute.write(false);
    wait();
  }
}

void ACCNAME::Send() {
  wait();
  while (1) {
    while (!send)
      wait();

    unsigned int write_length = acc_args.M * acc_args.N16;
    for (su10 m = 0; m < write_length; m++) {
      for (su10 n = 0; n < 16; n++) {
        DATA d;
        d.tlast = false;
        d.data = C_buffer[m][n];
        if (n + 1 == 16 && m + 1 == write_length)
          d.tlast = true;
        dout1.write(d);
        send_C_len++;
        wait();
        C_buffer[m][n] = 0;
        DWAIT();
      }
    }
    send.write(false);
    wait();
  }
}

int ACCNAME::mul_int32(int x, int y) { return x * y; }

#endif
