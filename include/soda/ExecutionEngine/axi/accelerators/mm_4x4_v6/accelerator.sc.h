#ifndef ACC_H
#define ACC_H

#include "../dma_engine.sc.h"
#define ACCNAME MM_4x4v6

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

struct opcode {
  unsigned int packet;
  bool read_A;
  bool read_B;
  bool compute_C;
  bool send_C;

  opcode(sc_uint<32> _packet) {
    cout << "OPCODE: " << _packet << endl;
    cout << "Time: " << sc_time_stamp() << endl;
    packet = _packet;
    read_A = _packet.range(0, 0);
    read_B = _packet.range(1, 1);
    compute_C = _packet.range(2, 2);
    send_C = _packet.range(3, 3);
  }
};

struct code_extension {
  sc_uint<16> N;
  sc_uint<16> M;

  sc_uint<16> K;
  sc_uint<16> in_stride;

  sc_uint<16> out_offset;
  sc_uint<16> out_stride;

  code_extension(sc_uint<32> _packetA, sc_uint<32> _packetB,
                 sc_uint<32> _packetC) {

    N = _packetA.range(15,0);
    M = _packetA.range(31,16);

    K = _packetB.range(15,0);
    in_stride = _packetB.range(31,16);

    out_offset = _packetC.range(15,0);
    out_stride = _packetC.range(31,16);

    cout << "Time: " << sc_time_stamp() << endl;
    cout << "N: " << N << ", M: " << M << ", K: " << K
         << ", in_stride: " << in_stride << ", out_offset: " << out_offset
         << ", out_stride: " << out_stride << endl;
  }
};

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> A_buffer[4096];
  sc_int<32> B_buffer[4096];
  sc_int<32> C_buffer[4096];
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

  // Debug variables
  int process_blocks;
  bool verbose;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send;
#else
  sc_signal<bool> compute;
  sc_signal<bool> send;
#endif

  code_extension acc_args = code_extension(0,0,0);

  void Recv();

  void Compute(int, int, int, int, int, int);

  void Schedule_Compute();

  void Send();

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Schedule_Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    process_blocks = 0;
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

void ACCNAME::Recv() {
  wait();
  while (1) {
    opcode packet(din1.read().data);
    code_extension op_args(din1.read().data, din1.read().data,
                           din1.read().data);
    acc_args = op_args;

    if (packet.read_A) {
      unsigned int read_length = op_args.N * op_args.K;
      for (int i = 0; i < read_length; i++) {
        A_buffer[i] = din1.read().data;
        DWAIT();
      }
    }

    if (packet.read_B) {
      unsigned int read_length = op_args.M * op_args.K;
      for (int i = 0; i < read_length; i++) {
        B_buffer[i] = din1.read().data;
        DWAIT();
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

void ACCNAME::Compute(int N, int M, int K, int in_stride, int out_offset,
                      int out_stride) {
  for (int n = 0; n < 4; n++) {
    for (int m = 0; m < 4; m++) {
      int acc = 0;
      for (int k = 0; k < 4; k++) {
        int a_data = A_buffer[(N + n) * in_stride + K + k];
        int b_data = B_buffer[(M + m) * in_stride + K + k];
        acc += a_data * b_data;
      }
      C_buffer[out_offset + (N + n) * out_stride + M + m] += acc;
    }
  }
}

void ACCNAME::Schedule_Compute() {
  wait();
  while (1) {
    while (!compute)
      wait();

    for (int n = 0; n < acc_args.N; n += 4) {
      for (int m = 0; m < acc_args.M; m += 4) {
        for (int k = 0; k < acc_args.K; k += 4) {
          Compute(n, m, k, acc_args.in_stride, acc_args.out_offset,
                  acc_args.out_stride);
        }
      }
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

    for (int n = 0; n < acc_args.N; n++) {
      for (int m = 0; m < acc_args.M; m++) {
        DATA d;
        d.tlast = false;
        d.data = C_buffer[acc_args.out_offset + n * acc_args.out_stride + m];
        if (n + 1 == acc_args.N && m + 1 == acc_args.M)
          d.tlast = true;
        dout1.write(d);
        C_buffer[acc_args.out_offset + n * acc_args.out_stride + m] = 0;
      }
    }
    send.write(false);
    wait();
  }
}


#endif
