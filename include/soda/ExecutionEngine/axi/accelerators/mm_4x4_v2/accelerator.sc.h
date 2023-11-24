#ifndef ACC_H
#define ACC_H

#include "../dma_engine.sc.h"

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

#define M 4
#define N 4
#define K 4

#define ACCNAME MM_4x4v2

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif

// OP-Code Stuct
// 000 : 0 = NOP;
// 001 : 1 = read_A;
// 010 : 2 = read_B;
// 011 : 3 = read_A -> read_B;
// 100 : 4 = compute_C;
// 101 : 5 = read_A -> compute_C;
// 110 : 6 = read_B -> compute_C;
// 111 : 7 = read_A -> read_B -> compute_C;

struct opcode {
  unsigned int packet;
  bool read_A;
  bool read_B;
  bool compute_C;

  opcode(sc_uint<32> _packet) {
    // ALOG("OPCODE: " << _packet);
    // ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    read_A = _packet.range(0, 0);
    read_B = _packet.range(1, 1);
    compute_C = _packet.range(2, 2);
  }
};

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> inputs[M][K];
  sc_int<32> weights[K][N];
  sc_int<32> outputs[M][N];
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

  void Recv();

  void Compute();

  void Send();

  void print_profile();

  int mul_int32(int, int);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    process_blocks = 0;
    read_A_len = 0;
    read_B_len = 0;
    compute_C_len = 0;
    send_C_len = 0;
    verbose = false;

    // clang-format off
    // #pragma HLS RESOURCE variable=din1 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA1" port_map={{din1_0 TDATA} {din1_1 TLAST}}
    // #pragma HLS RESOURCE variable=dout1 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA1" port_map={{dout1_0 TDATA} {dout1_1 TLAST}}
    // #pragma HLS RESET variable=reset

    // #pragma HLS array_partition variable=inputs complete dim=2
    // #pragma HLS array_partition variable=weights complete dim=0
    // #pragma HLS array_partition variable=outputs complete dim=2
    // clang-format on
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
    while (compute)
      wait();

    opcode packet(din1.read().data);

    if (packet.read_A) {
      for (int m = 0; m < M; m++) {
        // #pragma HLS pipeline
        for (int k = 0; k < K; k++) {
          inputs[m][k] = din1.read().data;
          read_A_len++;
        }
      }
      if (verbose) {
        cout << "=========================" << endl;
        cout << "Read BLOCK A: " << read_A_len++ << endl;
        cout << "=========================" << endl;
        for (int m = 0; m < M; m++) {
          for (int k = 0; k < K; k++)
            cout << inputs[m][k] << ",";
          cout << endl;
        }
        cout << "=========================" << endl;
      }
    }

    if (packet.read_B) {
      for (int k = 0; k < K; k++) {
        // #pragma HLS pipeline
        for (int n = 0; n < N; n++) {
          weights[k][n] = din1.read().data;
          read_B_len++;
        }
      }
      if (verbose) {
        cout << "=========================" << endl;
        cout << "Read BLOCK B: " << read_B_len++ << endl;
        cout << "=========================" << endl;
        for (int k = 0; k < K; k++) {
          for (int n = 0; n < N; n++)
            cout << weights[k][n] << ",";
          cout << endl;
        }
        cout << "=========================" << endl;
      }
    }

    if (packet.compute_C) {
      wait();
      compute.write(true);
    }
    wait();
  }
}

void ACCNAME::Compute() {
  wait();
  while (1) {
    while (!compute)
      wait();

    for (int m = 0; m < M; m++) {
      // #pragma HLS pipeline
      for (int n = 0; n < N; n++) {
        int acc = 0;
        for (int k = 0; k < K; k++) {
          int x = inputs[m][k];
          int y = weights[k][n];
          acc += mul_int32(x, y);
          compute_C_len++;
        }
        outputs[m][n] = acc;
      }
    }

    if (verbose) {
      cout << "=========================" << endl;
      cout << "Compute BLOCK C: " << compute_C_len++ << endl;
      cout << "=========================" << endl;
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++)
          cout << outputs[m][n] << ",";
        cout << endl;
      }
      cout << "=========================" << endl;
    }
    wait();
    compute.write(false);
    send.write(true);
    wait();
  }
}

int ACCNAME::mul_int32(int x, int y) { return x * y; }

void ACCNAME::Send() {
  wait();
  while (1) {
    while (!send)
      wait();

    for (int m = 0; m < M; m++) {
      // #pragma HLS pipeline
      for (int n = 0; n < N; n++) {
        DATA d;
        d.tlast = false;
        if (m == M - 1 && n == N - 1)
          d.tlast = true;
        d.data = outputs[m][n];
        dout1.write(d);
        send_C_len++;
        DWAIT();
      }
    }
    send.write(false);
    wait();
  }
}

#endif
