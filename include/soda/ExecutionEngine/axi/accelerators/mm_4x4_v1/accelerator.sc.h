#ifndef ACC_H
#define ACC_H

#include "../dma_engine.sc.h"


#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

#define ACCNAME MM_4x4v1

#define M 4
#define N 4
#define K 4

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif


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
  bool verbose = true;

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

    // #pragma HLS RESOURCE variable=din1 core=AXI4Stream metadata="-bus_bundle
    // S_AXIS_DATA1" port_map={{din1_0 TDATA} {din1_1 TLAST}} #pragma HLS
    // RESOURCE variable=dout1 core=AXI4Stream metadata="-bus_bundle
    // M_AXIS_DATA1" port_map={{dout1_0 TDATA} {dout1_1 TLAST}} #pragma HLS
    // RESET variable=reset
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
  ALOG("++++++++++++++++++++++++++++++++++++++++" );
  ALOG("Read A data_len: " << read_A_len);
  ALOG("Read B data_len: " << read_B_len);
  ALOG("MACs count: " << compute_C_len);
  ALOG("Send C data_len: " << send_C_len);
  ALOG("++++++++++++++++++++++++++++++++++++++++" );
  ALOG("Executed with :" << __FILE__ );
  ALOG("- - - - - - - - - - - - - - - - - - - - ");
}

void ACCNAME::Recv() {
  wait();
  while (1) {
    while (compute)
      wait();

    for (int m = 0; m < M; m++) {
      // #pragma HLS pipeline
      for (int k = 0; k < K; k++) {
        inputs[m][k] = din1.read().data;
        read_A_len++;
        DWAIT();
      }
    }

    for (int k = 0; k < K; k++) {
      // #pragma HLS pipeline
      for (int n = 0; n < N; n++) {
        weights[k][n] = din1.read().data;
        read_B_len++;
        DWAIT();
      }
    }

    // DEBUG ONLY
    if (verbose) {
      cout << "=========================" << endl;
      cout << "BLOCK: " << process_blocks++ << endl;
      cout << "=========================" << endl;
      for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++)
          cout << inputs[m][k] << ",";
        cout << endl;
      }
      cout << "=========================" << endl;
      for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++)
          cout << weights[k][n] << ",";
        cout << endl;
      }
      cout << "=========================" << endl;
    }
    // DEBUG ONLY

    wait();
    compute.write(true);
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

    // DEBUG ONLY
    if (verbose) {
      cout << "=========================" << endl;
      cout << "Output: " << process_blocks - 1 << endl;
      cout << "=========================" << endl;
      cout << "=========================" << endl;
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++)
          cout << outputs[m][n] << ",";
        cout << endl;
      }
      cout << "=========================" << endl;
    }
    // DEBUG ONLY

    wait();
    compute.write(false);
    send.write(true);
    wait();
  }
}

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

int ACCNAME::mul_int32(int x, int y) { return x * y; }

#endif
