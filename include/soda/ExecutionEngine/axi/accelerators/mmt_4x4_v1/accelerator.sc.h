#ifndef ACC_H
#define ACC_H

#include "../dma_engine.sc.h"
#define ACCNAME MM_4x4v1

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> inputs[4096];
  sc_int<32> weights[4096];
  sc_int<32> outputs[4096];
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

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    process_blocks = 0;
    read_A_len=0;
    read_B_len=0;
    compute_C_len=0;
    send_C_len=0;
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
  cout << "++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "Read A data_len: " << read_A_len << endl;
  cout << "Read B data_len: " << read_B_len << endl;
  cout << "MACs count: " << compute_C_len << endl;
  cout << "Send C data_len: " << send_C_len << endl;
  cout << "++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "Executed with :" << __FILE__ << endl;
  cout << "- - - - - - - - - - - - - - - - - - - - " << endl;;
}

void ACCNAME::Recv() {
  wait();
  while (1) {
    while (compute)
      wait();

    for (int i = 0; i < 16; i++) {
      inputs[i] = din1.read().data;
      read_A_len++;
      DWAIT();
    }

    for (int i = 0; i < 16; i++) {
      weights[i] = din1.read().data;
      read_B_len++;
      DWAIT();
    }

    // DEBUG ONLY
    if (true) {
      cout << "=========================" << endl;
      cout << "BLOCK: " << process_blocks++ << endl;
      cout << "=========================" << endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
          cout << inputs[i * 4 + j] << ",";
        cout << endl;
      }
      cout << "=========================" << endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
          cout << weights[i * 4 + j] << ",";
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
    for (int i = 0; i < 4; i++) {
      for (int w = 0; w < 4; w++) {
        int acc = 0;
        for (int d = 0; d < 4; d++) {
          int x = inputs[i * 4 + d];
          int y = weights[w * 4 + d];
          acc += x * y;
          compute_C_len++;
        }
        outputs[i * 4 + w] = acc;
      }
    }

    // DEBUG ONLY
    if (verbose) {
      cout << "=========================" << endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
          cout << outputs[i * 4 + j] << ",";
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
    for (int i = 0; i < 16; i++) {
      DATA d;
      d.tlast = false;
      if (i == 15)
        d.tlast = true;
      d.data = outputs[i];
      dout1.write(d);
      send_C_len++;
      DWAIT();
    }
    send.write(false);
    wait();
  }
}

#endif
