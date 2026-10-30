#ifndef ACC_H
#define ACC_H

#include "../dma_engine.sc.h"

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#define HLSPRAGMA(X)
#else
#define DWAIT(x)
#define HLSPRAGMA(X) Pragma(#X)
typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;
#endif

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif

// Accelerator parameters
#define H 7   // Filter H 1,3,5,7
#define W 7   // Filter W 1,3,5,7
#define C 512 // Input C

// OP-Code Stuct
// 0000 : 0 = NOP;
// 0001 : 1 = read_fliters;
// 0010 : 2 = read_inputs;
// 0011 : 3 = read_fliters -> read_inputs;
// 0100 : 4 = compute_outputs;
// 0101 : 5 = read_fliters -> compute_outputs;
// 0110 : 6 = read_inputs -> compute_outputs;
// 0111 : 7 = read_fliters -> read_inputs -> compute_outputs;

// 1000 : 8 = send_outputs;
// 1001 : 9 = read_fliters -> send_outputs;
// 1010 : 10 = read_inputs -> send_outputs;
// 1011 : 11 = read_fliters -> read_inputs -> send_outputs;
// 1100 : 12 = compute_outputs -> send_outputs;
// 1101 : 13 = read_fliters -> compute_outputs -> send_outputs;
// 1110 : 14 = read_inputs -> compute_outputs -> send_outputs;
// 1111 : 15 = read_fliters -> read_inputs -> compute_outputs -> send_outputs;

// 10000 : 16 = set_channels;
// 100000 : 32 = set_filter_size;
// 1000000 : 64 = save_output;

struct opcode {
  unsigned int packet;
  bool read_fliters;
  bool read_inputs;
  bool compute_outputs;
  bool send_outputs;
  bool set_channels;
  bool set_filter_size;
  bool save_output;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    read_fliters = _packet.range(0, 0);
    read_inputs = _packet.range(1, 1);
    compute_outputs = _packet.range(2, 2);
    send_outputs = _packet.range(3, 3);
    set_channels = _packet.range(4, 4);
    set_filter_size = _packet.range(5, 5);
    save_output = _packet.range(6, 6);
  }
};

#define ACCNAME CONV_ACC_V3
SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_int<32> inputs[H * W][C];
  sc_int<32> filters[H * W][C]; // C ==  IC 
  sc_int<32> output[16384]; // OW * OH
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send;
  sc_signal<bool, SC_MANY_WRITERS> save_output;
  sc_signal<int, SC_MANY_WRITERS> output_size;
#else
  sc_signal<bool> compute;
  sc_signal<bool> send;
  sc_signal<bool> save_output;
  sc_signal<int> output_size;
#endif

  sc_signal<int> fs;
  sc_signal<int> ic;
  sc_signal<int> ln;

  // Debug variables
  int process_blocks;
  int read_A_len;
  int read_B_len;
  int compute_C_len;
  int send_C_len;
  bool verbose;

  void Recv() {
    ic.write(0);
    wait();
    while (1) {
      opcode packet(din1.read().data);
      if (packet.set_channels) {
        int tic = din1.read().data;
        int fs_ic = fs * tic;
        int x = fs_ic + 48;
        int nic = (x - (x % 49)) / 49;
        ic.write(nic);
        ln.write(fs_ic);
      }

      if (packet.set_filter_size) {
        int nfs = din1.read().data;
        fs.write(nfs * nfs);
      }

      if (packet.read_inputs) {
        HLSPRAGMA(HLS pipeline)
        int hid = 0;
        int cid = 0;
        for (int hw = 0; hw < ln; hw++) {
          inputs[hid++][cid] = din1.read().data;
          if (hid == 49) {
            hid = 0;
            cid++;
          }
        }
      }

      if (packet.read_fliters) {
        HLSPRAGMA(HLS pipeline)
        int hid = 0;
        int cid = 0;
        for (int hw = 0; hw < ln; hw++) {
          filters[hid++][cid] = din1.read().data;
          if (hid == 49) {
            hid = 0;
            cid++;
          }
        }
      }

      if (packet.compute_outputs) {
        compute.write(true);
        wait();
      }

      wait();
      while (compute)
        wait();

      if (packet.save_output)
        output_size.write(output_size + 1);

      if (packet.send_outputs) {
        wait();
        send.write(true);
      }

      while (send)
        wait();
    }
  }

  void Compute() {
    int ic_acc[49][C];
    HLSPRAGMA(HLS array_partition variable = ic_acc complete dim = 1)

    wait();
    while (1) {
      while (!compute)
        wait();

      for (int c = 0; c < C; c++) {
        HLSPRAGMA(HLS pipeline)
        for (int hw = 0; hw < 49; hw++) {
          HLSPRAGMA(HLS unroll)
          ic_acc[hw][c] = 0;
        }
      }

      for (int c = 0; c < ic; c++) {
        HLSPRAGMA(HLS pipeline)
        for (int hw = 0; hw < H * W; hw++) {
          int x = inputs[hw][c];
          int y = filters[hw][c];
          ic_acc[hw][c] += mul_int32(x, y);
        }
      }

      int hid = 0;
      int cid = 0;
      for (int hw = 0; hw < ln; hw++) {
        output[output_size] += ic_acc[hid++][cid];
        if (hid == 49) {
          hid = 0;
          cid++;
        }
      }

      wait();
      compute.write(false);
      wait();
    }
  }

  int mul_int32(int x, int y) { return x * y; }

  void Send() {
    output_size.write(0);
    wait();
    while (1) {
      while (!send)
        wait();

      for (int i = 0; i < output_size; i++) {
        DATA d;
        d.data = output[i];
        bool last = (i + 1 == output_size);
        d.tlast = last;
        dout1.write(d);
      }
      wait();

      for (int i = 0; i < output_size; i++) {
        HLSPRAGMA(HLS unroll)
        output[i] = 0;
      }
      output_size.write(0);
      send.write(false);
      wait();
    }
  }

  void print_profile() {
    cout << "++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "Executed with :" << __FILE__ << endl;
    cout << "++++++++++++++++++++++++++++++++++++++++" << endl;
  }

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_)
      : sc_module(
            name_) // @suppress("Class members should be properly initialized")
  {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    // clang-format off
#ifdef __SYNTHESIS__
#pragma HLS RESOURCE variable=din1 core=AXI4Stream metadata="-bus_bundle S_AXIS_DATA1" port_map={{din1_0 TDATA} {din1_1 TLAST}}
#pragma HLS RESOURCE variable=dout1 core=AXI4Stream metadata="-bus_bundle M_AXIS_DATA1" port_map={{dout1_0 TDATA} {dout1_1 TLAST}}
#pragma HLS RESET variable=reset

#pragma HLS array_partition variable=inputs complete dim=1
#pragma HLS array_partition variable=filters complete dim=1
#endif
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

#endif
