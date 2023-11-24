#ifndef DMA_DRIVER_H
#define DMA_DRIVER_H

#include <systemc.h>

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;

SC_MODULE(DMA_DRIVER) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_fifo_in<DATA> dout1;
  sc_fifo_out<DATA> din1;
  bool send;
  bool recv;

  void DMA_MMS2() {
    while (1) {

      while (!send)
        wait();
      
      int send_len = input_len * isize;
      for (int i = 0; i < send_len; i++) {
        sc_uint<32> d;
        d.range(7, 0) = DMA_input_buffer[(input_offset * isize) + i++];
        if (isize > 1 && i < send_len)
          d.range(15, 8) = DMA_input_buffer[(input_offset * isize) + i++];
        if (isize > 2 && i < send_len)
          d.range(23, 16) = DMA_input_buffer[(input_offset * isize) + i++];
        if (isize > 3 && i < send_len)
          d.range(31, 24) = DMA_input_buffer[(input_offset * isize) + i++];
        wait();
        din1.write({d, 1});
        wait();
      }

      send = false;
      wait();
      sc_pause();
      wait();
    }
  };

  void DMA_S2MM() {
    while (1) {
      while (!recv)
        wait();
      bool last = false;
      int i = 0;

      do {
        DATA d = dout1.read();
        int recv_len = output_len * osize;
        while (i >= recv_len)
          wait();
        last = d.tlast;
        DMA_output_buffer[(output_offset * osize) + i++] = d.data.range(7, 0);
        if (osize > 1)
          DMA_output_buffer[(output_offset * osize) + i++] =
              d.data.range(15, 8);
        if (osize > 2)
          DMA_output_buffer[(output_offset * osize) + i++] =
              d.data.range(23, 16);
        if (osize > 3)
          DMA_output_buffer[(output_offset * osize) + i++] =
              d.data.range(31, 24);
        wait();
      } while (!last);

      recv_len = i;
      recv = false;
      // To ensure wait_send() does not evoke the sc_pause
      while (send)
        wait(2);
      sc_pause();
      wait();
    }
  };

  SC_HAS_PROCESS(DMA_DRIVER);

  DMA_DRIVER(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(DMA_MMS2, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(DMA_S2MM, clock.pos());
    reset_signal_is(reset, true);
  }

  char *DMA_input_buffer;
  char *DMA_output_buffer;

  // length = Number of elements
  unsigned int input_len;
  unsigned int input_offset;
  unsigned int isize;

  unsigned int output_len;
  unsigned int output_offset;
  unsigned int osize;
};

#endif