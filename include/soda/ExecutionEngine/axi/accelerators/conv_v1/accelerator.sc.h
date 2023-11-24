#ifndef ACC_H
#define ACC_H

#include "../dma_engine.sc.h"
#define ACCNAME MM_4x4v1

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

  // Debug variables
  int process_blocks;
  int read_A_len;
  int read_B_len;
  int compute_C_len;
  int send_C_len;
  bool verbose;


  void Recv();

  void print_profile();

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    process_blocks = 0;
    read_A_len=0;
    read_B_len=0;
    compute_C_len=0;
    send_C_len=0;
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
			bool tlast = false;
			int output = 0;
			while(!tlast){
				DATA inp = din1.read();
				DATA wgt = din1.read();
				output+= inp.data*wgt.data;

        // cout <<  inp.data << "*" << wgt.data << endl;
				tlast = (inp.tlast || wgt.tlast);
        DWAIT();
			}
			DATA d;
			d.tlast = true;
			d.data = output;
			dout1.write(d);
      DWAIT();
  }
}
#endif
