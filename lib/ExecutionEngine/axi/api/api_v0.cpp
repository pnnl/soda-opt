//**********************Deprecated**********************

#include "mlir/ExecutionEngine/axi/api_v0.h"

void dma::init(int id) {
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 4);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 4);
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 0);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 0);
  dma_set(dma_address, S2MM_DESTINATION_ADDRESS,
          (unsigned long)dma_output_addr); // Write destination address
  dma_set(dma_address, MM2S_START_ADDRESS,
          (unsigned long)dma_input_addr); // Write source address
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 0xf001);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 0xf001);
}

void dma_collection::dma_init(int dma_count, unsigned int *dma_address,
                              unsigned int *dma_input_addr,
                              unsigned int *dma_input_len,
                              unsigned int *dma_output_addr,
                              unsigned int *dma_output_len) {
  // Open /dev/mem which represents the whole physical memory
  int dh = open("/dev/mem", O_RDWR | O_SYNC);
  dma_list = new dma[dma_count];
  int id_count = 0;

  for (int i = 0; i < dma_count; i++) {
    void *dma_mm = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, dh,
                        dma_address[i]); // Memory map AXI Lite register block
    void *dma_in_mm =
        mmap(NULL, dma_input_len[i], PROT_READ | PROT_WRITE, MAP_SHARED, dh,
             dma_input_addr[i]); // Memory map source address
    void *dma_out_mm =
        mmap(NULL, dma_output_len[i], PROT_READ, MAP_SHARED, dh,
             dma_output_addr[i]); // Memory map destination address
    unsigned int *dma_addr = reinterpret_cast<unsigned int *>(dma_mm);
    unsigned int *dma_in = reinterpret_cast<unsigned int *>(dma_in_mm);
    unsigned int *dma_out = reinterpret_cast<unsigned int *>(dma_out_mm);

    dma_list[i].dma_address = dma_addr;
    dma_list[i].dma_input_addr = dma_in;
    dma_list[i].dma_output_addr = dma_out;
    dma_list[i].dma_input_len = dma_input_len[i];
    dma_list[i].dma_output_len = dma_output_len[i];
    dma_list[i].init(id_count++);
  }
}