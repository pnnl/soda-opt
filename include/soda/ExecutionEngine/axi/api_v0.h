//**********************Deprecated**********************

#ifndef AXI_APIv0
#define AXI_APIv0

#include <cstdlib>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

// API Model = One DMA is allocated with an input and an output buffer
// TODO: Struct based representation of API model

struct dma {
#define MM2S_CONTROL_REGISTER 0x00
#define MM2S_STATUS_REGISTER 0x04
#define MM2S_START_ADDRESS 0x18
#define MM2S_LENGTH 0x28

#define S2MM_CONTROL_REGISTER 0x30
#define S2MM_STATUS_REGISTER 0x34
#define S2MM_DESTINATION_ADDRESS 0x48
#define S2MM_LENGTH 0x58

  unsigned int id;
  unsigned int *dma_address;
  unsigned int *dma_input_addr;
  unsigned int *dma_output_addr;

  unsigned int dma_input_len;
  unsigned int dma_output_len;

  void init(int id);

  void dma_set(unsigned int *dma_virtual_address, int offset,
                       unsigned int value) {
    dma_virtual_address[offset >> 2] = value;
  }

  unsigned int dma_get(unsigned int *dma_virtual_address, int offset) {
    return dma_virtual_address[offset >> 2];
  }
};

struct dma_collection {

  // Variables
  int dma_count;
  struct dma *dma_list;

  //-----------------DMA Functions-----------------
  /**
   * dma_address is base address of dma
   * dma_input_addr is starting memory location for the dma input buffer,
   * dma_input_len is length of the buffer dma_output_addr is starting memory
   * location for the dma output buffer, dma_output_len is length of the buffer
   * Memory maps dma's base address
   * Runs starting controls signals and sets MMS2, S2MM address registers to
   * start memory locations of the input and output buffers
   */
  void dma_init(int dma_count, unsigned int *dma_address,
                unsigned int *dma_input_addr, unsigned int *dma_input_len,
                unsigned int *dma_output_addr, unsigned int *dma_output_len);

  // Memory unmaps DMA base addresses and Input and output buffers
  void dma_free();

  // Get base address for dma represented by dma_id,
  unsigned int *dma_get_regaddr();

  //-----------------BUFFER Functions-----------------
  // Get the MMap address of the input buffer of the dma
  unsigned int *dma_get_inbuffer();

  // Get the MMap address of the output buffer of the dma
  unsigned int *dma_get_outbuffer();

  //-----------------DMA MMS2 Functions-----------------
  /**
   * Checks if input buffer size is >= length
   * Sets DMA MMS2 transfer length to length
   * Starts transfers to the accelerator using dma associated with dma_id
   * Return 0 if successful, returns negative if error occurs
   */
  int dma_set_transfer(int dma_id, int length);

  // Blocks thread until dma MMS2 transfer is complete
  void dma_send(int dma_id, int buffer_ID, int length);

  // Same as dma_send but thread does not block, returns if 0
  int dma_send_nb(int dma_id, int buffer_ID, int length);

  //-----------------DMA S2MM Functions-----------------
  /**
   * Checks if buffer size is >= length
   * Sets 2SMM store length
   * Starts storing data recieved through dma associated with dma_id
   * Return 0 if successful, returns negative if error occurs
   */
  int dma_set_store(int dma_id, int buffer_ID, int length);

  // Blocks thread until dma S2MM transfer is complete (TLAST signal is seen)
  void dma_recv(int dma_id, int buffer_ID, int length);

  // Same as dma_recv but thread does not block, returns if 0
  int dma_recv_nb(int dma_id, int buffer_ID, int length);
};

#endif