#include <stdio.h>
#include "../src/bn.h"
#include "../src/combo.h"
#include "../src/helpers.h"

#define COMBO_SIZE 10
#define NUMBER_OF_ELEMENTS 23

__global__ void exampleKernel(unsigned int *d_combo)
{
  printf("Kernel Start\n");

  struct bn id, number, result;
  bignum_from_int(&id, 0);
  bignum_from_int(&number, 250000);

  bignum_add(&id, &number, &result); // 250000

  bignum_inc(&id);
  bignum_inc(&id);

  bignum_mul(&id, &number, &result); // 500000

  unsigned int combo[COMBO_SIZE];
  id2combo(&result, COMBO_SIZE, NUMBER_OF_ELEMENTS, combo, NULL);

  // Copy the combo array to the device output memory
  for (int i = 0; i < COMBO_SIZE; i++)
  {
    d_combo[i] = combo[i];
  }

  // Print from device (for debugging, though output might not be reliable)
  for (int i = 0; i < COMBO_SIZE; i++)
  {
    printf("%u", combo[i]);
  }
  printf("\n");
}

int main()
{

  // Host computation (unchanged)
  struct bn id, number, result;
  bignum_from_int(&id, 0);
  bignum_from_int(&number, 250000);

  bignum_add(&id, &number, &result); // 250000

  bignum_inc(&id);
  bignum_inc(&id);

  bignum_mul(&id, &number, &result); // 500000

  unsigned int combo[COMBO_SIZE];
  id2combo(&result, COMBO_SIZE, NUMBER_OF_ELEMENTS, combo, NULL);

  // printf("Host:\n");
  // for (int i = 0; i < COMBO_SIZE; i++)
  // {
  //   printf("%u ", combo[i]);
  // }
  // printf("\n");

  // Device computation and result retrieval
  unsigned int *d_combo;            // Device pointer for combo array
  unsigned int h_combo[COMBO_SIZE]; // Host array to store device result

  // Allocate device memory
  cudaMalloc((void **)&d_combo, COMBO_SIZE * sizeof(unsigned int));

  // Launch kernel
  exampleKernel<<<1, 1>>>(d_combo); // Use 1 thread since this isn't parallelized

  // Synchronize to ensure kernel completion
  cudaDeviceSynchronize();

  // Copy result from device to host
  cudaMemcpy(h_combo, d_combo, COMBO_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  bool combos_match = true;

  printf("Device result copied to host.\n");

  // Print device result on host
  printf("Device | Host\n");
  for (int i = 0; i < COMBO_SIZE; i++)
  {
    printf("%u | %u\n", h_combo[i], combo[i]);
    if (h_combo[i] != combo[i])
      combos_match = false;
  }
  printf("\n");

  if (combos_match)
    printf("Test Passed!\n");
  else
    printf("Test Failed!\n");

  // Free device memory
  cudaFree(d_combo);

  return 0;
}