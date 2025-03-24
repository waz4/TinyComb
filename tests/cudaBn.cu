#include <stdio.h>
#include "../src/bn.h"

// Kernel to compute factorial
__global__ void factorial(struct bn *result)
{
  struct bn num;
  struct bn tmp;

  // Initialize num to 100
  bignum_from_int(&num, 100);

  // Copy num to tmp
  bignum_assign(&tmp, &num);

  // Decrement num by one
  bignum_dec(&num);

  // Calculate factorial
  while (!bignum_is_zero(&num))
  {
    bignum_mul(&tmp, &num, result);
    bignum_dec(&num);
    bignum_assign(&tmp, result);
  }
}

int main()
{
  // Device and host variables
  struct bn *d_result; // Device result
  struct bn h_result;  // Host result

  // Allocate memory on device
  cudaMalloc((void **)&d_result, sizeof(struct bn));

  // Initialize host result
  bignum_init(&h_result);

  // Launch kernel with 1 block and 1 thread since factorial is not parallelizable
  factorial<<<1, 1>>>(d_result);

  // Synchronize to ensure kernel completion
  cudaDeviceSynchronize();

  // Copy result from device to host
  cudaMemcpy(&h_result, d_result, sizeof(struct bn), cudaMemcpyDeviceToHost);

  // Convert result to string for printing
  char result_str[1024]; // Buffer for the result string
  bignum_to_string(&h_result, result_str, 1024);

  // Print the result
  printf("Factorial(100) = %s\n", result_str);

  // Free device memory
  cudaFree(d_result);

  return 0;
}