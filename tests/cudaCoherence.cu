#include <stdio.h>
// #include "../src/bn.h"
// #include "../src/combo.h"
// #include "../src/helpers.h"
#include "../src/bn.c"
#include "../src/combo.c"
#include "../src/helpers.c"

#define COMBO_SIZE 10
#define NUMBER_OF_ELEMENTS 23

__global__ void exampleKernel()
{
  printf("Device:\n");

  struct bn id, number, result;
  bignum_from_int(&id, 0);
  bignum_from_int(&number, 250000);

  bignum_add(&id, &number, &result); // 250000

  bignum_inc(&id);
  bignum_inc(&id);

  bignum_mul(&id, &number, &result); // 50000

  unsigned int combo[10];
  id2combo(&result, COMBO_SIZE, NUMBER_OF_ELEMENTS, combo, NULL);

  for (int i = 0; i < COMBO_SIZE; i++)
  {
    printf("%u", combo[i]);
  }
  printf("\n");
}

int main()
{
  printf("Host:\n");

  struct bn id, number, result;
  bignum_from_int(&id, 0);
  bignum_from_int(&number, 250000);

  bignum_add(&id, &number, &result); // 250000

  bignum_inc(&id);
  bignum_inc(&id);

  bignum_mul(&id, &number, &result); // 50000

  unsigned int combo[10];
  id2combo(&result, COMBO_SIZE, NUMBER_OF_ELEMENTS, combo, NULL);

  for (int i = 0; i < COMBO_SIZE; i++)
  {
    printf("%u", combo[i]);
  }
  printf("\n");

  exampleKernel<<<1, 10>>>();
  return 0;
}