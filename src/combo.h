#ifndef COMBO_H
#define COMBO_H
#include "bn.h"

#ifndef __NVCC__ // for CUDA/NVCC compatibility
#define __host__
#define __device__
#endif
/*
    This library helps calculate combinations with repetition based on the index of occurance.
    Where 'n' is the number of elements and 'k' is the number of slots. As in C(n + k - 1, k)

    - The maximum number of combinations is returned by g()
    - Index 0 should be (0, 0, ..., 0)  And Index g(n, k, ...) should be (n-1, n-1, ...., n-1)
                         0  1       k                                      0    0          k
    - nextCombo of index(x) should return index(x + 1)

    TODO: remove reverse from id2combo probably by changing next_combo
*/

// Factorial calculation
__device__ __host__ void factorial(struct bn *n, struct bn *res);
__device__ __host__ void factorial_safe(int n, struct bn *result);

// Functions to convert number into combination,
//  - factMap is optionall and if not being used can be ignored by passing null
__device__ __host__ void id2combo(struct bn *id, int n, int k, unsigned int *combo, struct bn *factMap);
__device__ __host__ unsigned int largest(struct bn *i, int nn, int kk, struct bn *x, struct bn *factMap);
__device__ __host__ void g(int n, int k, struct bn *result, struct bn *factMap); // Calculates (n + k - 1)! / (k! * (n - 1)!) into result

// Functions to deal with a single Combo
__device__ __host__ void init_combo(unsigned int *combo, int k);
__device__ __host__ void next_combo(unsigned int *ar, unsigned int n, unsigned int k);
__device__ __host__ char combosMatch(unsigned int *comboA, unsigned int *comboB, int k);
__device__ __host__ void reverse_combo(unsigned int *combo, int k);

// Factorial Map
void makeFactMap(struct bn *factMap, int factMap_lenght);
int getFactMapLength(int n, int k); // Returns the required factMap_size for a combinations with n elements and k slots

#endif // End of #ifndef COMBO_H