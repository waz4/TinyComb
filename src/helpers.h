#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include "bn.h"
#include "combo.h"

#ifndef __NVCC__ // for CUDA/NVCC compatibility
#define __host__
#define __device__
#endif

// char *debug_env = getenv("DEBUG"); \
// Macro definition
#define DEBUG_PRINT(level, fmt, ...)                         \
    if (getenv("DEBUG") && atoi(getenv("DEBUG")) >= (level)) \
    {                                                        \
        printf(fmt, ##__VA_ARGS__);                          \
    }

/* usage: like printf
DEBUG_PRINT(1, "Debug level 1: Basic information.\n");
DEBUG_PRINT(2, "Debug level 2: Detailed information.\n");
DEBUG_PRINT(3, "Debug level 3: Verbose debugging information.\n");
*/

__host__ void printBigNum(struct bn *num);                         // Prints Big number
__host__ void print_factMap(struct bn *factMap, int factMap_size); // Prints Factorial Map
__host__ void print_combo(unsigned int *combo, int k);             // Prints an entire Combo

#endif // End of #ifndef HELPERS_H