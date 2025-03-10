#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include "bn.h"
#include "combo.h"

// This macro prints only when "DEBUG" env variable is set
#include <stdio.h>
#include <stdlib.h>

// char *debug_env = getenv("DEBUG"); \
// Macro definition
#define DEBUG_PRINT(level, fmt, ...) \
    if (getenv("DEBUG") && atoi(getenv("DEBUG")) >= (level)) { \
        printf(fmt, ##__VA_ARGS__); \
    } \


/* usage: like printf
DEBUG_PRINT(1, "Debug level 1: Basic information.\n");
DEBUG_PRINT(2, "Debug level 2: Detailed information.\n");
DEBUG_PRINT(3, "Debug level 3: Verbose debugging information.\n");
*/

void printBigNum(struct bn* num); // Prints Big number
void print_factMap(struct bn* factMap, int factMap_size); // Prints Factorial Map
void print_combo(unsigned int *combo, int k); // Prints an entire Combo

#endif // End of #ifndef HELPERS_H