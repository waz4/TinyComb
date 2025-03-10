#include "helpers.h"

#include <stdio.h>
#include <stdlib.h>

void print_combo(unsigned int *combo, int k) {
    for (int i = 0; i < k; i++)
        printf("%d ", combo[i]);
    printf("\n");
}

void printBigNum(struct bn* num) {
    char buf[8192];
    if (bignum_is_zero(num))
        printf("0\n");
    else {
        bignum_to_string(num, buf, sizeof(buf));
        printf("%s\n", buf);
    }
}

void print_factMap(struct bn* factMap, int factMap_size) {
    printf("Factorials: \n");
    for (int i = 0; i < factMap_size; i++)
    {
        printBigNum(&factMap[i]);
    }
    
}