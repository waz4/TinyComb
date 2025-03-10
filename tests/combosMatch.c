// Test if id2combo is matching nextCombo

#include <stdio.h>
#include "../src/bn.h"
#include "../src/combo.h"
#include "../src/helpers.h"

int main()
{
    int n = 5;
    int k = 3;
    struct bn number, nr_of_combinations_big;

    unsigned int id_combo[k], sequential_combo[k];
    init_combo(id_combo, k);
    init_combo(sequential_combo, k);

    int nr_of_factorials = getFactMapLength(n, k);
    struct bn factMap[nr_of_factorials];
    makeFactMap(factMap, nr_of_factorials);

    g(n, k, &nr_of_combinations_big, factMap);
    unsigned int nr_of_combinations = bignum_to_int(&nr_of_combinations_big);

    for (int i = 0; i < nr_of_combinations; i++)
    {
        // printf("Current Index: %d\n", i);

        bignum_from_int(&number, i);
        id2combo(&number, n, k, id_combo, factMap);

        // printf("Id combo:\t\t");
        // print_combo(id_combo, k);
        // printf("Sequential combo: \t");
        // print_combo(sequential_combo, k);

        if (!combosMatch(id_combo, sequential_combo, k))
        {
            printf("Combos Match test failed.\n N= %d, K= %d, Id= %d", n, k, i);
            return -1;
        }

        next_combo(sequential_combo, n, k);
    }

    printf("CombosMatch test Passed\n");
    return 0;
}
