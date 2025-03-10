// Test if id2combo is matching nextCombo

#include <stdio.h>
#include "../src/bn.h"
#include "../src/combo.h"
#include "../src/helpers.h"

void write_combo(FILE *fp, unsigned int *combo, int combo_size)
{
    for (int i = 0; i < combo_size; i++)
    {
        fprintf(fp, "%d ", combo[i]);
    }
    fprintf(fp, "\n");
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s sequential_combinations_output_path id_combinations_output_path", argv[0]);
        return -1;
    }
    char *sequential_output_file_path = argv[1];
    char *id_output_file_path = argv[2];

    int n = 16;
    int k = 4;
    struct bn number, nr_of_combinations_big;

    unsigned int id_combo[k], sequential_combo[k];
    init_combo(id_combo, k);
    init_combo(sequential_combo, k);

    int nr_of_factorials = getFactMapLength(n, k);
    struct bn factMap[nr_of_factorials];
    makeFactMap(factMap, nr_of_factorials);

    g(n, k, &nr_of_combinations_big, factMap);
    unsigned int nr_of_combinations = bignum_to_int(&nr_of_combinations_big);

    FILE *sequential_fp = fopen(sequential_output_file_path, "w+");
    FILE *id_fp = fopen(id_output_file_path, "w+");

    if (sequential_fp == NULL || id_fp == NULL)
    {
        printf("Error opening save files.\n");
        return -1;
    }

    printf("Making combinationList files with %d entries.\n", nr_of_combinations);

    printf("Starting...\n");
    for (int i = 0; i < nr_of_combinations; i++)
    {
        // printf("Current Index: %d\n", i);

        bignum_from_int(&number, i);
        id2combo(&number, n, k, id_combo, factMap);

        // printf("Id combo:\t\t");
        // print_combo(id_combo, k);
        // printf("Sequential combo: \t");
        // print_combo(sequential_combo, k);

        write_combo(id_fp, id_combo, k);
        write_combo(sequential_fp, sequential_combo, k);

        next_combo(sequential_combo, n, k);
    }
    printf("Finished.\n");

    fclose(id_fp);
    fclose(sequential_fp);
    return 0;
}
