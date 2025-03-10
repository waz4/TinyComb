#ifndef COMBO_H
#define COMBO_H
#include "bn.h"
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
void factorial(struct bn *n, struct bn *res);
void factorial_safe(int n, struct bn *result);

// Functions to convert number into combination,
//  - factMap is optionall and if not being used can be ignored by passing null
void id2combo(struct bn *id, int n, int k, unsigned int *combo, struct bn *factMap);
unsigned int largest(struct bn *i, int nn, int kk, struct bn *x, struct bn *factMap);
void g(int n, int k, struct bn *result, struct bn *factMap); // Calculates (n + k - 1)! / (k! * (n - 1)!) into result

// Functions to deal with a single Combo
void init_combo(unsigned int *combo, int k);
void next_combo(unsigned int *ar, unsigned int n, unsigned int k);
char combosMatch(unsigned int *comboA, unsigned int *comboB, int k);
void reverse_combo(unsigned int *combo, int k);

// Factorial Map
int getFactMapLength(int n, int k); // Returns the required factMap_size for a combinations with n elements and k slots
void makeFactMap(struct bn *factMap, int factMap_lenght);

#endif // End of #ifndef COMBO_H