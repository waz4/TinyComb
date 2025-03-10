#include <stdio.h>
#include "combo.h"
#include "helpers.h"

int getFactMapLength(int n, int k)
{
    return n + k + 1;
}

void factorial(struct bn *n, struct bn *res)
{
    struct bn tmp;

    /* Copy n -> tmp */
    bignum_assign(&tmp, n);

    /* Decrement n by one */
    bignum_dec(n);

    /* Begin summing products: */
    while (!bignum_is_zero(n))
    {
        /* res = tmp * n */
        bignum_mul(&tmp, n, res);

        /* n -= 1 */
        bignum_dec(n);

        /* tmp = res */
        bignum_assign(&tmp, res);
    }

    /* res = tmp */
    bignum_assign(res, &tmp);
}

void factorial_safe(int n, struct bn *result)
{
    if (n <= 1)
    {
        bignum_from_int(result, 1);
        return;
    }
    struct bn n_big;
    bignum_from_int(&n_big, n);

    factorial(&n_big, result);
}

// Calculate (n + k - 1)! / (k! * (n - 1)!) into result
void g(int n, int k, struct bn *result, struct bn *factMap)
{
    if (factMap != NULL)
    {
        struct bn bot;

        if (n <= 1)
            bignum_assign(&bot, &factMap[k]);
        else
            bignum_mul(&factMap[k], &factMap[n - 1], &bot); // bot = k! * (n - 1)!

        // Actual Division
        bignum_div(&factMap[n + k - 1], &bot, result); // result = (n + k - 1)! / (k! * (n - 1)!)
    }
    else
    {
        struct bn top, bot;
        struct bn k_fact, n_minus_one_fact;

        // Top of division (n + k - 1)!
        factorial_safe((n + k - 1), &top); // top = (n + k - 1)!

        // Botttom of division (k! * (n - 1)!)
        factorial_safe(k, &k_fact);                 // k = (k)!
        factorial_safe((n - 1), &n_minus_one_fact); // n_minus_one_fact = (n - 1)!

        bignum_mul(&k_fact, &n_minus_one_fact, &bot); // bot = k! * (n - 1)!

        // Actual Division
        bignum_div(&top, &bot, result); // result = (n + k - 1)! / (k! * (n - 1)!)
    }
}

unsigned int largest(struct bn *i, int nn, int kk, struct bn *x, struct bn *factMap)
{
    g(nn, kk, x, factMap);

    while (bignum_cmp(x, i) == LARGER && nn > 0)
    {
        nn--;
        g(nn, kk, x, factMap);
    }

    return nn;
}

void reverse_combo(unsigned int *combo, int k)
{
    unsigned int tmp;

    for (int i = 0; i < k; i++, k--)
    {
        tmp = combo[k - 1];
        combo[k - 1] = combo[i];
        combo[i] = tmp;
    }
}

void id2combo(struct bn *id, int n, int k, unsigned int *combo, struct bn *factMap)
{
    if (k == 0)
        return;

    struct bn offset, new_id;
    unsigned int val;

    for (int i = k; i > 0; i--)
    {

        val = largest(id, n, i, &offset, factMap);

        combo[i - 1] = val;

        bignum_sub(id, &offset, &new_id);
        bignum_assign(id, &new_id);
    }

    reverse_combo(combo, k);
}

void init_combo(unsigned int *combo, int k)
{
    for (int i = 0; i < k; i++)
        combo[i] = 0;
}

void makeFactMap(struct bn *factMap, int factMap_size)
{
    for (int i = 0; i < factMap_size; i++)
    {
        factorial_safe(i, (factMap + i));
    }
}

void next_combo(unsigned int *ar, unsigned int n, unsigned int k)
{
    int i, lowest_i;

    for (i = lowest_i = 0; i < k; ++i)
        lowest_i = (ar[i] < ar[lowest_i]) ? i : lowest_i;

    ++ar[lowest_i];

    i = (ar[lowest_i] >= n)
            ? 0             // 0 -> all combinations have been exhausted, reset to first combination.
            : lowest_i + 1; // _ -> base incremented. digits to the right of it are now zero.

    for (; i < k; ++i)
        ar[i] = 0;
}

char combosMatch(unsigned int *comboA, unsigned int *comboB, int k)
{
    for (; k > 0; k--)
        if (comboA[k] != comboB[k])
            return 0;

    return 1;
}