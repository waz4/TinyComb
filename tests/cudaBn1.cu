/*

    Testing Big-Number library by calculating factorial(100) a.k.a. 100!
    ====================================================================

    For the uninitiated:

        factorial(N) := N * (N-1) * (N-2) * ... * 1


    Example:

        factorial(5) = 5 * 4 * 3 * 2 * 1 = 120



    Validated by Python implementation of big-numbers:
    --------------------------------------------------

        In [1]: import math

        In [2]: "%x" % math.factorial(100)
        Out[]: '1b30964ec395dc24069528d54bbda40d16e966ef9a70eb21b5b2943a321cdf10391745570cca9420c6ecb3b72ed2ee8b02ea2735c61a000000000000000000000000'


    ... which should also be the result of this program's calculation


*/

#include <stdio.h>
#include "../src/bn.h"

__global__ void factorial()
{
  printf("Inside kernel");
  struct bn num;
  struct bn result;

  bignum_from_int(&num, 100);

  struct bn tmp;

  /* Copy n -> tmp */
  bignum_assign(&tmp, &num);

  /* Decrement n by one */
  bignum_dec(&num);

  /* Begin summing products: */
  while (!bignum_is_zero(&num))
  {
    /* res = tmp * n */
    bignum_mul(&tmp, &num, &result);

    /* n -= 1 */
    bignum_dec(&num);

    /* tmp = res */
    bignum_assign(&tmp, &result);
  }

  /* res = tmp */
  bignum_assign(&result, &tmp);
}

__global__ void fml()
{
  printf("Fadkwajdawjda");
}

int main()
{
  printf("Test");
  factorial<<<128, 1024>>>();
  fml<<<1, 10>>>();
  printf("ADWADWADAW");

  return 0;
}
