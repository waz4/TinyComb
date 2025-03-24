#ifndef __BIGNUM_H__
#define __BIGNUM_H__
/*

Big number library - arithmetic on multiple-precision unsigned integers.

This library is an implementation of arithmetic on arbitrarily large integers.

The difference between this and other implementations, is that the data structure
has optimal memory utilization (i.e. a 1024 bit integer takes up 128 bytes RAM),
and all memory is allocated statically: no dynamic allocation for better or worse.

Primary goals are correctness, clarity of code and clean, portable implementation.
Secondary goal is a memory footprint small enough to make it suitable for use in
embedded applications.


The current state is correct functionality and adequate performance.
There may well be room for performance-optimizations and improvements.

*/

#include <stdint.h>
#include <assert.h>

#ifndef __NVCC__ // for CUDA/NVCC compatibility
#define __host__
#define __device__
#endif

#define WORD_SIZE 4
/* Size of big-numbers in bytes */
#define BN_ARRAY_SIZE (128 / WORD_SIZE)

#define DTYPE uint32_t
#define DTYPE_TMP uint64_t
#define DTYPE_MSB ((DTYPE_TMP)(0x80000000))
#define SPRINTF_FORMAT_STR "%.08x"
#define SSCANF_FORMAT_STR "%8x"
#define MAX_VAL ((DTYPE_TMP)0xFFFFFFFF)

/* Custom assert macro - easy to disable */
#define require(p, msg) assert(p &&msg)
// #define require(p, msg) assert()

/* Data-holding structure: array of DTYPEs */
struct bn
{
  DTYPE array[BN_ARRAY_SIZE];
};

/* Tokens returned by bignum_cmp() for value comparison */
enum
{
  SMALLER = -1,
  EQUAL = 0,
  LARGER = 1
};

/* Initialization functions: */
__host__ __device__ void bignum_init(struct bn *n);
__host__ __device__ void bignum_from_int(struct bn *n, DTYPE_TMP i);
__host__ int bignum_to_int(struct bn *n);
__host__ void bignum_from_string(struct bn *n, char *str, int nbytes);
__host__ void bignum_to_string(struct bn *n, char *str, int maxsize);

/* Basic arithmetic operations: */
__host__ __device__ void bignum_add(struct bn *a, struct bn *b, struct bn *c);                  /* c = a + b */
__host__ __device__ void bignum_sub(struct bn *a, struct bn *b, struct bn *c);                  /* c = a - b */
__host__ __device__ void bignum_mul(struct bn *a, struct bn *b, struct bn *c);                  /* c = a * b */
__host__ __device__ void bignum_div(struct bn *a, struct bn *b, struct bn *c);                  /* c = a / b */
__host__ __device__ void bignum_mod(struct bn *a, struct bn *b, struct bn *c);                  /* c = a % b */
__host__ __device__ void bignum_divmod(struct bn *a, struct bn *b, struct bn *c, struct bn *d); /* c = a/b, d = a%b */

/* Bitwise operations: */
__host__ __device__ void bignum_and(struct bn *a, struct bn *b, struct bn *c); /* c = a & b */
__host__ __device__ void bignum_or(struct bn *a, struct bn *b, struct bn *c);  /* c = a | b */
__host__ __device__ void bignum_xor(struct bn *a, struct bn *b, struct bn *c); /* c = a ^ b */
__host__ __device__ void bignum_lshift(struct bn *a, struct bn *b, int nbits); /* b = a << nbits */
__host__ __device__ void bignum_rshift(struct bn *a, struct bn *b, int nbits); /* b = a >> nbits */

/* Special operators and comparison */
__host__ __device__ int bignum_cmp(struct bn *a, struct bn *b);                /* Compare: returns LARGER, EQUAL or SMALLER */
__host__ __device__ int bignum_is_zero(struct bn *n);                          /* For comparison with zero */
__host__ __device__ void bignum_inc(struct bn *n);                             /* Increment: add one to n */
__host__ __device__ void bignum_dec(struct bn *n);                             /* Decrement: subtract one from n */
__host__ __device__ void bignum_pow(struct bn *a, struct bn *b, struct bn *c); /* Calculate a^b -- e.g. 2^10 => 1024 */
__host__ __device__ void bignum_isqrt(struct bn *a, struct bn *b);             /* Integer square root -- e.g. isqrt(5) => 2*/
__host__ __device__ void bignum_assign(struct bn *dst, struct bn *src);        /* Copy src into dst -- dst := src */

#endif /* #ifndef __BIGNUM_H__ */
