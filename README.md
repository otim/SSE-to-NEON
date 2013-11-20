SSE-to-NEON
===========

(Incomplete) header file to translate SSE instructions to ARM NEON instructions

No guarantee for correctness.

Translating instructions will never be as efficient as building up your algorithm with NEON instructions, however, it can be seen as a starting point to doing so.

Usage:

  #if defined(__ARM_NEON__)
  #include "SSE to NEON/sse_to_neon.hpp"
  #else
  #include <emmintrin.h>
  #endif
