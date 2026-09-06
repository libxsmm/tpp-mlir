//===- TensorInitRNG.h - Deterministic tensor initialization RNG ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reproducible pseudo random numbers for seeded tensor initialization.
//
// The C++ standard only fixes the statistical properties of the
// std::*_distribution templates, not the value sequence they produce for a
// given seed, and std::default_random_engine is implementation defined too.
// Seeded output therefore changes with the standard library version, which
// makes tests that check exact generated values fail when the same sources are
// built against a different libstdc++ (e.g. Fedora vs. Rocky Linux).
//
// This engine is the xoshiro128+ generator used by libxsmm (see
// libxsmm/src/libxsmm_rng.c): same 16-lane state, same seed table, same jump
// and same float conversion, so `uniformFloat()` reproduces
// `libxsmm_rng_f32_seq()` element for element. It is kept self contained and
// per instance to avoid libxsmm's global RNG state, and the distributions on
// top of it only use exactly rounded IEEE-754 operations so that a seed maps
// to the same values on any platform, compiler, standard library and CPU.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_TENSORINITRNG_H
#define TPP_TRANSFORMS_UTILS_TENSORINITRNG_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

// xoshiro128+ generator, bit compatible with libxsmm's `libxsmm_rng_f32_seq`.
class TensorInitRNG {
public:
  explicit TensorInitRNG(uint32_t seed) {
    // Seed table and jump replicated from libxsmm's internal_rng_set_seed_sw.
    static const uint32_t seedTable[numStates][numLanes] = {
        {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16},
        {131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118,
         117, 116},
        {231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218,
         217, 216},
        {331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318,
         317, 316}};

    for (unsigned s = 0; s < numStates; ++s)
      for (unsigned i = 0; i < numLanes; ++i)
        state[s][i] = seed + seedTable[s][i];
    for (unsigned i = 0; i < numLanes; ++i)
      jump(i);
  }

  // Uniform float in [0, 1).
  float uniformFloat() {
    // Take the upper bits as the mantissa of a number in [1, 2), as suggested
    // by the xoshiro authors for float generation.
    uint32_t mantissa = nextUInt() >> 9;
    uint32_t bits = 0x3f800000u | mantissa;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value - 1.0f;
  }

  // Uniform integer in [lo, hi].
  uint64_t uniformInt(uint64_t lo, uint64_t hi) {
    assert(lo <= hi && "Invalid range");
    uint64_t range = hi - lo;
    if (range == 0)
      return lo;
    if (range == std::numeric_limits<uint64_t>::max())
      return nextUInt64();

    // Reject the incomplete bucket at the bottom to keep the mapping unbiased.
    uint64_t bound = range + 1;
    uint64_t threshold = (0u - bound) % bound;
    uint64_t value;
    do {
      value = nextUInt64();
    } while (value < threshold);
    return lo + value % bound;
  }

  // Normal distribution, approximated with the Irwin-Hall construction.
  // Transcendental alternatives (Box-Muller, Marsaglia polar) would reintroduce
  // the portability problem because libm is not bit reproducible across
  // versions, while summing uniforms only needs exact IEEE-754 additions.
  float normalFloat(float mean, float stddev) {
    double sum = 0.0;
    for (unsigned i = 0; i < irwinHallTerms; ++i)
      sum += uniformFloat();
    // Sum of n uniforms has mean n/2 and variance n/12, so n = 12 normalises to
    // unit variance and only needs the mean subtracted.
    double normalized = sum - (irwinHallTerms / 2.0);
    return static_cast<float>(mean + stddev * normalized);
  }

  // Binomial distribution, drawn as a sum of Bernoulli trials.
  uint64_t binomial(uint64_t trials, double probability) {
    uint64_t successes = 0;
    for (uint64_t i = 0; i < trials; ++i)
      if (uniformFloat() < probability)
        ++successes;
    return successes;
  }

private:
  static constexpr unsigned numStates = 4;
  static constexpr unsigned numLanes = 16;
  static constexpr unsigned irwinHallTerms = 12;

  uint32_t state[numStates][numLanes];
  // Draws round robin over the lanes, matching libxsmm's sequence layout.
  unsigned lane = 0;

  uint32_t nextUInt() {
    unsigned i = lane;
    lane = (lane + 1) % numLanes;

    uint32_t result = state[0][i] + state[3][i];
    uint32_t t = state[1][i] << 9;
    state[2][i] ^= state[0][i];
    state[3][i] ^= state[1][i];
    state[1][i] ^= state[2][i];
    state[0][i] ^= state[3][i];
    state[2][i] ^= t;
    state[3][i] = (state[3][i] << 11) | (state[3][i] >> (32 - 11));
    return result;
  }

  uint64_t nextUInt64() {
    uint64_t hi = nextUInt();
    uint64_t lo = nextUInt();
    return (hi << 32) | lo;
  }

  // Advance one lane by 2^64 draws so that the lanes do not overlap.
  void jump(unsigned i) {
    static const uint32_t jumpTable[numStates] = {0x8764000b, 0xf542d2d3,
                                                  0x6fa035c3, 0x77f2db5b};
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (unsigned w = 0; w < numStates; ++w) {
      for (unsigned b = 0; b < 32; ++b) {
        if (jumpTable[w] & (1U << b)) {
          s0 ^= state[0][i];
          s1 ^= state[1][i];
          s2 ^= state[2][i];
          s3 ^= state[3][i];
        }
        uint32_t t = state[1][i] << 9;
        state[2][i] ^= state[0][i];
        state[3][i] ^= state[1][i];
        state[1][i] ^= state[2][i];
        state[0][i] ^= state[3][i];
        state[2][i] ^= t;
        state[3][i] = (state[3][i] << 11) | (state[3][i] >> (32 - 11));
      }
    }
    state[0][i] = s0;
    state[1][i] = s1;
    state[2][i] = s2;
    state[3][i] = s3;
  }
};

#endif // TPP_TRANSFORMS_UTILS_TENSORINITRNG_H
