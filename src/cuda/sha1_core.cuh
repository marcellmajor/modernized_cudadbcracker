#pragma once
#include <cstdint>

namespace cudadb_cuda {

__device__ __forceinline__ std::uint32_t rol32(std::uint32_t x, int y) {
    return (x << y) | (x >> (32 - y));
}

struct ShaState {
    std::uint32_t h[5];
};

__device__ __forceinline__ ShaState shaInit() {
    ShaState s;
    s.h[0] = 0x67452301u; s.h[1] = 0xEFCDAB89u; s.h[2] = 0x98BADCFEu;
    s.h[3] = 0x10325476u; s.h[4] = 0xC3D2E1F0u;
    return s;
}

// One compression over 16 initial words; full schedule recurrence (spec 8).
__device__ __forceinline__ void shaCompress(ShaState& st, const std::uint32_t initW[16]) {
    std::uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = initW[i];
    std::uint32_t a = st.h[0], b = st.h[1], c = st.h[2], d = st.h[3], e = st.h[4];
#pragma unroll
    for (int i = 0; i < 80; ++i) {
        if (i >= 16) {
            int j = i & 15;
            w[j] = rol32(w[(i - 3) & 15] ^ w[(i - 8) & 15] ^ w[(i - 14) & 15] ^ w[j], 1);
        }
        std::uint32_t f, k;
        if (i < 20)      { f = (b & c) | ((~b) & d);        k = 0x5A827999u; }
        else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
        else             { f = b ^ c ^ d;                   k = 0xCA62C1D6u; }
        std::uint32_t tmp = rol32(a, 5) + f + e + k + w[i & 15];
        e = d; d = c; c = rol32(b, 30); b = a; a = tmp;
    }
    st.h[0] += a; st.h[1] += b; st.h[2] += c; st.h[3] += d; st.h[4] += e;
}

// Two independent compressions stepped round-by-round (spec 9.3).
__device__ __forceinline__ void shaCompressPair(ShaState& s0, const std::uint32_t w0init[16],
                                                ShaState& s1, const std::uint32_t w1init[16]) {
    std::uint32_t w[2][16];
#pragma unroll
    for (int i = 0; i < 16; ++i) { w[0][i] = w0init[i]; w[1][i] = w1init[i]; }
    std::uint32_t a[2] = {s0.h[0], s1.h[0]}, b[2] = {s0.h[1], s1.h[1]}, c[2] = {s0.h[2], s1.h[2]},
                  d[2] = {s0.h[3], s1.h[3]}, e[2] = {s0.h[4], s1.h[4]};
#pragma unroll
    for (int i = 0; i < 80; ++i) {
        if (i >= 16) {
            int j = i & 15;
#pragma unroll
            for (int q = 0; q < 2; ++q) w[q][j] = rol32(w[q][(i - 3) & 15] ^ w[q][(i - 8) & 15] ^ w[q][(i - 14) & 15] ^ w[q][j], 1);
        }
        std::uint32_t f, k;
        if (i < 20)      { k = 0x5A827999u; }
        else if (i < 40) { k = 0x6ED9EBA1u; }
        else if (i < 60) { k = 0x8F1BBCDCu; }
        else             { k = 0xCA62C1D6u; }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            if (i < 20) f = (b[q] & c[q]) | ((~b[q]) & d[q]);
            else if (i < 40) f = b[q] ^ c[q] ^ d[q];
            else if (i < 60) f = (b[q] & c[q]) | (b[q] & d[q]) | (c[q] & d[q]);
            else f = b[q] ^ c[q] ^ d[q];
            std::uint32_t tmp = rol32(a[q], 5) + f + e[q] + k + w[q][i & 15];
            e[q] = d[q]; d[q] = c[q]; c[q] = rol32(b[q], 30); b[q] = a[q]; a[q] = tmp;
        }
    }
    s0.h[0] += a[0]; s0.h[1] += b[0]; s0.h[2] += c[0]; s0.h[3] += d[0]; s0.h[4] += e[0];
    s1.h[0] += a[1]; s1.h[1] += b[1]; s1.h[2] += c[1]; s1.h[3] += d[1]; s1.h[4] += e[1];
}

// Four independent compressions stepped round-by-round (spec 9.3,
// optional factor-4 experiment). Register-heavy: opt-in only, selected
// by measurement, never the default.
__device__ __forceinline__ void shaCompressQuad(ShaState s[4], const std::uint32_t winit[4][16]) {
    std::uint32_t w[4][16];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        for (int i = 0; i < 16; ++i) w[q][i] = winit[q][i];
    std::uint32_t a[4], b[4], c[4], d[4], e[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) { a[q] = s[q].h[0]; b[q] = s[q].h[1]; c[q] = s[q].h[2]; d[q] = s[q].h[3]; e[q] = s[q].h[4]; }
#pragma unroll
    for (int i = 0; i < 80; ++i) {
        if (i >= 16) {
            int j = i & 15;
#pragma unroll
            for (int q = 0; q < 4; ++q) w[q][j] = rol32(w[q][(i - 3) & 15] ^ w[q][(i - 8) & 15] ^ w[q][(i - 14) & 15] ^ w[q][j], 1);
        }
        std::uint32_t k;
        if (i < 20) k = 0x5A827999u; else if (i < 40) k = 0x6ED9EBA1u;
        else if (i < 60) k = 0x8F1BBCDCu; else k = 0xCA62C1D6u;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            std::uint32_t f;
            if (i < 20) f = (b[q] & c[q]) | ((~b[q]) & d[q]);
            else if (i < 40) f = b[q] ^ c[q] ^ d[q];
            else if (i < 60) f = (b[q] & c[q]) | (b[q] & d[q]) | (c[q] & d[q]);
            else f = b[q] ^ c[q] ^ d[q];
            std::uint32_t tmp = rol32(a[q], 5) + f + e[q] + k + w[q][i & 15];
            e[q] = d[q]; d[q] = c[q]; c[q] = rol32(b[q], 30); b[q] = a[q]; a[q] = tmp;
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) { s[q].h[0] += a[q]; s[q].h[1] += b[q]; s[q].h[2] += c[q]; s[q].h[3] += d[q]; s[q].h[4] += e[q]; }
}

__device__ __forceinline__ bool stateEquals(const ShaState& s, const std::uint32_t digest[5]) {
    return s.h[0] == digest[0] && s.h[1] == digest[1] && s.h[2] == digest[2] &&
           s.h[3] == digest[3] && s.h[4] == digest[4];
}

} // namespace cudadb_cuda
