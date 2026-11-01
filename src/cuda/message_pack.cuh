#pragma once
#include "sha1_core.cuh"
#include <cstdint>

namespace cudadb_cuda {

constexpr int kOra = 0;
constexpr int kMs = 1;
constexpr int kMaxEncoded = 260;  // MSSQL L=128: 2*128+4
constexpr int kMaxPadded = 320;   // ceil((260+9)/64)*64

struct KernelParams {
    std::uint8_t baseDigits[128];
    std::uint8_t alphabet[256];
    std::uint8_t salt[10];
    std::uint32_t digest[5];
    std::uint16_t length;       // candidate length L
    std::uint16_t alphabetSize; // B
    std::uint64_t batchCount;
    int algorithm;              // kOra | kMs
};

// Add offset to base digits in base B (digit 0 fastest). Digits stay in range
// because the host bounds batches to the requested space.
__device__ __forceinline__ void digitsPlusOffset(const KernelParams& p, std::uint64_t offset,
                                                 std::uint8_t digits[128]) {
    const std::uint32_t B = p.alphabetSize;
    for (int i = 0; i < p.length; ++i) digits[i] = p.baseDigits[i];
    if (B <= 1) return;
    std::uint64_t carry = offset;
    int i = 0;
    while (carry && i < p.length) {
        std::uint64_t add = carry % B;
        carry /= B;
        std::uint64_t sum = std::uint64_t(digits[i]) + add;
        if (sum >= B) { sum -= B; carry += 1; }
        digits[i] = std::uint8_t(sum);
        ++i;
    }
}

// Build the encoded (pre-padding) message for a candidate password.
// Returns message length M. Buffer must hold kMaxEncoded bytes.
// The Ex form takes algorithm/length explicitly so callers never need to
// copy a whole KernelParams just to override those two fields.
__device__ __forceinline__ int buildEncodedEx(const KernelParams& p, int algo, int len,
                                              const std::uint8_t* password, std::uint8_t* msg) {
    int n = 0;
    if (algo == kOra) {
        for (int i = 0; i < len; ++i) msg[n++] = password[i];
        for (int i = 0; i < 10; ++i) msg[n++] = p.salt[i];
    } else {
        for (int i = 0; i < len; ++i) { msg[n++] = password[i]; msg[n++] = 0; }
        for (int i = 0; i < 4; ++i) msg[n++] = p.salt[i];
    }
    return n;
}
__device__ __forceinline__ int buildEncoded(const KernelParams& p, const std::uint8_t* password,
                                            std::uint8_t* msg) {
    return buildEncodedEx(p, p.algorithm, p.length, password, msg);
}

// Full generic digest: encode, pad, compress every block.
__device__ __forceinline__ ShaState digestGeneric(const KernelParams& p, const std::uint8_t password[128]) {
    std::uint8_t buf[kMaxPadded];
    int n = buildEncoded(p, password, buf);
    std::uint64_t bitLen = std::uint64_t(n) * 8;
    buf[n++] = 0x80;
    int total = ((n + 8 + 63) / 64) * 64;
    for (int i = n; i < total - 8; ++i) buf[i] = 0;
    for (int i = 0; i < 8; ++i) buf[total - 8 + i] = std::uint8_t(bitLen >> (56 - 8 * i));
    ShaState st = shaInit();
    for (int off = 0; off < total; off += 64) {
        std::uint32_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(buf[off + 4*i]) << 24) | (std::uint32_t(buf[off + 4*i + 1]) << 16) |
                   (std::uint32_t(buf[off + 4*i + 2]) << 8) | std::uint32_t(buf[off + 4*i + 3]);
        shaCompress(st, w);
    }
    return st;
}

// Prepare the 16 initial words of the single padded block for a candidate
// whose first byte is left as password[0]. Caller guarantees single block.
__device__ __forceinline__ void packSingleBlockEx(const KernelParams& p, int algo, int len,
                                                  const std::uint8_t* password, std::uint32_t w[16]) {
    std::uint8_t buf[64];
    int n = buildEncodedEx(p, algo, len, password, buf);
    std::uint64_t bitLen = std::uint64_t(n) * 8;
    buf[n++] = 0x80;
    for (int i = n; i < 56; ++i) buf[i] = 0;
    for (int i = 0; i < 8; ++i) buf[56 + i] = std::uint8_t(bitLen >> (56 - 8 * i));
#pragma unroll
    for (int i = 0; i < 16; ++i)
        w[i] = (std::uint32_t(buf[4*i]) << 24) | (std::uint32_t(buf[4*i + 1]) << 16) |
               (std::uint32_t(buf[4*i + 2]) << 8) | std::uint32_t(buf[4*i + 3]);
}
__device__ __forceinline__ void packSingleBlock(const KernelParams& p, const std::uint8_t* password,
                                                std::uint32_t w[16]) {
    packSingleBlockEx(p, p.algorithm, p.length, password, w);
}

// Encoded message length M for a candidate length, and whether it fits in
// one SHA-1 block (M <= 55).
__device__ __forceinline__ int encodedLength(const KernelParams& p) {
    return p.algorithm == kOra ? (int)p.length + 10 : 2 * (int)p.length + 4;
}

// Single-block digest without the multi-block staging buffer. Caller must
// guarantee encodedLength(p) <= 55.
__device__ __forceinline__ ShaState digestSingleBlock(const KernelParams& p, const std::uint8_t* password) {
    std::uint32_t w[16];
    packSingleBlock(p, password, w);
    ShaState st = shaInit();
    shaCompress(st, w);
    return st;
}

} // namespace cudadb_cuda
