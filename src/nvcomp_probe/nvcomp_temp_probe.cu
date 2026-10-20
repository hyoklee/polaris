// Issue 6 probe (doc/20261019_claude_terra_fusion_clio_nvcomp_hbm_polaris_plan.md):
// does nvCOMP 6's batched decompress need max_total_uncompressed_bytes in
// DecompressGetTempSizeSync? clio-core passes 0 there.
//
// usage: nvcomp_temp_probe <lz4|deflate|gdeflate|zstd|ans> <total|zero> [smooth|zeros|mixed]
// Compresses 32 x 64 KiB chunks (smooth float field, all zeros, or every 4th
// chunk zero like the stream bench's 30% zero pages), asks for the decompress
// temp size with the real total (or 0, as clio-core does), decompresses with
// exactly that much temp, and checks every chunk's status, size and bytes.
#include <cuda_runtime.h>
#include <nvcomp/ans.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s: %s (%d)\n", #x, cudaGetErrorString(e_), (int)e_); return 2; } } while (0)

template <typename CO, typename DO>
struct Codec {
  nvcompStatus_t (*max_out)(size_t, CO, size_t *);
  nvcompStatus_t (*ctemp)(size_t, size_t, CO, size_t *, size_t, cudaStream_t);
  nvcompStatus_t (*comp)(const void *const *, const size_t *, size_t, size_t, void *, size_t,
                         void *const *, size_t *, CO, nvcompStatus_t *, cudaStream_t);
  nvcompStatus_t (*dtemp)(const void *const *, const size_t *, size_t, size_t, size_t *, size_t,
                          DO, nvcompStatus_t *, cudaStream_t);
  nvcompStatus_t (*decomp)(const void *const *, const size_t *, const size_t *, size_t *, size_t,
                           void *, size_t, void *const *, DO, nvcompStatus_t *, cudaStream_t);
  CO copts;
  DO dopts;
};
#define CODEC(A) Codec<nvcompBatched##A##CompressOpts_t, nvcompBatched##A##DecompressOpts_t>{ \
  nvcompBatched##A##CompressGetMaxOutputChunkSize, nvcompBatched##A##CompressGetTempSize,  \
  nvcompBatched##A##CompressAsync, nvcompBatched##A##DecompressGetTempSizeSync,             \
  nvcompBatched##A##DecompressAsync, nvcompBatched##A##CompressDefaultOpts,                 \
  nvcompBatched##A##DecompressDefaultOpts}

constexpr size_t kChunk = 65536, kN = 32, kTotal = kChunk * kN;

template <typename C>
int Run(const char *name, C c, bool pass_total, const std::string &data) {
  cudaStream_t s; CK(cudaStreamCreate(&s));
  std::vector<float> h(kTotal / 4);
  for (size_t i = 0; i < h.size(); ++i)   // smooth field + small noise: compressible
    h[i] = 280.0f + 10.0f * std::sin(i * 1e-3f) + (float)((i * 2654435761u) % 97) * 1e-3f;
  const size_t per_chunk = kChunk / 4;
  for (size_t i = 0; i < h.size(); ++i)
    if (data == "zeros" || (data == "mixed" && (i / per_chunk) % 4 == 0)) h[i] = 0.0f;
  size_t max_c = 0; c.max_out(kChunk, c.copts, &max_c);
  char *d_in, *d_cmp, *d_out; CK(cudaMalloc(&d_in, kTotal)); CK(cudaMalloc(&d_cmp, max_c * kN));
  CK(cudaMalloc(&d_out, kTotal)); CK(cudaMemcpy(d_in, h.data(), kTotal, cudaMemcpyHostToDevice));
  std::vector<void *> hin(kN), hcmp(kN), hout(kN); std::vector<size_t> hsz(kN, kChunk);
  for (size_t i = 0; i < kN; ++i) { hin[i] = d_in + i * kChunk; hcmp[i] = d_cmp + i * max_c; hout[i] = d_out + i * kChunk; }
  void **d_pin, **d_pcmp, **d_pout; size_t *d_isz, *d_csz, *d_osz, *d_actual; nvcompStatus_t *d_st;
  CK(cudaMalloc(&d_pin, kN * 8)); CK(cudaMalloc(&d_pcmp, kN * 8)); CK(cudaMalloc(&d_pout, kN * 8));
  CK(cudaMalloc(&d_isz, kN * 8)); CK(cudaMalloc(&d_csz, kN * 8)); CK(cudaMalloc(&d_osz, kN * 8));
  CK(cudaMalloc(&d_actual, kN * 8)); CK(cudaMalloc(&d_st, kN * sizeof(nvcompStatus_t)));
  CK(cudaMemcpy(d_pin, hin.data(), kN * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pcmp, hcmp.data(), kN * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pout, hout.data(), kN * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_isz, hsz.data(), kN * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_osz, hsz.data(), kN * 8, cudaMemcpyHostToDevice));

  size_t ct = 0; nvcompStatus_t r = c.ctemp(kN, kChunk, c.copts, &ct, kTotal, s);
  void *d_ct = nullptr; if (ct) CK(cudaMalloc(&d_ct, ct));
  r = c.comp((const void *const *)d_pin, d_isz, kChunk, kN, d_ct, ct, d_pcmp, d_csz, c.copts, d_st, s);
  CK(cudaStreamSynchronize(s));
  std::vector<size_t> hcsz(kN); CK(cudaMemcpy(hcsz.data(), d_csz, kN * 8, cudaMemcpyDeviceToHost));
  size_t csum = 0; for (auto v : hcsz) csum += v;
  printf("%s compress (%s): rc=%d ratio=%.2fx csz0=%zu csz1=%zu\n", name, data.c_str(), (int)r, (double)kTotal / csum, hcsz[0], hcsz[1]);

  size_t t_zero = 0, t_total = 0;
  nvcompStatus_t r0 = c.dtemp((const void *const *)d_pcmp, d_csz, kN, kChunk, &t_zero, 0, c.dopts, d_st, s);
  nvcompStatus_t r1 = c.dtemp((const void *const *)d_pcmp, d_csz, kN, kChunk, &t_total, kTotal, c.dopts, d_st, s);
  printf("%s decompress temp: total=0 -> %zu B (rc=%d), total=%zu -> %zu B (rc=%d)\n",
         name, t_zero, (int)r0, kTotal, t_total, (int)r1);

  // PROBE_SHIFT=k moves every compressed chunk k bytes past its aligned
  // slot, as a blob packed after an odd-sized one would be.
  const size_t shift = getenv("PROBE_SHIFT") ? strtoul(getenv("PROBE_SHIFT"), nullptr, 10) : 0;
  if (shift) {
    char *d_cmp2; CK(cudaMalloc(&d_cmp2, (max_c + 64) * kN));
    for (size_t i = 0; i < kN; ++i) {
      hcmp[i] = d_cmp2 + i * (max_c + 64) + shift;
      CK(cudaMemcpy(hcmp[i], d_cmp + i * max_c, hcsz[i], cudaMemcpyDeviceToDevice));
    }
    CK(cudaMemcpy(d_pcmp, hcmp.data(), kN * 8, cudaMemcpyHostToDevice));
    printf("%s inputs shifted by %zu B\n", name, shift);
  }
  size_t tb = pass_total ? t_total : t_zero;
  void *d_dt = nullptr; if (tb) CK(cudaMalloc(&d_dt, tb));
  CK(cudaMemset(d_out, 0, kTotal)); CK(cudaMemset(d_st, 0xff, kN * sizeof(nvcompStatus_t)));
  r = c.decomp((const void *const *)d_pcmp, d_csz, d_osz, d_actual, kN, d_dt, tb, d_pout, c.dopts, d_st, s);
  cudaError_t e = cudaStreamSynchronize(s);
  printf("%s decompress (%s temp = %zu B): rc=%d sync=%s\n", name, pass_total ? "real-total" : "zero-total",
         tb, (int)r, cudaGetErrorString(e));
  if (e != cudaSuccess) { printf("RESULT %s %s %s FAIL(cuda %d)\n", name, data.c_str(), pass_total ? "total" : "zero", (int)e); return 1; }
  std::vector<nvcompStatus_t> hst(kN); std::vector<size_t> hact(kN); std::vector<float> back(h.size());
  CK(cudaMemcpy(hst.data(), d_st, kN * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hact.data(), d_actual, kN * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(back.data(), d_out, kTotal, cudaMemcpyDeviceToHost));
  size_t bad = 0; for (size_t i = 0; i < kN; ++i) bad += (hst[i] != nvcompSuccess || hact[i] != kChunk);
  bool same = std::memcmp(back.data(), h.data(), kTotal) == 0;
  printf("%s chunks bad=%zu/%zu st0=%d st1=%d actual0=%zu bytes_match=%d\n", name, bad, kN, (int)hst[0], (int)hst[1], hact[0], (int)same);
  printf("RESULT %s %s %s %s\n", name, data.c_str(), pass_total ? "total" : "zero", (bad == 0 && same) ? "OK" : "FAIL");
  return (bad == 0 && same) ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 3) { printf("usage: %s <codec> <total|zero>\n", argv[0]); return 2; }
  std::string a = argv[1]; bool t = std::string(argv[2]) == "total";
  std::string d = argc > 3 ? argv[3] : "smooth";
  if (a == "lz4") return Run("lz4", CODEC(LZ4), t, d);
  if (a == "deflate") return Run("deflate", CODEC(Deflate), t, d);
  if (a == "gdeflate") return Run("gdeflate", CODEC(Gdeflate), t, d);
  if (a == "zstd") return Run("zstd", CODEC(Zstd), t, d);
  if (a == "ans") return Run("ans", CODEC(ANS), t, d);
  return 2;
}
