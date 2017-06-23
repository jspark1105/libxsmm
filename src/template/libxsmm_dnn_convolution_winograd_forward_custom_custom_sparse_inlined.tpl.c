/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/

int ltid;
int work;
int chunksize;
int thr_begin;
int thr_end;
int job;
int img;
int ifm1;
int ofm1;
int oj;
int oi;
unsigned int i, j, k, l;

const int JTILES = 4, ITILES = 4;
const int BIMG = 1;

const int K = 384, C = 256; // conv3
//const int K = 192, C = 192; // conv4
//const int K = 128, C = 192; // conv5

LIBXSMM_VLA_DECL(5, const float, input,  (const float*)handle->reg_input->data, C, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, (float*)handle->reg_output->data, K, ALPHA-2, ALPHA-2, TDVLEN);

// The main part of Winograd convolution is
// ALPHA x ALPHA independent GEMMs
// Each GEMM is K x C x (bimg x jtiles x itiles)

// In Kunal's implementation,
// Each GEMM is implemented with
// (K/TDVLEN) x (C/TDVLEN) small GEMMs with each size
// TDVLEN x TDVLEN x (bimg x jtiles x itiles)

#define BOFM (1)

// V is N/bimg x ALPHA x ALPHA x C x bimg x jtiles x itiles instead of N/bimg x ALPHA x ALPHA x C/TDVLEN x bimg x jtiles x itiles x TDVLEN
LIBXSMM_VLA_DECL(7, float, V,   (float*)handle->scratch3, C, BIMG, ALPHA, ALPHA, JTILES, ITILES);
// M is N/bimg x ALPHA x ALPHA x K x bimg x jtiles x itiles instead of N/bimg x ALPHA x ALPHA x K/TDVLEN x bimg x jtiles x itiles x TDVLEN
LIBXSMM_VLA_DECL(7, float, M,   (float*)handle->scratch4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Iwp, (float*)handle->scratchIw, JTILES*ITILES, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Owp, (float*)handle->scratchOw, JTILES*ITILES, ALPHA, ALPHA, TDVLEN);
#if 1
typedef libxsmm_sconvfunction libxsmm_convfunction;
libxsmm_convfunction jitted_conv_fp = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
#endif
LIBXSMM_ASSUME_ALIGNED(handle->reg_input->data,  64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_output->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_filter->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch1, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch3, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch4, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratchIw, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratchOw, 64);

/* computing first logical thread */
ltid = tid - start_thread;
//libxsmm_barrier_init((libxsmm_barrier*)handle->barrier, ltid);

//#define FTIME
#ifdef FTIME
unsigned long long t_input  = 0;
unsigned long long t_wt     = 0;
unsigned long long t_output = 0;
unsigned long long t_gemm   = 0;
unsigned long long t_start  = 0;
#endif

#if 0 // def SEP
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
static int sep_cnt = 0;
if (0 == ltid) {
  ++sep_cnt;
  if (10 == sep_cnt) VTResumeSampling();
}
#endif

/* number of tasks that could be run in parallel */
work = LIBXSMM_MAX(handle->desc.N, handle->desc.threads);
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
int thr_per_img = LIBXSMM_MAX(1, handle->desc.threads/handle->desc.N);

// TODO: change input and output format so that we can vectorize over tiles

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / thr_per_img;
  ifm1 = (job % thr_per_img)*(handle->blocksifm/thr_per_img);
//  ifm1 = job % handle->blocksifm;

  const float *inp = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1*TDVLEN, 0, 0, 0, C, ALPHA, ALPHA, TDVLEN);
  float *tinp = &LIBXSMM_VLA_ACCESS(7, V, img/BIMG, ifm1*TDVLEN, img%BIMG, 0, 0, 0, 0, C, BIMG, ALPHA, ALPHA, JTILES, ITILES);
  float *Iwp2 = &LIBXSMM_VLA_ACCESS(5, Iwp, tid, 0, 0, 0, 0, ITILES*JTILES, ALPHA, ALPHA, TDVLEN);

  const int total_tiles = JTILES*ITILES;
  LIBXSMM_VLA_DECL(4, const float, input, inp, ALPHA, ALPHA, TDVLEN);
  LIBXSMM_VLA_DECL(5, float, output, tinp, BIMG, ALPHA, ALPHA, total_tiles);
  LIBXSMM_VLA_DECL(4, float, Iw, Iwp2, ALPHA, ALPHA, TDVLEN);
  unsigned int ti, tj;
  int i, j;
  int xdim, ydim;

#ifdef __AVX512F__

#define SIMDTYPE_FP32 __m512
#define _MM_SETZERO_FP32 _mm512_setzero_ps
#define _MM_SET1_FP32 _mm512_set1_ps
#define SIMD_PER_LINE (1)

#elif defined(__AVX2__)

#define SIMDTYPE_FP32 __m256
#define _MM_SETZERO_FP32 _mm256_setzero_ps
#define _MM_SET1_FP32 _mm256_set1_ps
#define SIMD_PER_LINE (2)

#else

#define SIMDTYPE_FP32 __m128
#define _MM_SETZERO_FP32 _mm_setzero_ps
#define _MM_SET1_FP32 _mm_set1_ps
#define SIMD_PER_LINE (4)

#endif

  SIMDTYPE_FP32 T[ALPHA][ALPHA][SIMD_PER_LINE];
  SIMDTYPE_FP32
    t0[SIMD_PER_LINE], t1[SIMD_PER_LINE],
    t2[SIMD_PER_LINE], t3[SIMD_PER_LINE],
    t4[SIMD_PER_LINE], t5[SIMD_PER_LINE];
  SIMDTYPE_FP32 I[ALPHA][SIMD_PER_LINE];

  int ifm2;
  for (ifm2 = 0; ifm2 < handle->blocksifm*TDVLEN/thr_per_img; ifm2++) {
#ifdef __AVX512F__
    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (i = 0; i < ALPHA; i++) {
      I[0][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 0, ALPHA, ALPHA, TDVLEN));
      I[1][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 0, ALPHA, ALPHA, TDVLEN));
      I[2][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 0, ALPHA, ALPHA, TDVLEN));
      I[3][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 0, ALPHA, ALPHA, TDVLEN));
      I[4][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 0, ALPHA, ALPHA, TDVLEN));
      I[5][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 0, ALPHA, ALPHA, TDVLEN));

      t0[0] = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), I[2][0], I[4][0]);
      t1[0] = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), I[1][0], I[3][0]);
      t2[0] = _mm512_sub_ps(I[4][0], I[2][0]);
      t3[0] = _mm512_sub_ps(I[3][0], I[1][0]);
      t4[0] = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), I[2][0], I[4][0]);
      t5[0] = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), I[3][0], I[5][0]);

      T[0][i][0] = _mm512_fmadd_ps(_mm512_set1_ps(4.0f), I[0][0], t4[0]);
      T[1][i][0] = _mm512_add_ps(t0[0], t1[0]);
      T[2][i][0] = _mm512_sub_ps(t0[0], t1[0]);
      T[3][i][0] = _mm512_fmadd_ps(_mm512_set1_ps(2.0f), t3[0], t2[0]);
      T[4][i][0] = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), t3[0], t2[0]);
      T[5][i][0] = _mm512_fmadd_ps(_mm512_set1_ps(4.0f), I[1][0], t5[0]);
    }

    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (j = 0; j < ALPHA; j++) {
      t0[0] = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), T[j][2][0], T[j][4][0]);
      t1[0] = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), T[j][1][0], T[j][3][0]);
      t2[0] = _mm512_sub_ps(T[j][4][0], T[j][2][0]);
      t3[0] = _mm512_sub_ps(T[j][3][0], T[j][1][0]);
      t4[0] = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), T[j][2][0], T[j][4][0]);
      t5[0] = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), T[j][3][0], T[j][5][0]);

      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fmadd_ps(_mm512_set1_ps(4.0f), T[j][0][0], t4[0]));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_add_ps(t0[0], t1[0]));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_sub_ps(t0[0], t1[0]));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fmadd_ps(_mm512_set1_ps(2.0f), t3[0], t2[0]));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), t3[0], t2[0]));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fmadd_ps(_mm512_set1_ps(4.0f), T[j][1][0], t5[0]));
    }
#elif defined(__AVX2__)
    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (i = 0; i < ALPHA; i++) {
      I[0][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 0, ALPHA, ALPHA, TDVLEN));
      I[0][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 8, ALPHA, ALPHA, TDVLEN));
      I[1][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 0, ALPHA, ALPHA, TDVLEN));
      I[1][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 8, ALPHA, ALPHA, TDVLEN));
      I[2][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 0, ALPHA, ALPHA, TDVLEN));
      I[2][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 8, ALPHA, ALPHA, TDVLEN));
      I[3][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 0, ALPHA, ALPHA, TDVLEN));
      I[3][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 8, ALPHA, ALPHA, TDVLEN));
      I[4][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 0, ALPHA, ALPHA, TDVLEN));
      I[4][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 8, ALPHA, ALPHA, TDVLEN));
      I[5][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 0, ALPHA, ALPHA, TDVLEN));
      I[5][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 8, ALPHA, ALPHA, TDVLEN));

      t0[0] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), I[2][0], I[4][0]);
      t1[0] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), I[1][0], I[3][0]);
      t2[0] = _mm256_sub_ps(I[4][0], I[2][0]);
      t3[0] = _mm256_sub_ps(I[3][0], I[1][0]);
      t4[0] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), I[2][0], I[4][0]);
      t5[0] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), I[3][0], I[5][0]);

      t0[1] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), I[2][1], I[4][1]);
      t1[1] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), I[1][1], I[3][1]);
      t2[1] = _mm256_sub_ps(I[4][1], I[2][1]);
      t3[1] = _mm256_sub_ps(I[3][1], I[1][1]);
      t4[1] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), I[2][1], I[4][1]);
      t5[1] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), I[3][1], I[5][1]);

      T[0][i][0] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), I[0][0], t4[0]);
      T[1][i][0] = _mm256_add_ps(t0[0], t1[0]);
      T[2][i][0] = _mm256_sub_ps(t0[0], t1[0]);
      T[3][i][0] = _mm256_fmadd_ps(_mm256_set1_ps(2.0f), t3[0], t2[0]);
      T[4][i][0] = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), t3[0], t2[0]);
      T[5][i][0] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), I[1][0], t5[0]);

      T[0][i][1] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), I[0][1], t4[1]);
      T[1][i][1] = _mm256_add_ps(t0[1], t1[1]);
      T[2][i][1] = _mm256_sub_ps(t0[1], t1[1]);
      T[3][i][1] = _mm256_fmadd_ps(_mm256_set1_ps(2.0f), t3[1], t2[1]);
      T[4][i][1] = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), t3[1], t2[1]);
      T[5][i][1] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), I[1][1], t5[1]);
    }

    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (j = 0; j < ALPHA; j++) {
      t0[0] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), T[j][2][0], T[j][4][0]);
      t1[0] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), T[j][1][0], T[j][3][0]);
      t2[0] = _mm256_sub_ps(T[j][4][0], T[j][2][0]);
      t3[0] = _mm256_sub_ps(T[j][3][0], T[j][1][0]);
      t4[0] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), T[j][2][0], T[j][4][0]);
      t5[0] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), T[j][3][0], T[j][5][0]);

      t0[1] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), T[j][2][1], T[j][4][1]);
      t1[1] = _mm256_fnmadd_ps(_mm256_set1_ps(4.0f), T[j][1][1], T[j][3][1]);
      t2[1] = _mm256_sub_ps(T[j][4][1], T[j][2][1]);
      t3[1] = _mm256_sub_ps(T[j][3][1], T[j][1][1]);
      t4[1] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), T[j][2][1], T[j][4][1]);
      t5[1] = _mm256_fnmadd_ps(_mm256_set1_ps(5.0f), T[j][3][1], T[j][5][1]);

      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fmadd_ps(_mm256_set1_ps(4.0f), T[j][0][0], t4[0]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_add_ps(t0[0], t1[0]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_sub_ps(t0[0], t1[0]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fmadd_ps(_mm256_set1_ps(2.0f), t3[0], t2[0]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), t3[0], t2[0]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fmadd_ps(_mm256_set1_ps(4.0f), T[j][1][0], t5[0]));

      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fmadd_ps(_mm256_set1_ps(4.0f), T[j][0][1], t4[1]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_add_ps(t0[1], t1[1]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_sub_ps(t0[1], t1[1]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fmadd_ps(_mm256_set1_ps(2.0f), t3[1], t2[1]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), t3[1], t2[1]));
      _mm256_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm256_fmadd_ps(_mm256_set1_ps(4.0f), T[j][1][1], t5[1]));
    }
#else
    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (i = 0; i < ALPHA; i++) {
      I[0][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 0, ALPHA, ALPHA, TDVLEN));
      I[0][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 4, ALPHA, ALPHA, TDVLEN));
      I[0][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 8, ALPHA, ALPHA, TDVLEN));
      I[0][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 12, ALPHA, ALPHA, TDVLEN));

      I[1][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 0, ALPHA, ALPHA, TDVLEN));
      I[1][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 4, ALPHA, ALPHA, TDVLEN));
      I[1][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 8, ALPHA, ALPHA, TDVLEN));
      I[1][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 12, ALPHA, ALPHA, TDVLEN));

      I[2][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 0, ALPHA, ALPHA, TDVLEN));
      I[2][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 4, ALPHA, ALPHA, TDVLEN));
      I[2][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 8, ALPHA, ALPHA, TDVLEN));
      I[2][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 12, ALPHA, ALPHA, TDVLEN));

      I[3][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 0, ALPHA, ALPHA, TDVLEN));
      I[3][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 4, ALPHA, ALPHA, TDVLEN));
      I[3][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 8, ALPHA, ALPHA, TDVLEN));
      I[3][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 12, ALPHA, ALPHA, TDVLEN));

      I[4][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 0, ALPHA, ALPHA, TDVLEN));
      I[4][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 4, ALPHA, ALPHA, TDVLEN));
      I[4][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 8, ALPHA, ALPHA, TDVLEN));
      I[4][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 12, ALPHA, ALPHA, TDVLEN));

      I[5][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 0, ALPHA, ALPHA, TDVLEN));
      I[5][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 4, ALPHA, ALPHA, TDVLEN));
      I[5][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 8, ALPHA, ALPHA, TDVLEN));
      I[5][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 12, ALPHA, ALPHA, TDVLEN));

#define _mm_fmadd_ps(a, b, c)  _mm_add_ps(c, _mm_mul_ps(a, b))
#define _mm_fnmadd_ps(a, b, c) _mm_sub_ps(c, _mm_mul_ps(a, b))

      t0[0] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[2][0], I[4][0]);
      t1[0] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[1][0], I[3][0]);
      t2[0] = _mm_sub_ps(I[4][0], I[2][0]);
      t3[0] = _mm_sub_ps(I[3][0], I[1][0]);
      t4[0] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[2][0], I[4][0]);
      t5[0] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[3][0], I[5][0]);

      t0[1] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[2][1], I[4][1]);
      t1[1] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[1][1], I[3][1]);
      t2[1] = _mm_sub_ps(I[4][1], I[2][1]);
      t3[1] = _mm_sub_ps(I[3][1], I[1][1]);
      t4[1] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[2][1], I[4][1]);
      t5[1] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[3][1], I[5][1]);

      t0[2] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[2][2], I[4][2]);
      t1[2] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[1][2], I[3][2]);
      t2[2] = _mm_sub_ps(I[4][2], I[2][2]);
      t3[2] = _mm_sub_ps(I[3][2], I[1][2]);
      t4[2] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[2][2], I[4][2]);
      t5[2] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[3][2], I[5][2]);

      t0[3] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[2][3], I[4][3]);
      t1[3] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), I[1][3], I[3][3]);
      t2[3] = _mm_sub_ps(I[4][3], I[2][3]);
      t3[3] = _mm_sub_ps(I[3][3], I[1][3]);
      t4[3] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[2][3], I[4][3]);
      t5[3] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), I[3][3], I[5][3]);

      T[0][i][0] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[0][0], t4[0]);
      T[1][i][0] = _mm_add_ps(t0[0], t1[0]);
      T[2][i][0] = _mm_sub_ps(t0[0], t1[0]);
      T[3][i][0] = _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[0], t2[0]);
      T[4][i][0] = _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[0], t2[0]);
      T[5][i][0] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[1][0], t5[0]);

      T[0][i][1] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[0][1], t4[1]);
      T[1][i][1] = _mm_add_ps(t0[1], t1[1]);
      T[2][i][1] = _mm_sub_ps(t0[1], t1[1]);
      T[3][i][1] = _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[1], t2[1]);
      T[4][i][1] = _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[1], t2[1]);
      T[5][i][1] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[1][1], t5[1]);

      T[0][i][2] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[0][2], t4[2]);
      T[1][i][2] = _mm_add_ps(t0[2], t1[2]);
      T[2][i][2] = _mm_sub_ps(t0[2], t1[2]);
      T[3][i][2] = _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[2], t2[2]);
      T[4][i][2] = _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[2], t2[2]);
      T[5][i][2] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[1][2], t5[2]);

      T[0][i][3] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[0][3], t4[3]);
      T[1][i][3] = _mm_add_ps(t0[3], t1[3]);
      T[2][i][3] = _mm_sub_ps(t0[3], t1[3]);
      T[3][i][3] = _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[3], t2[3]);
      T[4][i][3] = _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[3], t2[3]);
      T[5][i][3] = _mm_fmadd_ps(_mm_set1_ps(4.0f), I[1][3], t5[3]);
    }

    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (j = 0; j < ALPHA; j++) {
      t0[0] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][2][0], T[j][4][0]);
      t1[0] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][1][0], T[j][3][0]);
      t2[0] = _mm_sub_ps(T[j][4][0], T[j][2][0]);
      t3[0] = _mm_sub_ps(T[j][3][0], T[j][1][0]);
      t4[0] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][2][0], T[j][4][0]);
      t5[0] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][3][0], T[j][5][0]);

      t0[1] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][2][1], T[j][4][1]);
      t1[1] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][1][1], T[j][3][1]);
      t2[1] = _mm_sub_ps(T[j][4][1], T[j][2][1]);
      t3[1] = _mm_sub_ps(T[j][3][1], T[j][1][1]);
      t4[1] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][2][1], T[j][4][1]);
      t5[1] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][3][1], T[j][5][1]);

      t0[2] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][2][2], T[j][4][2]);
      t1[2] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][1][2], T[j][3][2]);
      t2[2] = _mm_sub_ps(T[j][4][2], T[j][2][2]);
      t3[2] = _mm_sub_ps(T[j][3][2], T[j][1][2]);
      t4[2] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][2][2], T[j][4][2]);
      t5[2] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][3][2], T[j][5][2]);

      t0[3] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][2][3], T[j][4][3]);
      t1[3] = _mm_fnmadd_ps(_mm_set1_ps(4.0f), T[j][1][3], T[j][3][3]);
      t2[3] = _mm_sub_ps(T[j][4][3], T[j][2][3]);
      t3[3] = _mm_sub_ps(T[j][3][3], T[j][1][3]);
      t4[3] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][2][3], T[j][4][3]);
      t5[3] = _mm_fnmadd_ps(_mm_set1_ps(5.0f), T[j][3][3], T[j][5][3]);

      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][0][0], t4[0]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 4, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][0][1], t4[1]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][0][2], t4[2]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 12, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][0][3], t4[3]));

      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_add_ps(t0[0], t1[0]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 4, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_add_ps(t0[1], t1[1]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_add_ps(t0[2], t1[2]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 12, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_add_ps(t0[3], t1[3]));

      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_sub_ps(t0[0], t1[0]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 4, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_sub_ps(t0[1], t1[1]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_sub_ps(t0[2], t1[2]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 12, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_sub_ps(t0[3], t1[3]));

      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[0], t2[0]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 4, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[1], t2[1]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[2], t2[2]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 12, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(2.0f), t3[3], t2[3]));

      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[0], t2[0]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 4, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[1], t2[1]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[2], t2[2]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 12, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fnmadd_ps(_mm_set1_ps(2.0f), t3[3], t2[3]));

      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][1][0], t5[0]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 4, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][1][1], t5[1]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 8, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][1][2], t5[2]));
      _mm_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 12, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm_fmadd_ps(_mm_set1_ps(4.0f), T[j][1][3], t5[3]));
    }
#endif
  } /* for each input channel */
}

if (handle->desc.N < handle->desc.threads)
  libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);

#ifdef FTIME
//libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
t_input = __rdtsc() - t_start;
#endif
#if 0 // def SEP
if (0 == ltid && 10 == sep_cnt) VTPauseSampling();
#endif

#ifdef FTIME
t_wt = 0;
t_gemm = 0;
t_output = 0;
#endif

for (job = thr_begin; job < thr_end; job++) {
  img  = job / thr_per_img;

  const int n = BIMG*JTILES*ITILES;

#define SPMDM_UNROLL_FACTOR (1)

  SIMDTYPE_FP32 sum[n/TDVLEN*SPMDM_UNROLL_FACTOR*SIMD_PER_LINE];
  SIMDTYPE_FP32 I[ALPHA][SIMD_PER_LINE];
  SIMDTYPE_FP32 T[ALPHA-2][ALPHA][SIMD_PER_LINE];
  SIMDTYPE_FP32 t0[SIMD_PER_LINE], t1[SIMD_PER_LINE], t2[SIMD_PER_LINE], t3[SIMD_PER_LINE];

  int ofm1, ofm2, j;
  int ofm1_begin = (job % thr_per_img)*(K/BOFM/thr_per_img);
  int ofm1_end = ofm1_begin + K/BOFM/thr_per_img;

  for (ofm1 = ofm1_begin; ofm1 < ofm1_end; ++ofm1) {

#ifdef FTIME
    t_start = __rdtsc();
#endif

    for (oj = 0; oj < ALPHA; ++oj) {
      for (oi = 0; oi < ALPHA; ++oi) {
        for (ofm2 = 0; ofm2 < BOFM; ++ofm2) {
          LIBXSMM_PRAGMA_UNROLL_N(n/TDVLEN*SPMDM_UNROLL_FACTOR*SIMD_PER_LINE)
          for (j = 0; j < n/TDVLEN*SPMDM_UNROLL_FACTOR*SIMD_PER_LINE; ++j) {
            sum[j] = _MM_SETZERO_FP32();
          }

          // TODO: block output channels
          const float *V_temp = &LIBXSMM_VLA_ACCESS(7, V, img, 0, 0, oj, oi, 0, 0, C, BIMG, ALPHA, ALPHA, JTILES, ITILES);
          int sparse_row = ((ofm1*ALPHA + oj)*ALPHA + oi)*BOFM + ofm2;
          int k0 = handle->sparse_filter_rowptr[sparse_row];
          int k0_end = handle->sparse_filter_rowptr[sparse_row + 1];
//          int k1 = k0_end;
//          int k1_end = handle->sparse_filter_rowptr[sparse_row + 2];
          for ( ; k0 < k0_end/* - 1 && k1 < k1_end - 1*/; k0 += 1/*, k1 += 2*/) {
            SIMDTYPE_FP32
              v0 = _MM_SET1_FP32(handle->sparse_filter_values[k0])/*,
              v1 = _MM_SET1_FP32(handle->sparse_filter_values[k0 + 1]),
              v2 = _MM_SET1_FP32(handle->sparse_filter_values[k1]),
              v3 = _MM_SET1_FP32(handle->sparse_filter_values[k1 + 1])*/;
            int
              colidx0 = handle->sparse_filter_colidx[k0]/*,
              colidx1 = handle->sparse_filter_colidx[k0 + 1],
              colidx2 = handle->sparse_filter_colidx[k1],
              colidx3 = handle->sparse_filter_colidx[k1 + 1]*/;

//            _mm_prefetch(
//                (const char *)(V_temp + handle->sparse_filter_colidx[k0 + 1]*BIMG*ALPHA*ALPHA*TDVLEN),
//                _MM_HINT_T0);

#ifdef __AVX512F__
            sum[0] = _mm512_fmadd_ps(v0, _mm512_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
//            sum[1] = _mm512_fmadd_ps(v1, _mm512_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN), sum[1]);
//            sum[2] = _mm512_fmadd_ps(v2, _mm512_load_ps(V_temp + colidx2*BIMG*ALPHA*ALPHA*TDVLEN), sum[2]);
//            sum[3] = _mm512_fmadd_ps(v3, _mm512_load_ps(V_temp + colidx3*BIMG*ALPHA*ALPHA*TDVLEN), sum[3]);
#elif defined(__AVX2__)
            sum[0] = _mm256_fmadd_ps(v0, _mm256_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
            sum[1] = _mm256_fmadd_ps(v0, _mm256_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[1]);

            sum[2] = _mm256_fmadd_ps(v1, _mm256_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN), sum[2]);
            sum[3] = _mm256_fmadd_ps(v1, _mm256_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[3]);

            sum[4] = _mm256_fmadd_ps(v2, _mm256_load_ps(V_temp + colidx2*BIMG*ALPHA*ALPHA*TDVLEN), sum[4]);
            sum[5] = _mm256_fmadd_ps(v2, _mm256_load_ps(V_temp + colidx2*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[5]);

            sum[6] = _mm256_fmadd_ps(v3, _mm256_load_ps(V_temp + colidx3*BIMG*ALPHA*ALPHA*TDVLEN), sum[6]);
            sum[7] = _mm256_fmadd_ps(v3, _mm256_load_ps(V_temp + colidx3*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[7]);
#else
            sum[0] = _mm_fmadd_ps(v0, _mm_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
            sum[1] = _mm_fmadd_ps(v0, _mm_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN + 4), sum[1]);
            sum[2] = _mm_fmadd_ps(v0, _mm_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[2]);
            sum[3] = _mm_fmadd_ps(v0, _mm_load_ps(V_temp + colidx0*BIMG*ALPHA*ALPHA*TDVLEN + 12), sum[3]);

            sum[4] = _mm_fmadd_ps(v1, _mm_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN), sum[4]);
            sum[5] = _mm_fmadd_ps(v1, _mm_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN + 4), sum[5]);
            sum[6] = _mm_fmadd_ps(v1, _mm_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[6]);
            sum[7] = _mm_fmadd_ps(v1, _mm_load_ps(V_temp + colidx1*BIMG*ALPHA*ALPHA*TDVLEN + 12), sum[7]);
#endif
          }

//          for ( ; k0 < k0_end; ++k0) {
//            SIMDTYPE_FP32 v = _MM_SET1_FP32(handle->sparse_filter_values[k0]);
//            int colidx = handle->sparse_filter_colidx[k0];
//
//  //            _mm_prefetch(
//  //                (const char *)&LIBXSMM_VLA_ACCESS(7, V, img, handle->sparse_filter_colidx[k+1], 0, oj, oi, 0, j*TDVLEN, C, BIMG, ALPHA, ALPHA, JTILES, ITILES),
//  //                _MM_HINT_T0);
//
//#ifdef __AVX512F__
//            sum[0] = _mm512_fmadd_ps(v, _mm512_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
//#elif defined(__AVX2__)
//            sum[0] = _mm256_fmadd_ps(v, _mm256_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
//            sum[1] = _mm256_fmadd_ps(v, _mm256_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[1]);
//#else
//            sum[0] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
//            sum[1] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 4), sum[1]);
//            sum[2] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[2]);
//            sum[3] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 12), sum[3]);
//#endif
//          }

//          for ( ; k1 < k1_end; ++k1) {
//            SIMDTYPE_FP32 v = _MM_SET1_FP32(handle->sparse_filter_values[k1]);
//            int colidx = handle->sparse_filter_colidx[k1];
//
//  //            _mm_prefetch(
//  //                (const char *)&LIBXSMM_VLA_ACCESS(7, V, img, handle->sparse_filter_colidx[k+1], 0, oj, oi, 0, j*TDVLEN, C, BIMG, ALPHA, ALPHA, JTILES, ITILES),
//  //                _MM_HINT_T0);
//
//#ifdef __AVX512F__
//            sum[2] = _mm512_fmadd_ps(v, _mm512_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN), sum[2]);
//#elif defined(__AVX2__)
//            sum[4] = _mm256_fmadd_ps(v, _mm256_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN), sum[4]);
//            sum[5] = _mm256_fmadd_ps(v, _mm256_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[5]);
//#else
//            sum[0] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN), sum[0]);
//            sum[1] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 4), sum[1]);
//            sum[2] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 8), sum[2]);
//            sum[3] = _mm_fmadd_ps(v, _mm_load_ps(V_temp + colidx*BIMG*ALPHA*ALPHA*TDVLEN + 12), sum[3]);
//#endif
//          }

#ifdef __AVX512F__
  //        _mm512_store_ps(
  //            &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES),
  //            _mm512_add_ps(_mm512_add_ps(sum[0], sum[1]), _mm512_add_ps(sum[2], sum[3])));
          _mm512_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              /*_mm512_add_ps(*/sum[0]/*, sum[1])*/);
//          _mm512_store_ps(
//              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2 + 1, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
//              _mm512_add_ps(sum[2], sum[3]));
#elif defined(__AVX2__)
          _mm256_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm256_add_ps(sum[0], sum[2]));
          _mm256_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm256_add_ps(sum[1], sum[3]));
          _mm256_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2 + 1, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm256_add_ps(sum[4], sum[6]));
          _mm256_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2 + 1, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm256_add_ps(sum[5], sum[7]));
#else
          _mm_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm_add_ps(sum[0], sum[4]));
          _mm_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm_add_ps(sum[1], sum[5]));
          _mm_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm_add_ps(sum[2], sum[6]));
          _mm_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN),
              _mm_add_ps(sum[3], sum[7]));
#endif
        }
      }
    }

#ifdef FTIME
    t_gemm += __rdtsc() - t_start;
    t_start = __rdtsc();
#endif

    for (ofm2 = 0; ofm2 < BOFM; ++ofm2) {
#ifdef __AVX512F__
      LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
      for (i = 0; i < ALPHA; i++) {
        I[0][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[1][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[2][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[3][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[4][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[5][0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        t0[0] = _mm512_add_ps(I[1][0], I[2][0]);
        t1[0] = _mm512_add_ps(I[3][0], I[4][0]);
        t2[0] = _mm512_sub_ps(I[1][0], I[2][0]);
        t3[0] = _mm512_sub_ps(I[3][0], I[4][0]);

        T[0][i][0] = _mm512_add_ps(_mm512_add_ps(t0[0], t1[0]), I[0][0]);
        T[1][i][0] = _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3[0], t2[0]);
        T[2][i][0] = _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1[0], t0[0]);
        T[3][i][0] = _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3[0], t2[0]), I[5][0]);
      }

      LIBXSMM_PRAGMA_UNROLL_N(ALPHA-2)
      for (j = 0; j < ALPHA-2; j++) {
        t0[0] = _mm512_add_ps(T[j][1][0], T[j][2][0]);
        t1[0] = _mm512_add_ps(T[j][3][0], T[j][4][0]);
        t2[0] = _mm512_sub_ps(T[j][1][0], T[j][2][0]);
        t3[0] = _mm512_sub_ps(T[j][3][0], T[j][4][0]);

#if 0 // LIBXSMM_X86_AVX512_CORE == LIBXSMM_STATIC_TARGET_ARCH
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_add_ps(_mm512_add_ps(t0[0], t1[0]), T[j][0][0]));
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3[0], t2[0]));
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1[0], t0[0]));
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3[0], t2[0]), T[j][5][0]));
#else
        _mm512_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_add_ps(_mm512_add_ps(t0[0], t1[0]), T[j][0][0]));
        _mm512_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3[0], t2[0]));
        _mm512_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1[0], t0[0]));
        _mm512_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3[0], t2[0]), T[j][5][0]));
#endif
      }
#elif defined(__AVX2__)
      LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
      for (i = 0; i < ALPHA; i++) {
        I[0][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[0][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[1][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[1][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[2][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[2][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[3][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[3][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[4][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[4][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[5][0] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[5][1] = _mm256_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        t0[0] = _mm256_add_ps(I[1][0], I[2][0]);
        t1[0] = _mm256_add_ps(I[3][0], I[4][0]);
        t2[0] = _mm256_sub_ps(I[1][0], I[2][0]);
        t3[0] = _mm256_sub_ps(I[3][0], I[4][0]);

        t0[1] = _mm256_add_ps(I[1][1], I[2][1]);
        t1[1] = _mm256_add_ps(I[3][1], I[4][1]);
        t2[1] = _mm256_sub_ps(I[1][1], I[2][1]);
        t3[1] = _mm256_sub_ps(I[3][1], I[4][1]);

        T[0][i][0] = _mm256_add_ps(_mm256_add_ps(t0[0], t1[0]), I[0][0]);
        T[1][i][0] = _mm256_fmadd_ps(_mm256_set1_ps(2.f), t3[0], t2[0]);
        T[2][i][0] = _mm256_fmadd_ps(_mm256_set1_ps(4.f), t1[0], t0[0]);
        T[3][i][0] = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(8.f), t3[0], t2[0]), I[5][0]);

        T[0][i][1] = _mm256_add_ps(_mm256_add_ps(t0[1], t1[1]), I[0][1]);
        T[1][i][1] = _mm256_fmadd_ps(_mm256_set1_ps(2.f), t3[1], t2[1]);
        T[2][i][1] = _mm256_fmadd_ps(_mm256_set1_ps(4.f), t1[1], t0[1]);
        T[3][i][1] = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(8.f), t3[1], t2[1]), I[5][1]);
      }

      LIBXSMM_PRAGMA_UNROLL_N(ALPHA-2)
      for (j = 0; j < ALPHA-2; j++) {
        t0[0] = _mm256_add_ps(T[j][1][0], T[j][2][0]);
        t1[0] = _mm256_add_ps(T[j][3][0], T[j][4][0]);
        t2[0] = _mm256_sub_ps(T[j][1][0], T[j][2][0]);
        t3[0] = _mm256_sub_ps(T[j][3][0], T[j][4][0]);

        t0[1] = _mm256_add_ps(T[j][1][1], T[j][2][1]);
        t1[1] = _mm256_add_ps(T[j][3][1], T[j][4][1]);
        t2[1] = _mm256_sub_ps(T[j][1][1], T[j][2][1]);
        t3[1] = _mm256_sub_ps(T[j][3][1], T[j][4][1]);

        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_add_ps(_mm256_add_ps(t0[0], t1[0]), T[j][0][0]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_add_ps(_mm256_add_ps(t0[1], t1[1]), T[j][0][1]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_fmadd_ps(_mm256_set1_ps(2.f), t3[0], t2[0]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_fmadd_ps(_mm256_set1_ps(2.f), t3[1], t2[1]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_fmadd_ps(_mm256_set1_ps(4.f), t1[0], t0[0]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_fmadd_ps(_mm256_set1_ps(4.f), t1[1], t0[1]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(8.f), t3[0], t2[0]), T[j][5][0]));
        _mm256_store_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(8.f), t3[1], t2[1]), T[j][5][1]));
      }
#else
      LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
      for (i = 0; i < ALPHA; i++) {
        I[0][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[0][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[0][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[0][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        I[1][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[1][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[1][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[1][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        I[2][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[2][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[2][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[2][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        I[3][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[3][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[3][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[3][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        I[4][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[4][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[4][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[4][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        I[5][0] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 0, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[5][1] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 4, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[5][2] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 8, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));
        I[5][3] = _mm_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, ofm2, 12, K/BOFM, BIMG, ALPHA, ALPHA, BOFM, TDVLEN));

        t0[0] = _mm_add_ps(I[1][0], I[2][0]);
        t1[0] = _mm_add_ps(I[3][0], I[4][0]);
        t2[0] = _mm_sub_ps(I[1][0], I[2][0]);
        t3[0] = _mm_sub_ps(I[3][0], I[4][0]);

        t0[1] = _mm_add_ps(I[1][1], I[2][1]);
        t1[1] = _mm_add_ps(I[3][1], I[4][1]);
        t2[1] = _mm_sub_ps(I[1][1], I[2][1]);
        t3[1] = _mm_sub_ps(I[3][1], I[4][1]);

        t0[2] = _mm_add_ps(I[1][2], I[2][2]);
        t1[2] = _mm_add_ps(I[3][2], I[4][2]);
        t2[2] = _mm_sub_ps(I[1][2], I[2][2]);
        t3[2] = _mm_sub_ps(I[3][2], I[4][2]);

        t0[3] = _mm_add_ps(I[1][3], I[2][3]);
        t1[3] = _mm_add_ps(I[3][3], I[4][3]);
        t2[3] = _mm_sub_ps(I[1][3], I[2][3]);
        t3[3] = _mm_sub_ps(I[3][3], I[4][3]);

        T[0][i][0] = _mm_add_ps(_mm_add_ps(t0[0], t1[0]), I[0][0]);
        T[1][i][0] = _mm_fmadd_ps(_mm_set1_ps(2.f), t3[0], t2[0]);
        T[2][i][0] = _mm_fmadd_ps(_mm_set1_ps(4.f), t1[0], t0[0]);
        T[3][i][0] = _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[0], t2[0]), I[5][0]);

        T[0][i][1] = _mm_add_ps(_mm_add_ps(t0[1], t1[1]), I[0][1]);
        T[1][i][1] = _mm_fmadd_ps(_mm_set1_ps(2.f), t3[1], t2[1]);
        T[2][i][1] = _mm_fmadd_ps(_mm_set1_ps(4.f), t1[1], t0[1]);
        T[3][i][1] = _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[1], t2[1]), I[5][1]);

        T[0][i][2] = _mm_add_ps(_mm_add_ps(t0[2], t1[2]), I[0][2]);
        T[1][i][2] = _mm_fmadd_ps(_mm_set1_ps(2.f), t3[2], t2[2]);
        T[2][i][2] = _mm_fmadd_ps(_mm_set1_ps(4.f), t1[2], t0[2]);
        T[3][i][2] = _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[2], t2[2]), I[5][2]);

        T[0][i][3] = _mm_add_ps(_mm_add_ps(t0[3], t1[3]), I[0][3]);
        T[1][i][3] = _mm_fmadd_ps(_mm_set1_ps(2.f), t3[3], t2[3]);
        T[2][i][3] = _mm_fmadd_ps(_mm_set1_ps(4.f), t1[3], t0[3]);
        T[3][i][3] = _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[3], t2[3]), I[5][3]);
      }

      LIBXSMM_PRAGMA_UNROLL_N(ALPHA-2)
      for (j = 0; j < ALPHA-2; j++) {
        t0[0] = _mm_add_ps(T[j][1][0], T[j][2][0]);
        t1[0] = _mm_add_ps(T[j][3][0], T[j][4][0]);
        t2[0] = _mm_sub_ps(T[j][1][0], T[j][2][0]);
        t3[0] = _mm_sub_ps(T[j][3][0], T[j][4][0]);

        t0[1] = _mm_add_ps(T[j][1][1], T[j][2][1]);
        t1[1] = _mm_add_ps(T[j][3][1], T[j][4][1]);
        t2[1] = _mm_sub_ps(T[j][1][1], T[j][2][1]);
        t3[1] = _mm_sub_ps(T[j][3][1], T[j][4][1]);

        t0[2] = _mm_add_ps(T[j][1][2], T[j][2][2]);
        t1[2] = _mm_add_ps(T[j][3][2], T[j][4][2]);
        t2[2] = _mm_sub_ps(T[j][1][2], T[j][2][2]);
        t3[2] = _mm_sub_ps(T[j][3][2], T[j][4][2]);

        t0[3] = _mm_add_ps(T[j][1][3], T[j][2][3]);
        t1[3] = _mm_add_ps(T[j][3][3], T[j][4][3]);
        t2[3] = _mm_sub_ps(T[j][1][3], T[j][2][3]);
        t3[3] = _mm_sub_ps(T[j][3][3], T[j][4][3]);

        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_add_ps(t0[0], t1[0]), T[j][0][0]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 4, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_add_ps(t0[1], t1[1]), T[j][0][1]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_add_ps(t0[2], t1[2]), T[j][0][2]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 0, 12, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_add_ps(t0[3], t1[3]), T[j][0][3]));

        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(2.f), t3[0], t2[0]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 4, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(2.f), t3[1], t2[1]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(2.f), t3[2], t2[2]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 1, 12, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(2.f), t3[3], t2[3]));

        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(4.f), t1[0], t0[0]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 4, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(4.f), t1[1], t0[1]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(4.f), t1[2], t0[2]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 2, 12, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_fmadd_ps(_mm_set1_ps(4.f), t1[3], t0[3]));

        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[0], t2[0]), T[j][5][0]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 4, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[1], t2[1]), T[j][5][1]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 8, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[2], t2[2]), T[j][5][2]));
        _mm_stream_ps(
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*BOFM + ofm2, j, 3, 12, K, ALPHA-2, ALPHA-2, TDVLEN),
                _mm_add_ps(_mm_fmadd_ps(_mm_set1_ps(8.f), t3[3], t2[3]), T[j][5][3]));
      }
#endif
    }

#ifdef FTIME
    t_output += __rdtsc() - t_start;
#endif
  }
}
#if defined(SEP)
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
static int sep_cnt = 0;
if (0 == ltid) {
  ++sep_cnt;
  if (10 == sep_cnt) VTResumeSampling();
}
#endif

#ifdef FTIME
if (tid == 0) {
  int nOfm = handle->blocksofm*TDVLEN;
  int nIfm = handle->blocksifm*TDVLEN;
  double b_input = 1.0*handle->desc.N*nIfm*(handle->ifhp*handle->ifwp + handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles*ALPHA*ALPHA) * sizeof(float);
  double b_wt    = 1.0*nOfm*nIfm*(handle->desc.R*handle->desc.S + ALPHA*ALPHA) * sizeof(float);
  double b_output= 1.0*handle->desc.N*nOfm*(handle->ofhp*handle->ofwp + handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles*ALPHA*ALPHA) * sizeof(float);
  double f_gemm = 2.0*handle->desc.N*nOfm*nIfm*handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles*ALPHA*ALPHA;
  printf("Time: i=%8.3f  w=%8.3f  o=%8.3f         g=%8.3f\n", t_input/1000.0, t_wt/1000.0, t_output/1000.0, t_gemm/1000.0);
  printf("BW:   i=%8.3f  w=%8.3f  o=%8.3f (b/c)   g=%8.3f (f/c)\n\n", b_input/t_input, b_wt/t_wt, b_output/t_output, f_gemm/t_gemm);
}
#endif
#undef FTIME

#undef SIMDTYPE_FP32
#undef _MM_SETZERO_FP32
#undef _MM_SET1_FP32
#undef SIMD_PER_LINE

#undef _mm_fmadd_ps
#undef _mm_fnmadd_ps

#undef SPMDM_UNROLL_FACTOR
#undef BOFM
