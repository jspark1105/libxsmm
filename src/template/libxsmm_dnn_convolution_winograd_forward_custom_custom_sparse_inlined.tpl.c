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

const int K = 384, C = 256;

LIBXSMM_VLA_DECL(5, const float, input,  (const float*)handle->reg_input->data, C, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, (float*)handle->reg_output->data, K, ALPHA-2, ALPHA-2, TDVLEN);

// The main part of Winograd convolution is
// ALPHA x ALPHA independent GEMMs
// Each GEMM is K x C x (bimg x jtiles x itiles)

// In Kunal's implementation,
// Each GEMM is implemented with
// (K/TDVLEN) x (C/TDVLEN) small GEMMs with each size
// TDVLEN x TDVLEN x (bimg x jtiles x itiles)

// V is N/bimg x ALPHA x ALPHA x C x bimg x jtiles x itiles instead of N/bimg x ALPHA x ALPHA x C/TDVLEN x bimg x jtiles x itiles x TDVLEN
LIBXSMM_VLA_DECL(7, float, V,   (float*)handle->scratch3, C, BIMG, ALPHA, ALPHA, JTILES, ITILES);
// M is N/bimg x ALPHA x ALPHA x K x bimg x jtiles x itiles instead of N/bimg x ALPHA x ALPHA x K/TDVLEN x bimg x jtiles x itiles x TDVLEN
LIBXSMM_VLA_DECL(7, float, M,   (float*)handle->scratch4, K, BIMG, ALPHA, ALPHA, JTILES, ITILES);
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

  __m512 T[ALPHA][ALPHA];
  __m512 t0, t1, t2, t3, t4, t5;
  __m512 I[ALPHA];

  int ifm2;
  for (ifm2 = 0; ifm2 < handle->blocksifm*TDVLEN/thr_per_img; ifm2++) {
    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (i = 0; i < ALPHA; i++) {
      I[0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, 0, ALPHA, ALPHA, TDVLEN));
      I[1] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, 0, ALPHA, ALPHA, TDVLEN));
      I[2] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, 0, ALPHA, ALPHA, TDVLEN));
      I[3] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, 0, ALPHA, ALPHA, TDVLEN));
      I[4] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, 0, ALPHA, ALPHA, TDVLEN));
      I[5] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, 0, ALPHA, ALPHA, TDVLEN));

      t0 = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), I[2], I[4]);
      t1 = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), I[1], I[3]);
      t2 = _mm512_sub_ps(I[4], I[2]);
      t3 = _mm512_sub_ps(I[3], I[1]);
      t4 = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), I[2], I[4]);
      t5 = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), I[3], I[5]);

      T[0][i] = _mm512_fmadd_ps(_mm512_set1_ps(4.0f), I[0], t4);
      T[1][i] = _mm512_add_ps(t0, t1);
      T[2][i] = _mm512_sub_ps(t0, t1);
      T[3][i] = _mm512_fmadd_ps(_mm512_set1_ps(2.0f), t3, t2);
      T[4][i] = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), t3, t2);
      T[5][i] = _mm512_fmadd_ps(_mm512_set1_ps(4.0f), I[1], t5);
    }

    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (j = 0; j < ALPHA; j++) {
      t0 = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), T[j][2], T[j][4]);
      t1 = _mm512_fnmadd_ps(_mm512_set1_ps(4.0f), T[j][1], T[j][3]);
      t2 = _mm512_sub_ps(T[j][4], T[j][2]);
      t3 = _mm512_sub_ps(T[j][3], T[j][1]);
      t4 = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), T[j][2], T[j][4]);
      t5 = _mm512_fnmadd_ps(_mm512_set1_ps(5.0f), T[j][3], T[j][5]);

      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 0, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fmadd_ps(_mm512_set1_ps(4.0f), T[j][0], t4));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 1, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_add_ps(t0, t1));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 2, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_sub_ps(t0, t1));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 3, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fmadd_ps(_mm512_set1_ps(2.0f), t3, t2));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 4, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), t3, t2));
      _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(5, output, ifm2, 0, j, 5, 0, BIMG, ALPHA, ALPHA, TDVLEN),
          _mm512_fmadd_ps(_mm512_set1_ps(4.0f), T[j][1], t5));
    }
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

  __m512 sum[n/TDVLEN];
  __m512 I[ALPHA];
  __m512 T[ALPHA-2][ALPHA];
  __m512 t0, t1, t2, t3;

  int ofm, j;
  int ofm_begin = (job % thr_per_img)*(handle->blocksofm/thr_per_img*TDVLEN);
  int ofm_end = ofm_begin + handle->blocksofm/thr_per_img*TDVLEN;

  for (ofm = ofm_begin; ofm < ofm_end; ++ofm) {

#ifdef FTIME
    t_start = __rdtsc();
#endif

    for (oj = 0; oj < ALPHA; ++oj) {
      for (oi = 0; oi < ALPHA; ++oi) {
        LIBXSMM_PRAGMA_UNROLL_N(n/TDVLEN)
        for (j = 0; j < n/TDVLEN; ++j) {
          sum[j] = _mm512_setzero_ps();
        }

        int sparse_row = (ofm*ALPHA + oj)*ALPHA + oi;
        LIBXSMM_PRAGMA_UNROLL_N(2)
        for (k = handle->sparse_filter_rowptr[sparse_row]; k < handle->sparse_filter_rowptr[sparse_row + 1]; ++k) {
          __m512 v = _mm512_set1_ps(handle->sparse_filter_values[k]);
          int colidx = handle->sparse_filter_colidx[k];

//          LIBXSMM_PRAGMA_UNROLL_N(n/TDVLEN)
          for (j = 0; j < n/TDVLEN; ++j) {
//            _mm_prefetch(
//                (const char *)&LIBXSMM_VLA_ACCESS(7, V, img, handle->sparse_filter_colidx[k+1], 0, oj, oi, 0, j*TDVLEN, C, BIMG, ALPHA, ALPHA, JTILES, ITILES),
//                _MM_HINT_T0);

            sum[j] = _mm512_fmadd_ps(
                v,
                _mm512_load_ps(
                    &LIBXSMM_VLA_ACCESS(7, V, img, colidx, 0, oj, oi, 0, j*TDVLEN, C, BIMG, ALPHA, ALPHA, JTILES, ITILES)),
                sum[j]);
          }
        }

        LIBXSMM_PRAGMA_UNROLL_N(n/TDVLEN)
        for (j = 0; j < n/TDVLEN; ++j) {
          _mm512_store_ps(
              &LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, oj, oi, 0, j*TDVLEN, K, BIMG, ALPHA, ALPHA, JTILES, ITILES),
              sum[j]);
        }
      }
    }

#ifdef FTIME
    t_gemm += __rdtsc() - t_start;
    t_start = __rdtsc();
#endif

    LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
    for (i = 0; i < ALPHA; i++) {
      I[0] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 0, i, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES));
      I[1] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 1, i, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES));
      I[2] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 2, i, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES));
      I[3] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 3, i, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES));
      I[4] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 4, i, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES));
      I[5] = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(7, M, 0, ltid, 0, 5, i, 0, 0, K, BIMG, ALPHA, ALPHA, JTILES, ITILES));

      t0 = _mm512_add_ps(I[1], I[2]);
      t1 = _mm512_add_ps(I[3], I[4]);
      t2 = _mm512_sub_ps(I[1], I[2]);
      t3 = _mm512_sub_ps(I[3], I[4]);

      T[0][i] = _mm512_add_ps(_mm512_add_ps(t0, t1), I[0]);
      T[1][i] = _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3, t2);
      T[2][i] = _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1, t0);
      T[3][i] = _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3, t2), I[5]);
    }

    LIBXSMM_PRAGMA_UNROLL_N(ALPHA-2)
    for (j = 0; j < ALPHA-2; j++) {
      t0 = _mm512_add_ps(T[j][1], T[j][2]);
      t1 = _mm512_add_ps(T[j][3], T[j][4]);
      t2 = _mm512_sub_ps(T[j][1], T[j][2]);
      t3 = _mm512_sub_ps(T[j][3], T[j][4]);

      _mm512_stream_ps(
          &LIBXSMM_VLA_ACCESS(5, output, img, ofm, j, 0, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
              _mm512_add_ps(_mm512_add_ps(t0, t1), T[j][0]));
      _mm512_stream_ps(
          &LIBXSMM_VLA_ACCESS(5, output, img, ofm, j, 1, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
              _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3, t2));
      _mm512_stream_ps(
          &LIBXSMM_VLA_ACCESS(5, output, img, ofm, j, 2, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
              _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1, t0));
      _mm512_stream_ps(
          &LIBXSMM_VLA_ACCESS(5, output, img, ofm, j, 3, 0, K, ALPHA-2, ALPHA-2, TDVLEN),
              _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3, t2), T[j][5]));
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
