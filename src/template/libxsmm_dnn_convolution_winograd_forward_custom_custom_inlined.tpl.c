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

//LIBXSMM_VLA_DECL(5, const float, input,  (const float*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, TDVLEN);
LIBXSMM_VLA_DECL(5, const float, input,  (const float*)handle->reg_input->data, handle->blocksifm*TDVLEN, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, (float*)handle->reg_output->data, handle->blocksofm*TDVLEN, ALPHA-2, ALPHA-2, TDVLEN);
LIBXSMM_VLA_DECL(6, float, weight, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN);
/*LIBXSMM_VLA_DECL(2, float, bias, handle->bias->data, TDVLEN);*/

// The main part of Winograd convolution is
// ALPHA x ALPHA independent GEMMs
// Each GEMM is K x C x (bimg x jtiles x itiles)

// In Kunal's implementation,
// Each GEMM is implemented with
// (K/TDVLEN) x (C/TDVLEN) small GEMMs with each size
// TDVLEN x TDVLEN x (bimg x jtiles x itiles)

// U is ALPHA x ALPHA x K x C instead of ALPHA x ALPHA x K/TDVLEN x C/TDVLEN x TDVLEN x TDVLEN
LIBXSMM_VLA_DECL(4, float, U,   (float*)handle->scratch1, ALPHA, handle->blocksofm*TDVLEN, handle->blocksifm*TDVLEN);
// V is N/bimg x ALPHA x ALPHA x C x bimg x jtiles x itiles instead of N/bimg x ALPHA x ALPHA x C/TDVLEN x bimg x jtiles x itiles x TDVLEN
LIBXSMM_VLA_DECL(7, float, V,   (float*)handle->scratch3, ALPHA, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles);
// M is N/bimg x ALPHA x ALPHA x K x bimg x jtiles x itiles instead of N/bimg x ALPHA x ALPHA x K/TDVLEN x bimg x jtiles x itiles x TDVLEN
LIBXSMM_VLA_DECL(7, float, M,   (float*)handle->scratch4, ALPHA, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles);
LIBXSMM_VLA_DECL(5, float, Iwp, (float*)handle->scratchIw, handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Owp, (float*)handle->scratchOw, handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles, ALPHA, ALPHA, TDVLEN);
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
libxsmm_barrier_init((libxsmm_barrier*)handle->barrier, ltid);

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
work = handle->desc.N*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

// TODO: change input and output format so that we can vectorize over tiles

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / handle->blocksifm;
  ifm1 = job % handle->blocksifm;

  const float *inp = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1*TDVLEN, 0, 0, 0, handle->blocksifm*TDVLEN, ALPHA, ALPHA, TDVLEN);
  float *tinp = &LIBXSMM_VLA_ACCESS(7, V, img/handle->cwino_fwd.bimg, 0, 0, ifm1*TDVLEN, img%handle->cwino_fwd.bimg, 0, 0, ALPHA, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles);
  float *Iwp2 = &LIBXSMM_VLA_ACCESS(5, Iwp, tid, 0, 0, 0, 0, handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles, ALPHA, ALPHA, TDVLEN);

//  internal_fwd_input_transform_custom_custom(
//    &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, TDVLEN),
//    &LIBXSMM_VLA_ACCESS(8, V, img/handle->cwino_fwd.bimg, 0, 0, ifm1, img%handle->cwino_fwd.bimg, 0, 0, 0, ALPHA, ALPHA, handle->blocksifm, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN),
//    &LIBXSMM_VLA_ACCESS(5, Iwp, tid, 0, 0, 0, 0, handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles, ALPHA, ALPHA, TDVLEN), handle);

  const int total_tiles = handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles;
  LIBXSMM_VLA_DECL(4, const float, input, inp, ALPHA, ALPHA, TDVLEN);
  LIBXSMM_VLA_DECL(5, float, output, tinp, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles);
  LIBXSMM_VLA_DECL(4, float, Iw, Iwp2, ALPHA, ALPHA, TDVLEN);
  unsigned int ti, tj;
  int i, j, k;
  int xdim, ydim;
  float T[6][6][TDVLEN];
  float t0[TDVLEN];
  float t1[TDVLEN];
  float t2[TDVLEN];
  float t3[TDVLEN];
  float t4[TDVLEN];
  float t5[TDVLEN];
  float I[ALPHA][TDVLEN];

  // assume total tiles = TDVLEN = 16

  int ifm2;
  for (ifm2 = 0; ifm2 < TDVLEN; ifm2++) {
    for (i = 0; i < 6; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        I[0][k] = LIBXSMM_VLA_ACCESS(4, input, ifm2, 0, i, k, ALPHA, ALPHA, TDVLEN);
        I[1][k] = LIBXSMM_VLA_ACCESS(4, input, ifm2, 1, i, k, ALPHA, ALPHA, TDVLEN);
        I[2][k] = LIBXSMM_VLA_ACCESS(4, input, ifm2, 2, i, k, ALPHA, ALPHA, TDVLEN);
        I[3][k] = LIBXSMM_VLA_ACCESS(4, input, ifm2, 3, i, k, ALPHA, ALPHA, TDVLEN);
        I[4][k] = LIBXSMM_VLA_ACCESS(4, input, ifm2, 4, i, k, ALPHA, ALPHA, TDVLEN);
        I[5][k] = LIBXSMM_VLA_ACCESS(4, input, ifm2, 5, i, k, ALPHA, ALPHA, TDVLEN);

        t0[k] = I[4][k] - 4.0f*I[2][k];
        t1[k] = I[3][k] - 4.0f*I[1][k];
        t2[k] = I[4][k] - I[2][k];
        t3[k] = I[3][k] - I[1][k];
        t4[k] = I[4][k] - 5.0f*I[2][k];
        t5[k] = I[5][k] - 5.0f*I[3][k];
        T[0][i][k] = t4[k] + 4.0f*I[0][k];
        T[1][i][k] = t0[k] + t1[k];
        T[2][i][k] = t0[k] - t1[k];
        T[3][i][k] = t2[k] + 2.0f*t3[k];
        T[4][i][k] = t2[k] - 2.0f*t3[k];
        T[5][i][k] = t5[k] + 4.0f*I[1][k];
      }
    }

    for (j = 0; j < 6; j++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        t0[k] = T[j][4][k] - 4.0f*T[j][2][k];
        t1[k] = T[j][3][k] - 4.0f*T[j][1][k];
        t2[k] = T[j][4][k] - T[j][2][k];
        t3[k] = T[j][3][k] - T[j][1][k];
        t4[k] = T[j][4][k] - 5.0f*T[j][2][k];
        t5[k] = T[j][5][k] - 5.0f*T[j][3][k];

        LIBXSMM_VLA_ACCESS(5, output, j, 0, ifm2, 0, k, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles) = t4[k] + 4.0f*T[j][0][k];
        LIBXSMM_VLA_ACCESS(5, output, j, 1, ifm2, 0, k, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles) = t0[k] + t1[k];
        LIBXSMM_VLA_ACCESS(5, output, j, 2, ifm2, 0, k, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles) = t0[k] - t1[k];
        LIBXSMM_VLA_ACCESS(5, output, j, 3, ifm2, 0, k, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles) = t2[k] + 2.0f*t3[k];
        LIBXSMM_VLA_ACCESS(5, output, j, 4, ifm2, 0, k, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles) = t2[k] - 2.0f*t3[k];
        LIBXSMM_VLA_ACCESS(5, output, j, 5, ifm2, 0, k, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, total_tiles) = t5[k] + 4.0f*T[j][1][k];
      }
    }
  } /* for each input channel */
}
#ifdef FTIME
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
t_input = __rdtsc() - t_start;
#endif
#if 0 // def SEP
if (0 == ltid && 10 == sep_cnt) VTPauseSampling();
#endif

#if 1
/* number of tasks that could be run in parallel */
work = handle->blocksofm*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  ofm1 = job / handle->blocksifm;
  ifm1 = job % handle->blocksifm;

  float *wp = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN);
  float *twp = &LIBXSMM_VLA_ACCESS(4, U, 0, 0, ofm1*TDVLEN, ifm1*TDVLEN, ALPHA, handle->blocksofm*TDVLEN, handle->blocksifm*TDVLEN);

  LIBXSMM_VLA_DECL(6, float, input, wp, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
  LIBXSMM_VLA_DECL(4, float, output, twp, ALPHA, handle->blocksofm*TDVLEN, handle->blocksifm*TDVLEN);
  float Fw[ALPHA][ALPHA][TDVLEN][TDVLEN];
  float F[3][3][TDVLEN][TDVLEN];
  unsigned int i, j;
  int k, l, v;
  const float rcp4  = 1.0f/4.0f;
  const float rcp6  = 1.0f/6.0f;
  const float rcp12 = 1.0f/12.0f;
  const float rcp24 = 1.0f/24.0f;
  float T[6][3][TDVLEN];
  float Fw_[6][TDVLEN];
  float t0[TDVLEN];
  float t1[TDVLEN];
  float t2[TDVLEN];

  for (j = 0; j < 3; j++) {
    for (i = 0; i < 3; i++) {
      for (k = 0; k < TDVLEN; k++) {
        LIBXSMM_PRAGMA_SIMD
        for (l = 0; l < TDVLEN; l++) {
          F[j][i][k][l] =
            LIBXSMM_VLA_ACCESS(6, input, 0, 0, j, i, k, l, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
        }
      }
    }
  }
  /*trans_F_4x4_3x3(TDVLEN, Fw, F);*/

  /* inline code start */
  for (j = 0; j < TDVLEN; j++) {
    for (i = 0; i < 3; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        t0[k] = rcp6 * F[2][i][j][k];
        t1[k] = -t0[k] - rcp6*F[0][i][j][k];
        t2[k] = t0[k] + rcp24*F[0][i][j][k];
        T[0][i][k] = rcp4 * F[0][i][j][k];
        T[1][i][k] = t1[k] - rcp6*F[1][i][j][k];
        T[2][i][k] = t1[k] + rcp6*F[1][i][j][k];
        T[3][i][k] = t2[k] + rcp12*F[1][i][j][k];
        T[4][i][k] = t2[k] - rcp12*F[1][i][j][k];
        T[5][i][k] = F[2][i][j][k];
      }
    }
    for (i = 0; i < 6; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        t0[k] = rcp6 * T[i][2][k];
        t1[k] = -t0[k] - rcp6*T[i][0][k];
        t2[k] = t0[k] + rcp24*T[i][0][k];
        Fw_[0][k] = rcp4 * T[i][0][k];
        Fw_[1][k] = t1[k] - rcp6*T[i][1][k];
        Fw_[2][k] = t1[k] + rcp6*T[i][1][k];
        Fw_[3][k] = t2[k] + rcp12*T[i][1][k];
        Fw_[4][k] = t2[k] - rcp12*T[i][1][k];
        Fw_[5][k] = T[i][2][k];

        for (l = 0; l < 6; l++) {
          Fw[i][l][j][k] = Fw_[l][k];
        }
      }
    }
  }
  /* inline code end */

  for (j = 0; j < ALPHA; j++) {
    for (i = 0; i < ALPHA; i++) {
      for (v = 0; v < TDVLEN; v++) {
        LIBXSMM_PRAGMA_SIMD
        for (k = 0; k < TDVLEN; k++) {
          LIBXSMM_VLA_ACCESS(4, output, j, i, k, v, ALPHA, handle->blocksofm*TDVLEN, handle->blocksifm*TDVLEN) =
            Fw[j][i][v][k];
        }
      }
    }
  }
  /* output has shape ALPHA*ALPHA*(blocksifm*blocksofm)*TDVLEN*TDVLEN */

//  internal_fwd_weight_transform(
//    &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN),
//    &LIBXSMM_VLA_ACCESS(6, U, 0, 0, ofm1, ifm1, 0, 0, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN), handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef FTIME
t_wt = __rdtsc() - t_start;
#endif
#else
#ifdef FTIME
t_wt = 0;
#endif
#endif

/* number of tasks that could be run in parallel */
work = (handle->desc.N/handle->cwino_fwd.bimg) * ALPHA * ALPHA;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  oj = job / ALPHA / (handle->desc.N/handle->cwino_fwd.bimg);
  oi = job / (handle->desc.N/handle->cwino_fwd.bimg) % ALPHA;
  img = job % (handle->desc.N/handle->cwino_fwd.bimg);

  int m = handle->blocksofm*TDVLEN;
  int n = handle->cwino_fwd.bimg*handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles;
  int k = handle->blocksifm*TDVLEN;

//  float alpha = 1, beta = 1;
//
//  libxsmm_sgemm("N", "N", &n, &m, &k,
//      &alpha, &LIBXSMM_VLA_ACCESS(7, V, img, oj, oi, 0, 0, 0, 0, ALPHA, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles), &n,
//      &LIBXSMM_VLA_ACCESS(4, U, oj, oi, 0, 0, ALPHA, handle->blocksofm*TDVLEN, handle->blocksifm*TDVLEN), &k,
//      &beta, &LIBXSMM_VLA_ACCESS(7, M, img, oj, oi, 0, 0, 0, 0, ALPHA, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles), &n);

  int i, j;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      float sum = 0;
      for (k = 0; k < handle->blocksifm*TDVLEN; ++k) {
        sum += LIBXSMM_VLA_ACCESS(4, U, oj, oi, i, k, ALPHA, m, handle->blocksifm*TDVLEN)
             * LIBXSMM_VLA_ACCESS(7, V, img, oj, oi, k, 0, 0, j, ALPHA, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles);
      }
      LIBXSMM_VLA_ACCESS(7, M, img, oj, oi, i, 0, 0, j, ALPHA, ALPHA, m, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles) = sum;
    }
  }

//  for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
//    for (i = 0; i < handle->cwino_fwd.bimg; i++) {
//      for (j = 0; j < handle->cwino_fwd.jtiles; j++) {
//        for (k = 0; k < handle->cwino_fwd.itiles; k++) {
//          LIBXSMM_PRAGMA_SIMD
//          for (l = 0; l < TDVLEN; l++) {
//            LIBXSMM_VLA_ACCESS(7, M, img, oj, oi, ofm1*TDVLEN + l, i, j, k, ALPHA, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles) = 0.0f;
//          }
//        }
//      }
//    }
//    int ofm2, ifm;
//    for (ofm2 = 0; ofm2 < TDVLEN; ofm2++) {
//      for (ifm = 0; ifm < handle->blocksifm*TDVLEN; ifm++) {
//        unsigned int ti, tj;
//        unsigned int img1;
//        for (img1 = 0; img1 < handle->cwino_fwd.bimg; img1++) {
//          for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
//            for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
//              LIBXSMM_VLA_ACCESS  (7, M, img, oj, oi, ofm1*TDVLEN + ofm2, img1, tj, ti, ALPHA, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles) +=
//                LIBXSMM_VLA_ACCESS(4, U, oj, oi, ofm1*TDVLEN + ofm2, ifm, ALPHA, handle->blocksofm*TDVLEN, handle->blocksifm*TDVLEN)
//              * LIBXSMM_VLA_ACCESS(7, V, img, oj, oi, ifm, img1, tj, ti, ALPHA, ALPHA, handle->blocksifm*TDVLEN, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles);
//            }
//          }
//        }
//      }
//    }
//  }
}
#if defined(SEP)
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
static int sep_cnt = 0;
if (0 == ltid) {
  ++sep_cnt;
  if (10 == sep_cnt) VTResumeSampling();
}
#endif
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef FTIME
t_gemm = __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksofm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / handle->blocksofm;
  ofm1 = job % handle->blocksofm;
//  internal_fwd_output_transform_custom_custom(
//    &LIBXSMM_VLA_ACCESS(8, M, img/handle->cwino_fwd.bimg, 0, 0, ofm1, img%handle->cwino_fwd.bimg, 0, 0, 0, ALPHA, ALPHA, handle->blocksofm, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN),
//    &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, TDVLEN),
//    &LIBXSMM_VLA_ACCESS(5, Owp, tid, 0, 0, 0, 0, handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles, ALPHA, ALPHA, TDVLEN), /*TDVLEN,*/ 0 /*&bias[ofm1]*/, handle);

  const float *toutp = &LIBXSMM_VLA_ACCESS(7, M, img/handle->cwino_fwd.bimg, 0, 0, ofm1*TDVLEN, img%handle->cwino_fwd.bimg, 0, 0, ALPHA, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles);
  float *outp = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1*TDVLEN, 0, 0, 0, handle->blocksofm*TDVLEN, ALPHA-2, ALPHA-2, TDVLEN);
  float *Owp2 = &LIBXSMM_VLA_ACCESS(5, Owp, tid, 0, 0, 0, 0, handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles, ALPHA, ALPHA, TDVLEN);

  const int total_tiles = handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles;
  LIBXSMM_VLA_DECL(5, const float, input, toutp, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, total_tiles);
  LIBXSMM_VLA_DECL(4, float, out, outp, ALPHA-2, ALPHA-2, TDVLEN);
  LIBXSMM_VLA_DECL(4, float, Ow, Owp2, ALPHA, ALPHA, TDVLEN);
  float O[ALPHA-2][ALPHA-2][TDVLEN];
  unsigned int ti, tj;
  int i, j, k;
  int ofm2;
  int xdim, ydim;
  float T[4][6][TDVLEN];
  float t0[TDVLEN];
  float t1[TDVLEN];
  float t2[TDVLEN];
  float t3[TDVLEN];

  for (ofm2 = 0; ofm2 < TDVLEN; ++ofm2) {
    for (i = 0; i < 6; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        t0[k] = LIBXSMM_VLA_ACCESS(5, input, 1, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN) + LIBXSMM_VLA_ACCESS(5, input, 2, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN);
        t1[k] = LIBXSMM_VLA_ACCESS(5, input, 3, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN) + LIBXSMM_VLA_ACCESS(5, input, 4, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN);
        t2[k] = LIBXSMM_VLA_ACCESS(5, input, 1, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN) - LIBXSMM_VLA_ACCESS(5, input, 2, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN);
        t3[k] = LIBXSMM_VLA_ACCESS(5, input, 3, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN) - LIBXSMM_VLA_ACCESS(5, input, 4, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN);

        T[0][i][k] = t0[k] + t1[k]     + LIBXSMM_VLA_ACCESS(5, input, 0, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN);
        T[1][i][k] = t2[k] + t3[k]*2.f;
        T[2][i][k] = t0[k] + t1[k]*4.f;
        T[3][i][k] = t2[k] + t3[k]*8.f + LIBXSMM_VLA_ACCESS(5, input, 5, i, ofm2, 0, k, ALPHA, handle->blocksofm*TDVLEN, handle->cwino_fwd.bimg, TDVLEN);
      }
    }

    for (j = 0; j < 4; j++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        t0[k] = T[j][1][k] + T[j][2][k];
        t1[k] = T[j][3][k] + T[j][4][k];
        t2[k] = T[j][1][k] - T[j][2][k];
        t3[k] = T[j][3][k] - T[j][4][k];

        LIBXSMM_VLA_ACCESS(4, out, ofm2, j, 0, k, ALPHA-2, ALPHA-2, TDVLEN) = t0[k] + t1[k]     + T[j][0][k];
        LIBXSMM_VLA_ACCESS(4, out, ofm2, j, 1, k, ALPHA-2, ALPHA-2, TDVLEN) = t2[k] + t3[k]*2.f;
        LIBXSMM_VLA_ACCESS(4, out, ofm2, j, 2, k, ALPHA-2, ALPHA-2, TDVLEN) = t0[k] + t1[k]*4.f;
        LIBXSMM_VLA_ACCESS(4, out, ofm2, j, 3, k, ALPHA-2, ALPHA-2, TDVLEN) = t2[k] + t3[k]*8.f + T[j][5][k];
      }
    }
  }
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef FTIME
t_output = __rdtsc() - t_start;
#endif
#ifdef SEP
if (0 == ltid && 10 == sep_cnt) VTPauseSampling();
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
