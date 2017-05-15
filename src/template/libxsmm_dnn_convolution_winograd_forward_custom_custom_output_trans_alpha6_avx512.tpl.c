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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

const int total_tiles = handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles;
LIBXSMM_VLA_DECL(5, const float, input, toutp, ALPHA, handle->blocksofm*handle->cwino_fwd.bimg, total_tiles, TDVLEN);
LIBXSMM_VLA_DECL(4, float, output, outp, handle->ofhp, handle->ofwp, TDVLEN);
LIBXSMM_VLA_DECL(4, float, Ow, Owp, ALPHA, ALPHA, TDVLEN);
__m512 O[ALPHA-2];
unsigned int ti, tj;
int i, j;
int xdim, ydim;
__m512 T[4][6]; // FIXME: too big and causing spills
__m512 t0, t1, t2, t3;
__m512 I0, I1, I2, I3, I4, I5;

/* We have this separate loop to avoid long stride accesses */
/* If we directly read from input in the loop below, we'd have blocksofm*bimg*total_tiles*64 Byte strides */
/* In this loop, the stride is reduced to 64 */
/* input has shape ALPHA*ALPHA*(blocksofm*cwino_fwd.bimg)*total_tiles*TDVLEN */
/* typically, blocksofm = K/VLEN ofmblock = VLEN */
/* the order of this loop is optimized for the source array which bigger */
for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
      for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
        _mm512_store_ps(
          &LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, j, i, 0, ALPHA, ALPHA, TDVLEN),
          _mm512_load_ps(
            &LIBXSMM_VLA_ACCESS(5, input, j, i, 0, tj*handle->cwino_fwd.itiles + ti, 0, ALPHA, handle->blocksofm*handle->cwino_fwd.bimg, total_tiles, TDVLEN)));
      }
    }
  }
}

for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
  for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
    /*trans_O_4x4_3x3(ALPHA-2, TDVLEN, Ow[tj*handle->cwino_fwd.itiles + ti], O);*/
    /* inline code start */
    if ((tj+1)*(ALPHA-2) <= handle->ofh && (ti+1)*(ALPHA-2) <= handle->ofw) { /* common case */
      LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
      for (i = 0; i < ALPHA; i++) {
        I0 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 0, i, 0, ALPHA, ALPHA, TDVLEN));
        I1 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 1, i, 0, ALPHA, ALPHA, TDVLEN));
        I2 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 2, i, 0, ALPHA, ALPHA, TDVLEN));
        I3 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 3, i, 0, ALPHA, ALPHA, TDVLEN));
        I4 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 4, i, 0, ALPHA, ALPHA, TDVLEN));
        I5 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 5, i, 0, ALPHA, ALPHA, TDVLEN));

        t0 = _mm512_add_ps(I1, I2);
        t1 = _mm512_add_ps(I3, I4);
        t2 = _mm512_sub_ps(I1, I2);
        t3 = _mm512_sub_ps(I3, I4);

        T[0][i] = _mm512_add_ps(_mm512_add_ps(t0, t1), I0);
        T[1][i] = _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3, t2);
        T[2][i] = _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1, t0);
        T[3][i] = _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3, t2), I5);
      }

      LIBXSMM_PRAGMA_UNROLL_N(ALPHA-2)
      for (j = 0; j < ALPHA-2; j++) {
        ydim = tj*(ALPHA - 2) + j;

        t0 = _mm512_add_ps(T[j][1], T[j][2]);
        t1 = _mm512_add_ps(T[j][3], T[j][4]);
        t2 = _mm512_sub_ps(T[j][1], T[j][2]);
        t3 = _mm512_sub_ps(T[j][3], T[j][4]);

        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2), 0, handle->ofhp, handle->ofwp, TDVLEN),
            _mm512_add_ps(
                _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2), 0, handle->ofhp, handle->ofwp, TDVLEN)),
                _mm512_add_ps(_mm512_add_ps(t0, t1), T[j][0])));
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2) + 1, 0, handle->ofhp, handle->ofwp, TDVLEN),
            _mm512_add_ps(
                _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2) + 1, 0, handle->ofhp, handle->ofwp, TDVLEN)),
                _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3, t2)));
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2) + 2, 0, handle->ofhp, handle->ofwp, TDVLEN),
            _mm512_add_ps(
                _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2) + 2, 0, handle->ofhp, handle->ofwp, TDVLEN)),
                _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1, t0)));
        _mm512_store_ps(
            &LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2) + 3, 0, handle->ofhp, handle->ofwp, TDVLEN),
            _mm512_add_ps(
                _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, output, 0, ydim, ti*(ALPHA-2) + 3, 0, handle->ofhp, handle->ofwp, TDVLEN)),
                _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3, t2), T[j][5])));
      }
    }
    else { /* corner case */
      LIBXSMM_PRAGMA_UNROLL_N(ALPHA)
      for (i = 0; i < ALPHA; i++) {
        I0 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 0, i, 0, ALPHA, ALPHA, TDVLEN));
        I1 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 1, i, 0, ALPHA, ALPHA, TDVLEN));
        I2 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 2, i, 0, ALPHA, ALPHA, TDVLEN));
        I3 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 3, i, 0, ALPHA, ALPHA, TDVLEN));
        I4 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 4, i, 0, ALPHA, ALPHA, TDVLEN));
        I5 = _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 5, i, 0, ALPHA, ALPHA, TDVLEN));

        t0 = _mm512_add_ps(I1, I2);
        t1 = _mm512_add_ps(I3, I4);
        t2 = _mm512_sub_ps(I1, I2);
        t3 = _mm512_sub_ps(I3, I4);

        T[0][i] = _mm512_add_ps(_mm512_add_ps(t0, t1), I0);
        T[1][i] = _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3, t2);
        T[2][i] = _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1, t0);
        T[3][i] = _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3, t2), I5);
      }

      for (j = 0; j < LIBXSMM_MIN(handle->ofh - (int)tj*(ALPHA-2), ALPHA-2); j++) {
        t0 = _mm512_add_ps(T[j][1], T[j][2]);
        t1 = _mm512_add_ps(T[j][3], T[j][4]);
        t2 = _mm512_sub_ps(T[j][1], T[j][2]);
        t3 = _mm512_sub_ps(T[j][3], T[j][4]);

        O[0] = _mm512_add_ps(_mm512_add_ps(t0, t1), T[j][0]);
        O[1] = _mm512_fmadd_ps(_mm512_set1_ps(2.f), t3, t2);
        O[2] = _mm512_fmadd_ps(_mm512_set1_ps(4.f), t1, t0);
        O[3] = _mm512_add_ps(_mm512_fmadd_ps(_mm512_set1_ps(8.f), t3, t2), T[j][5]);

        ydim = tj*(ALPHA - 2) + j;

        for (i = 0; i < LIBXSMM_MIN(handle->ofw - (int)ti*(ALPHA-2), ALPHA-2); i++) {
          xdim = ti*(ALPHA - 2) + i;
          _mm512_store_ps(
              &LIBXSMM_VLA_ACCESS(4, output, 0, ydim, xdim, 0, handle->ofhp, handle->ofwp, TDVLEN),
              _mm512_add_ps(
                  _mm512_load_ps(&LIBXSMM_VLA_ACCESS(4, output, 0, ydim, xdim, 0, handle->ofhp, handle->ofwp, TDVLEN)),
                  O[i]));
        }
      }
    }
    /* inline code end */
  } /* for each tile */
}

#undef TEMP_PREFETCH_DISTANCE
