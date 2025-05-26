#include "reseau.h"
#include "tenseur.h"
#include <math.h>
#include <stdlib.h>

// fonctions utilitaires
float f_act_leaky_relu(float x) { return x > 0 ? x : 0.01 * x; }

float df_act_leaky_relu(float x) { return x > 0 ? 1 : 0.01; }

float f_act_sigmoid(float x) {
  if (x > 16)
    return 1.0f; // 1/(1+e^-16) ≃ 0.999
  if (x < -16)
    return 0.0f;
  return 1.0f / (1.0f + expf(-x));
}

float df_act_sigmoid(float x) {
  float s = f_act_sigmoid(x);
  return s * (1 - s);
}

float f_act_tanh(float x) { return tanhf(x); }

float df_act_tanh(float x) {
  float t = f_act_tanh(x);
  return 1 - t * t;
}

// réseau
reseau initialiser_reseau(size_t nbr_couches, size_t nbr_lots,
                          TypeCouche *types, size_t *dim_poids,
                          size_t dim_entree[2], size_t *prof) {
  reseau r = {.nbr_couches = nbr_couches,
              .nbr_lots = nbr_lots,
              .t = malloc(sizeof(int)),
              .c = calloc(nbr_couches, sizeof(couche)),
              .threads = malloc(r.nbr_lots * sizeof(pthread_t)),
              .gel = false};
  *r.t = 0;
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];
    cur->type = types[l];

    switch (cur->type) {
    case COUCHE_ENTREE: {
      cur->z = t_init(nbr_lots, prof[l], dim_entree[0], dim_entree[1]);
      // z[n][p][h][w]
      break;
    }
    case COUCHE_CONV: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];
      size_t k = dim_poids[l];

      cur->z = t_init(nbr_lots, prof[l], prec->z.dim[2] - k + 1,
                      prec->z.dim[3] - k + 1);
      cur->poids =
          t_init(prof[l], prof[l - 1], k, k); // poids[c_out][c_in][k][k]
      cur->biais = t_init(1, 1, 1, prof[l]);  // biais[c_out][0][0][0]

      cur->m_p = t_init(prof[l], prof[l - 1], k, k); // m[c_out][c_in][k][k]
      cur->v_p = t_init(prof[l], prof[l - 1], k, k); // v[c_out][c_in][k][k]

      cur->m_b = t_init(1, 1, 1, prof[l]); // m[c_out][0][0][0]
      cur->v_b = t_init(1, 1, 1, prof[l]); // v[c_out][0][0][0]
      break;
    }
    case COUCHE_TCONV: {
      assert(l > 0);
      int s = 2; // stride
      couche *prec = &r.c[l - 1];
      size_t k = dim_poids[l];

      cur->z = t_init(nbr_lots, prof[l], (prec->z.dim[2] - 1) * s + k,
                      (prec->z.dim[3] - 1) * s + k);
      cur->poids =
          t_init(prof[l], prof[l - 1], k, k); // poids[c_out][c_in][k][k]
      cur->biais = t_init(1, 1, 1, prof[l]);  // biais[c_out][0][0][0]

      cur->m_p = t_init(prof[l], prof[l - 1], k, k); // m[c_out][c_in][k][k]
      cur->v_p = t_init(prof[l], prof[l - 1], k, k); // v[c_out][c_in][k][k]

      cur->m_b = t_init(1, 1, 1, prof[l]); // m[c_out][0][0][0]
      cur->v_b = t_init(1, 1, 1, prof[l]); // v[c_out][0][0][0]
      break;
    }
    case COUCHE_SE: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2] / 2,
                      prec->z.dim[3] / 2);
      break;
    }
    case COUCHE_APLAT: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, 1,
                      prec->z.dim[1] * prec->z.dim[2] * prec->z.dim[3], 1);
      break;
    }
    case COUCHE_DEAPLAT: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];
      int d = (int)sqrt((float)prec->z.dim[2] / prof[l]);

      cur->z = t_init(nbr_lots, prof[l], d, d);
      break;
    }
    case COUCHE_DENSE: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, 1, dim_poids[l], 1);
      cur->poids = t_init(1, 1, dim_poids[l], prec->z.dim[2]);
      cur->biais = t_init(1, 1, 1, dim_poids[l]);

      cur->m_p = t_init(1, 1, dim_poids[l], prec->z.dim[2]);
      cur->v_p = t_init(1, 1, dim_poids[l], prec->z.dim[2]);
      cur->m_b = t_init(1, 1, 1, dim_poids[l]);
      cur->v_b = t_init(1, 1, 1, dim_poids[l]);
      break;
    }
    case COUCHE_ACT_RELU: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2], prec->z.dim[3]);
      break;
    }
    case COUCHE_ACT_SIGM: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2], prec->z.dim[3]);
      break;
    }
    case COUCHE_ACT_TANH: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2], prec->z.dim[3]);
      break;
    }
    case COUCHE_SOFTMAX: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2], prec->z.dim[3]);
      break;
    }
    case COUCHE_SORTIE: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2], prec->z.dim[3]);
      break;
    }
    case COUCHE_SORTIE_SIGM: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      cur->z = t_init(nbr_lots, prec->z.dim[1], prec->z.dim[2], prec->z.dim[3]);
      break;
    }
    }
  }
  r.delta = init_erreur(r);
  return r;
}

void liberer_reseau(reseau r) {
  for (size_t l = 0; l < r.nbr_couches; l++) {
    t_liberer(r.c[l].z);
    if (r.c[l].type == COUCHE_CONV || r.c[l].type == COUCHE_DENSE ||
        r.c[l].type == COUCHE_TCONV) {
      t_liberer(r.c[l].poids);
      t_liberer(r.c[l].biais);
      t_liberer(r.c[l].m_p);
      t_liberer(r.c[l].v_p);
      t_liberer(r.c[l].m_b);
      t_liberer(r.c[l].v_b);
    }
  }
  free(r.c);
  free(r.t);
  free(r.threads);
  liberer_erreur(r, r.delta);
}

struct Thread_param_propagation_avant {
  reseau r;
  int n;
};

void *propagation_avant_aux(void *param) {
  struct Thread_param_propagation_avant p =
      *(struct Thread_param_propagation_avant *)param;
  reseau r = p.r;
  int n = p.n;
  free(param);
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];
    switch (cur->type) {
    case COUCHE_ENTREE: {
      break;
    }
      // rien à faire ici
    case COUCHE_CONV: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];
      size_t k = cur->poids.dim[2];

      for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
        for (size_t i = 0; i < cur->z.dim[2]; ++i) {
          for (size_t j = 0; j < cur->z.dim[3]; j++) {
            float sum = 0;
            for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
              for (size_t u = 0; u < k; u++) {
                for (size_t v = 0; v < k; v++) {
                  sum += t_get(cur->poids, c_out, c_in, u, v) *
                         t_get(prec->z, n, c_in, i + u, j + v);
                }
              }
            }
            sum += t_get(cur->biais, 0, 0, 0, c_out);
            if (!isfinite(sum)) {
              fprintf(stderr, "NaN ou INF – couche %zu, lot %d\n", l, n);
              exit(EXIT_FAILURE);
            }
            t_set(cur->z, n, c_out, i, j, sum);
          }
        }
      }
      break;
    }
    case COUCHE_TCONV: {
      assert(l > 0);
      int s = 2;
      couche *prec = &r.c[l - 1];
      size_t k = cur->poids.dim[2];
      for (int c_out = 0; c_out < cur->z.dim[1]; c_out++) {
        for (int i = 0; i < cur->z.dim[2]; i++) {
          for (int j = 0; j < cur->z.dim[3]; j++) {
            float sum = 0;
            for (int c_in = 0; c_in < prec->z.dim[1]; c_in++) {
              for (int u = 0; u < k; u++) {
                for (int v = 0; v < k; v++) {
                  int ii = (i + u - k + 1) / s;
                  int jj = (j + v - k + 1) / s;
                  if (ii >= 0 && jj >= 0 && ii < prec->z.dim[2] &&
                      jj < prec->z.dim[3] && (i + u - k + 1) % s == 0 &&
                      (j + v - k + 1) % s == 0) {
                    sum += t_get(cur->poids, c_out, c_in, u, v) *
                           t_get(prec->z, n, c_in, ii, jj);
                  }
                }
              }
            }
            sum += t_get(cur->biais, 0, 0, 0, c_out);
            if (!isfinite(sum)) {
              fprintf(stderr, "NaN ou INF – couche %zu, lot %d\n", l, n);
              exit(EXIT_FAILURE);
            }
            t_set(cur->z, n, c_out, i, j, sum);
          }
        }
      }
      break;
    }
      // à faire
    case COUCHE_SE: {
      {
        assert(l > 0);
        couche *prec = &r.c[l - 1];

        for (size_t p = 0; p < cur->z.dim[1]; ++p) {
          for (size_t i = 0; i < cur->z.dim[2]; ++i) {
            for (size_t j = 0; j < cur->z.dim[3]; ++j) {
              float max = t_get(prec->z, n, p, i * 2, j * 2);
              for (size_t u = 0; u < 2; ++u) {
                for (size_t v = 0; v < 2; ++v) {
                  if (max < t_get(prec->z, n, p, i * 2 + u, j * 2 + v)) {
                    max = t_get(prec->z, n, p, i * 2 + u, j * 2 + v);
                  }
                }
              }
              t_set(cur->z, n, p, i, j, max);
            }
          }
        }
        break;
      }
    }
      //
    case COUCHE_APLAT: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; ++j) {
            float val = t_get(prec->z, n, p, i, j);
            size_t idx =
                p * prec->z.dim[2] * prec->z.dim[3] + i * prec->z.dim[3] + j;
            t_set(cur->z, n, 0, idx, 0, val);
          }
        }
      }
      break;
    }
    case COUCHE_DEAPLAT: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < cur->z.dim[1]; ++p) {
        for (size_t i = 0; i < cur->z.dim[2]; ++i) {
          for (size_t j = 0; j < cur->z.dim[3]; ++j) {

            size_t idx =
                p * cur->z.dim[2] * cur->z.dim[3] + i * cur->z.dim[3] + j;
            float val = t_get(prec->z, n, 0, idx, 0);
            t_set(cur->z, n, p, i, j, val);
          }
        }
      }
      break;
    }
    case COUCHE_DENSE: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];
      for (size_t i = 0; i < cur->z.dim[2]; ++i) {
        float sum = 0;
        for (size_t k = 0; k < prec->z.dim[2]; ++k) {
          sum += t_get(cur->poids, 0, 0, i, k) * t_get(prec->z, n, 0, k, 0);
        }
        sum += t_get(cur->biais, 0, 0, 0, i);
        if (!isfinite(sum)) {
          fprintf(stderr, "NaN ou INF – couche %zu, lot %d\n", l, n);
          exit(EXIT_FAILURE);
        }
        t_set(cur->z, n, 0, i, 0, sum);
      }
      break;
    }
      //
    case COUCHE_ACT_RELU: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            t_set(cur->z, n, p, i, j, f_act_leaky_relu(val));
          }
        }
      }
      break;
    }
    case COUCHE_ACT_SIGM: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            t_set(cur->z, n, p, i, j, f_act_sigmoid(val));
          }
        }
      }
      break;
    }
    case COUCHE_ACT_TANH: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            t_set(cur->z, n, p, i, j, f_act_tanh(val));
          }
        }
      }
      break;
    }
    case COUCHE_SOFTMAX: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      float maxval = -10e10;
      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            maxval = fmaxf(maxval, val);
          }
        }
      }

      float sum = 0;
      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            sum += exp(val);
          }
        }
      }
      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            t_set(cur->z, n, p, i, j, exp(val) / sum);
          }
        }
      }
      break;
    }
    case COUCHE_SORTIE: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            t_set(cur->z, n, p, i, j, val);
          }
        }
      }
      break;
    }
    case COUCHE_SORTIE_SIGM: {
      assert(l > 0);
      couche *prec = &r.c[l - 1];

      for (size_t p = 0; p < prec->z.dim[1]; ++p) {
        for (size_t i = 0; i < prec->z.dim[2]; ++i) {
          for (size_t j = 0; j < prec->z.dim[3]; j++) {
            float val = t_get(prec->z, n, p, i, j);
            t_set(cur->z, n, p, i, j, f_act_sigmoid(val));
          }
        }
      }
      break;
    }
    }
  }
  return NULL;
}

void propagation_avant(reseau r, int nmb_optn) {
  int nmb;
  if (nmb_optn == -1) {
    nmb = r.nbr_lots;
  } else {
    nmb = nmb_optn;
  }

  for (int n = 0; n < nmb; ++n) {
    struct Thread_param_propagation_avant *param = malloc(sizeof *param);
    *param = (struct Thread_param_propagation_avant){r, n};

    pthread_create(&r.threads[n], NULL, propagation_avant_aux, param);
  }

  for (int n = 0; n < nmb; n++) {
    pthread_join(r.threads[n], NULL);
  }
}

tenseur *init_erreur(reseau r) {
  tenseur *delta = malloc(r.nbr_couches * sizeof(tenseur));
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];
    delta[l] = t_init(r.nbr_lots, cur->z.dim[1], cur->z.dim[2], cur->z.dim[3]);
  }
  return delta;
}

float distrib_alea(float x) {
  return x * (2 * (((float)rand()) / RAND_MAX) - 1);
}

float he_std_conv(size_t k, size_t c_in) {
  return sqrtf(2.0f / (k * k * c_in));
}
float he_std_dense(size_t fan_in) { return sqrtf(2.0f / fan_in); }

void init_poids_biais(reseau r) {
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];

    if (cur->type == COUCHE_CONV || cur->type == COUCHE_TCONV) {
      size_t k = cur->poids.dim[3];
      for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
        for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
          for (size_t u = 0; u < k; u++) {
            for (size_t v = 0; v < k; v++) {
              t_set(cur->poids, c_out, c_in, u, v,
                    distrib_alea(he_std_conv(k, cur->poids.dim[1])));
            }
          }
        }
        t_set(cur->biais, 0, 0, 0, c_out, 0);
      }
    }
    if (cur->type == COUCHE_DENSE) {
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        for (size_t j = 0; j < cur->poids.dim[3]; ++j) {
          t_set(cur->poids, 0, 0, i, j,
                distrib_alea(he_std_dense(cur->poids.dim[3])));
        }

        t_set(cur->biais, 0, 0, 0, i, 0);
      }
    }
  }
}

void liberer_erreur(reseau r, tenseur *delta) {
  for (size_t l = 0; l < r.nbr_couches; l++) {
    t_liberer(delta[l]);
  }
  free(delta);
}

void mettre_a_jour_poids_biais_moments(reseau r, float alpha, float beta1,
                                       float beta2) {
  if (r.gel == true) {
    return;
  }
  *r.t += 1;
  float beta1_exp_t = pow(beta1, *r.t);
  float beta2_exp_t = pow(beta2, *r.t);
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];
    if (cur->type == COUCHE_CONV || cur->type == COUCHE_DENSE ||
        cur->type == COUCHE_TCONV) {
      // Mise à jour des biais
      if (cur->type == COUCHE_DENSE) {
        for (size_t i = 0; i < cur->biais.dim[3]; ++i) {
          float grad_b = 0;
          for (size_t n = 0; n < r.nbr_lots; ++n) {
            grad_b += t_get(r.delta[l], n, 0, i, 0);
          }
          grad_b /= (r.nbr_lots * cur->biais.dim[3] * cur->biais.dim[2]);

          float b = t_get(cur->biais, 0, 0, 0, i);

          float m_prec = t_get(cur->m_b, 0, 0, 0, i);
          float v_prec = t_get(cur->v_b, 0, 0, 0, i);

          t_set(cur->m_b, 0, 0, 0, i,

                beta1 * m_prec + (1 - beta1) * (grad_b));
          t_set(cur->v_b, 0, 0, 0, i,
                beta2 * v_prec + (1 - beta2) * (grad_b) * (grad_b));

          float mt = t_get(cur->m_b, 0, 0, 0, i);
          float vt = t_get(cur->v_b, 0, 0, 0, i);

          float mt_hat = mt / (1 - beta1_exp_t);
          float vt_hat = vt / (1 - beta2_exp_t);

          t_set(cur->biais, 0, 0, 0, i,
                b - alpha * mt_hat / (sqrtf(vt_hat) + EPS));
        }
      } else if (cur->type == COUCHE_CONV || cur->type == COUCHE_TCONV) {
        for (size_t c_out = 0; c_out < cur->biais.dim[3]; ++c_out) {
          float grad_b = 0;
          for (size_t n = 0; n < r.nbr_lots; ++n) {
            for (size_t i = 0; i < r.delta[l].dim[2]; ++i) {
              for (size_t j = 0; j < r.delta[l].dim[3]; ++j) {
                grad_b += t_get(r.delta[l], n, c_out, i, j);
              }
            }
          }
          grad_b /= r.nbr_lots;
          float b = t_get(cur->biais, 0, 0, 0, c_out);
          float m_prec = t_get(cur->m_b, 0, 0, 0, c_out);
          float v_prec = t_get(cur->v_b, 0, 0, 0, c_out);
          t_set(cur->m_b, 0, 0, 0, c_out,

                beta1 * m_prec + (1 - beta1) * (grad_b));
          t_set(cur->v_b, 0, 0, 0, c_out,
                beta2 * v_prec + (1 - beta2) * (grad_b) * (grad_b));

          float mt = t_get(cur->m_b, 0, 0, 0, c_out);
          float vt = t_get(cur->v_b, 0, 0, 0, c_out);

          float mt_hat = mt / (1 - beta1_exp_t);
          float vt_hat = vt / (1 - beta2_exp_t);

          t_set(cur->biais, 0, 0, 0, c_out,
                b - alpha * mt_hat / (sqrtf(vt_hat) + EPS));
        }
      }

      // Mise à jour des poids
      if (cur->type == COUCHE_DENSE) {
        couche *prec = &r.c[l - 1];
        for (size_t i = 0; i < cur->poids.dim[2]; ++i) {   // couche courante
          for (size_t j = 0; j < cur->poids.dim[3]; ++j) { // couche précédente
            float grad_w = 0;
            for (size_t n = 0; n < r.nbr_lots; ++n) {
              float delta_val = t_get(r.delta[l], n, 0, i, 0);
              float a_prec = t_get(prec->z, n, 0, j, 0);
              grad_w += delta_val * a_prec;
            }
            grad_w /= r.nbr_lots;
            float w = t_get(cur->poids, 0, 0, i, j);

            float m_prec = t_get(cur->m_p, 0, 0, i, j);
            float v_prec = t_get(cur->v_p, 0, 0, i, j);
            t_set(cur->m_p, 0, 0, i, j,
                  beta1 * m_prec + (1 - beta1) * (grad_w));
            t_set(cur->v_p, 0, 0, i, j,
                  beta2 * v_prec + (1 - beta2) * (grad_w) * (grad_w));

            float mt = t_get(cur->m_p, 0, 0, i, j);
            float vt = t_get(cur->v_p, 0, 0, i, j);

            float mt_hat = mt / (1 - beta1_exp_t);
            float vt_hat = vt / (1 - beta2_exp_t);

            t_set(cur->poids, 0, 0, i, j,
                  w - alpha * mt_hat / (sqrtf(vt_hat) + EPS));
          }
        }
      } else if (cur->type == COUCHE_CONV) {
        couche *prec = &r.c[l - 1];
        size_t k = cur->poids.dim[2]; // Taille du noyau
        for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
          for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
            for (size_t u = 0; u < k; ++u) {
              for (size_t v = 0; v < k; ++v) {
                float grad_w = 0;
                for (size_t n = 0; n < r.nbr_lots; ++n) {
                  for (size_t i = 0; i < r.delta[l].dim[2]; ++i) {
                    for (size_t j = 0; j < r.delta[l].dim[3]; ++j) {
                      if (i + u < prec->z.dim[2] && j + v < prec->z.dim[3]) {
                        float a_prec = t_get(prec->z, n, c_in, i + u, j + v);
                        float delta_val = t_get(r.delta[l], n, c_out, i, j);
                        grad_w += a_prec * delta_val;
                      }
                    }
                  }
                }
                grad_w /= r.nbr_lots;
                float w = t_get(cur->poids, c_out, c_in, u, v);

                float m_prec = t_get(cur->m_p, c_out, c_in, u, v);
                float v_prec = t_get(cur->v_p, c_out, c_in, u, v);

                t_set(cur->m_p, c_out, c_in, u, v,

                      beta1 * m_prec + (1 - beta1) * (grad_w));
                t_set(cur->v_p, c_out, c_in, u, v,
                      beta2 * v_prec + (1 - beta2) * (grad_w) * (grad_w));

                float mt = t_get(cur->m_p, c_out, c_in, u, v);
                float vt = t_get(cur->v_p, c_out, c_in, u, v);

                float mt_hat = mt / (1 - beta1_exp_t);
                float vt_hat = vt / (1 - beta2_exp_t);

                t_set(cur->poids, c_out, c_in, u, v,
                      w - alpha * mt_hat / (sqrtf(vt_hat) + EPS));
              }
            }
          }
        }
      } else if (cur->type == COUCHE_TCONV) {
        couche *prec = &r.c[l - 1];
        int k = cur->poids.dim[2];
        int s = 2; /* stride */
        for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
          for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
            for (size_t u = 0; u < k; ++u) {
              for (size_t v = 0; v < k; ++v) {

                float grad_w = 0.f;
                for (size_t n = 0; n < r.nbr_lots; ++n) {
                  for (size_t i = 0; i < prec->z.dim[2]; ++i) {
                    for (size_t j = 0; j < prec->z.dim[3]; ++j) {
                      size_t out_i = i * s - u + k - 1;
                      size_t out_j = j * s - v + k - 1;
                      if (out_i >= 0 && out_i < r.delta[l].dim[2] &&
                          out_j >= 0 && out_j < r.delta[l].dim[3]) {
                        grad_w += t_get(prec->z, n, c_in, i, j) *
                                  t_get(r.delta[l], n, c_out, out_i, out_j);
                      }
                    }
                  }
                }
                grad_w /= r.nbr_lots;
                float w = t_get(cur->poids, c_out, c_in, u, v);

                float m_prec = t_get(cur->m_p, c_out, c_in, u, v);
                float v_prec = t_get(cur->v_p, c_out, c_in, u, v);

                t_set(cur->m_p, c_out, c_in, u, v,

                      beta1 * m_prec + (1 - beta1) * (grad_w));
                t_set(cur->v_p, c_out, c_in, u, v,
                      beta2 * v_prec + (1 - beta2) * (grad_w) * (grad_w));

                float mt = t_get(cur->m_p, c_out, c_in, u, v);
                float vt = t_get(cur->v_p, c_out, c_in, u, v);

                float mt_hat = mt / (1 - beta1_exp_t);
                float vt_hat = vt / (1 - beta2_exp_t);

                t_set(cur->poids, c_out, c_in, u, v,
                      w - alpha * mt_hat / (sqrtf(vt_hat) + EPS));
              }
            }
          }
        }
      }
    }
  }
}

struct Thread_param_retropropagations {
  reseau r;
  float **y_attendu;
  int n;
  bool delta_param;
};

void *retropropagation_aux(void *param) {
  struct Thread_param_retropropagations p =
      *(struct Thread_param_retropropagations *)param;
  reseau r = p.r;
  float **y_attendu = p.y_attendu;
  int n = p.n;
  int delta_param = p.delta_param;
  free(param);

  size_t L = r.nbr_couches - 1;
  for (int l = L; l >= 0; --l) {
    couche *cur = &r.c[l];
    if (l == L) {
      assert(cur->type == COUCHE_SORTIE || cur->type == COUCHE_SORTIE_SIGM);
      // for (size_t i = 0; i < cur->z.dim[2]; i++) { // Cross-Entropy
      //   float val = -y_attendu[n][i] / (1e-12f + t_get(cur->z, n, 0, i,
      //   0)); t_set(delta[l], n, 0, i, 0, val);
      // }
      if (p.delta_param == false && cur->type == COUCHE_SORTIE) {
        for (size_t i = 0; i < cur->z.dim[2];
             i++) { // MSE / Cross-Entropy + SOFTMAX
          float val = t_get(cur->z, n, 0, i, 0) - y_attendu[n][i];
          t_set(r.delta[l], n, 0, i, 0, val);
        }
      }
      if (p.delta_param == false && cur->type == COUCHE_SORTIE_SIGM) {
        for (size_t i = 0; i < cur->z.dim[2];
             i++) { // MSE / Cross-Entropy + SOFTMAX
          float val = t_get(cur->z, n, 0, i, 0) - y_attendu[n][i];
          t_set(r.delta[l], n, 0, i, 0, val);
        }
      }

    } else {
      couche *proc = &r.c[l + 1];
      switch (proc->type) {
      case COUCHE_SORTIE: {
        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              float val = t_get(r.delta[l + 1], n, p, i, j);
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      case COUCHE_ENTREE: {
        // rien à faire ici
        break;
      }
      case COUCHE_CONV: {

        int k = proc->poids.dim[2];

        for (int c_in = 0; c_in < cur->z.dim[1]; ++c_in) {
          for (int i = 0; i < cur->z.dim[2]; ++i) {
            for (int j = 0; j < cur->z.dim[3]; ++j) {
              // operation de convolution

              float sum = 0;
              for (int c_out = 0; c_out < proc->z.dim[1]; ++c_out) {
                for (int u = 0; u < k; u++) {
                  for (int v = 0; v < k; v++) {
                    if (i - u >= 0 && i - u < proc->z.dim[2] && j - v >= 0 &&
                        j - v < proc->z.dim[3]) {
                      sum += t_get(r.delta[l + 1], n, c_out, i - u, j - v) *
                             t_get(proc->poids, c_out, c_in, u, v);
                    }
                  }
                }
              }
              t_set(r.delta[l], n, c_in, i, j, sum);
            }
          }
        }
        break;
      }
      case COUCHE_TCONV: {
        int k = proc->poids.dim[2];
        int s = 2; /* stride */
        for (int c_in = 0; c_in < cur->z.dim[1]; ++c_in) {
          for (int ii = 0; ii < cur->z.dim[2]; ++ii) {
            for (int jj = 0; jj < cur->z.dim[3]; ++jj) {

              float sum = 0.f;

              for (int c_out = 0; c_out < proc->z.dim[1]; ++c_out) {
                for (int u = 0; u < k; ++u) {
                  for (int v = 0; v < k; ++v) {

                    int i_out = ii * s + (k - 1 - u);
                    int j_out = jj * s + (k - 1 - v);

                    if (i_out >= 0 && i_out < proc->z.dim[2] && j_out >= 0 &&
                        j_out < proc->z.dim[3]) {

                      sum += t_get(proc->poids, c_out, c_in, u, v) *
                             t_get(r.delta[l + 1], n, c_out, i_out, j_out);
                    }
                  }
                }
              }
              t_set(r.delta[l], n, c_in, ii, jj, sum);
            }
          }
        }
        break;
      }

      case COUCHE_SE: {

        for (size_t p = 0; p < cur->z.dim[1]; ++p) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              // on cherche le maximum parmis les voisins

              if (i / 2 < proc->z.dim[2] && j / 2 < proc->z.dim[3]) {
                if (fabs(t_get(cur->z, n, p, i, j) -
                         t_get(proc->z, n, p, i / 2, j / 2)) < 1e-6) {

                  float val = t_get(r.delta[l + 1], n, p, i / 2, j / 2);
                  t_set(r.delta[l], n, p, i, j, val);
                } else {
                  t_set(r.delta[l], n, p, i, j, 0);
                }
              }
            }
          }
        }
        break;
      }
      case COUCHE_APLAT: {
        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              size_t idx =
                  p * cur->z.dim[3] * cur->z.dim[2] + i * cur->z.dim[3] + j;
              float val = t_get(r.delta[l + 1], n, 0, idx, 0);
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      case COUCHE_DEAPLAT: {
        for (size_t p = 0; p < proc->z.dim[1]; ++p) {
          for (size_t i = 0; i < proc->z.dim[2]; ++i) {
            for (size_t j = 0; j < proc->z.dim[3]; ++j) {
              float val = t_get(r.delta[l + 1], n, p, i, j);
              size_t idx =
                  p * proc->z.dim[2] * proc->z.dim[3] + i * proc->z.dim[3] + j;

              t_set(r.delta[l], n, 0, idx, 0, val);
            }
          }
        }
        break;
      }
      case COUCHE_DENSE: {
        for (size_t i = 0; i < cur->z.dim[2]; i++) {
          float sum = 0;
          for (size_t j = 0; j < proc->z.dim[2]; j++) {
            sum += t_get(proc->poids, 0, 0, j, i) *
                   t_get(r.delta[l + 1], n, 0, j, 0);
          }
          t_set(r.delta[l], n, 0, i, 0, sum);
        }

        break;
      }
      case COUCHE_ACT_RELU: {
        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              float val = t_get(r.delta[l + 1], n, p, i, j) *
                          df_act_leaky_relu(t_get(cur->z, n, p, i, j));
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      case COUCHE_ACT_SIGM: {
        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              float val = t_get(r.delta[l + 1], n, p, i, j) *
                          df_act_sigmoid(t_get(cur->z, n, p, i, j));
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      case COUCHE_ACT_TANH: {
        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              float val = t_get(r.delta[l + 1], n, p, i, j) *
                          df_act_tanh(t_get(cur->z, n, p, i, j));
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      case COUCHE_SOFTMAX: {
        // softmax if not used with cross-entropy on the last layer
        //  for (size_t p_1 = 0; p_1 < cur->z.dim[1]; p_1++) {
        //    for (size_t i_1 = 0; i_1 < cur->z.dim[2]; i_1++) {
        //      for (size_t j_1 = 0; j_1 < cur->z.dim[3]; j_1++) {

        //       float sum = 0;
        //       float z1 = t_get(proc->z, n, p_1, i_1, j_1);
        //       for (size_t p_2 = 0; p_2 < cur->z.dim[1]; p_2++) {
        //         for (size_t i_2 = 0; i_2 < cur->z.dim[2]; i_2++) {
        //           for (size_t j_2 = 0; j_2 < cur->z.dim[3]; j_2++) {
        //             float val = t_get(delta[l + 1], n, p_2, i_2, j_2);

        //             float z2 = t_get(proc->z, n, p_2, i_2, j_2);
        //             if (i_1 == i_2 && j_1 == j_2 && p_1 == p_2) {
        //               sum += val * z1 * (1 - z1);
        //             } else {
        //               sum += val * -z1 * z2;
        //             }
        //           }
        //         }
        //       }
        //       t_set(delta[l], n, p_1, i_1, j_1, sum);
        //     }
        //   }
        // }

        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              float val = t_get(r.delta[l + 1], n, p, i, j);
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      case COUCHE_SORTIE_SIGM: {
        for (size_t p = 0; p < cur->z.dim[1]; p++) {
          for (size_t i = 0; i < cur->z.dim[2]; i++) {
            for (size_t j = 0; j < cur->z.dim[3]; j++) {
              float val = t_get(r.delta[l + 1], n, p, i, j);
              t_set(r.delta[l], n, p, i, j, val);
            }
          }
        }
        break;
      }
      }
    }
  }
  return NULL;
}

void retropropagation(reseau r, float alpha, float beta1, float beta2,
                      float **y_attendu, tenseur *delta) {
  bool delta_param = false;
  if (delta != NULL) {
    t_copier(*delta, r.delta[r.nbr_couches - 1]);
    delta_param = true;
  }

  for (int n = 0; n < r.nbr_lots; ++n) {
    struct Thread_param_retropropagations *param = malloc(sizeof *param);
    *param =
        (struct Thread_param_retropropagations){r, y_attendu, n, delta_param};

    pthread_create(&r.threads[n], NULL, retropropagation_aux, param);
  }

  for (int n = 0; n < r.nbr_lots; n++) {
    pthread_join(r.threads[n], NULL);
  }

  mettre_a_jour_poids_biais_moments(r, alpha, beta1, beta2);
}

void sauver_reseau(reseau r, const char *nom_fichier) {
  FILE *f = fopen(nom_fichier, "w");
  if (!f) {
    fprintf(stderr, "sauver_reseau : impossible d’ouvrir %s (%s)\n",
            nom_fichier, strerror(errno));
    return;
  }

  // sauver format
  fprintf(f, "%zu %zu", r.nbr_couches, r.nbr_lots);
  // types
  fprintf(f, "\n");
  for (int l = 0; l < r.nbr_couches; l++) {
    fprintf(f, "%d ", r.c[l].type);
  }
  fprintf(f, "\n");

  // dim_poids
  for (int l = 0; l < r.nbr_couches; l++) {
    if (r.c[l].type == COUCHE_CONV || r.c[l].type == COUCHE_TCONV) {
      fprintf(f, "%zu ", r.c[l].poids.dim[2]);
    } else if (r.c[l].type == COUCHE_DENSE) {
      fprintf(f, "%zu ", r.c[l].z.dim[2]);
    } else {
      fprintf(f, "0 ");
    }
  }
  fprintf(f, "\n");
  // dim_entree
  fprintf(f, "%zu  %zu", r.c[0].z.dim[2], r.c[0].z.dim[3]);
  fprintf(f, "\n");
  // prof
  for (int l = 0; l < r.nbr_couches; l++) {
    fprintf(f, "%zu ", r.c[l].z.dim[1]);
  }
  // sauver biais et poids
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];
    if (cur->type == COUCHE_CONV || cur->type == COUCHE_TCONV) {
      size_t k = cur->poids.dim[3];
      fprintf(f, "\n");
      for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
        for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
          for (size_t u = 0; u < k; u++) {
            fprintf(f, "\n");
            for (size_t v = 0; v < k; v++) {
              fprintf(f, "%f ", t_get(cur->poids, c_out, c_in, u, v));
            }
          }
          fprintf(f, "\n");
        }
        fprintf(f, "%f ", t_get(cur->biais, 0, 0, 0, c_out));
      }
    }
    if (cur->type == COUCHE_DENSE) {
      fprintf(f, "\n");
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        for (size_t j = 0; j < cur->poids.dim[3]; ++j) {
          fprintf(f, "%f ", t_get(cur->poids, 0, 0, i, j));
        }
        fprintf(f, "\n");
      }
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        fprintf(f, "%f ", t_get(cur->biais, 0, 0, 0, i));
      }
    }
  }
  fclose(f);

  // adam
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];
    if (cur->type == COUCHE_CONV || cur->type == COUCHE_TCONV) {
      size_t k = cur->poids.dim[3];
      fprintf(f, "\n");
      for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
        for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
          for (size_t u = 0; u < k; u++) {
            fprintf(f, "\n");
            for (size_t v = 0; v < k; v++) {
              fprintf(f, "%f ", t_get(cur->m_p, c_out, c_in, u, v));
              fprintf(f, "%f ", t_get(cur->v_p, c_out, c_in, u, v));
            }
          }
          fprintf(f, "\n");
        }
        fprintf(f, "%f ", t_get(cur->m_b, 0, 0, 0, c_out));
        fprintf(f, "%f ", t_get(cur->v_b, 0, 0, 0, c_out));
      }
    }
    if (cur->type == COUCHE_DENSE) {
      fprintf(f, "\n");
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        for (size_t j = 0; j < cur->poids.dim[3]; ++j) {
          fprintf(f, "%f ", t_get(cur->m_p, 0, 0, i, j));
          fprintf(f, "%f ", t_get(cur->v_p, 0, 0, i, j));
        }
        fprintf(f, "\n");
      }
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        fprintf(f, "%f ", t_get(cur->m_b, 0, 0, 0, i));
        fprintf(f, "%f ", t_get(cur->v_b, 0, 0, 0, i));
      }
    }
  }
  fclose(f);
}

reseau charger_reseau(const char *nom_fichier) {
  FILE *f = fopen(nom_fichier, "r");
  if (!f) {
    fprintf(stderr, "charger_reseau : impossible d’ouvrir %s (%s)\n",
            nom_fichier, strerror(errno));
  }

  // sauver format
  size_t nbr_couches, nbr_lots;
  fscanf(f, "%zu %zu", &nbr_couches, &nbr_lots);
  // types
  fscanf(f, "\n");
  TypeCouche *types = calloc(nbr_couches, sizeof((types)));
  size_t *dim_poids = calloc(nbr_couches, sizeof((types)));
  size_t dim_entree[2];
  size_t *prof = calloc(nbr_couches, sizeof((types)));

  for (int l = 0; l < nbr_couches; l++) {
    int t;
    fscanf(f, "%d ", &t);
    types[l] = (TypeCouche)t;
  }
  fscanf(f, "\n");
  // dim_poids

  for (int l = 0; l < nbr_couches; l++) {
    fscanf(f, "%zu ", &dim_poids[l]);
  }
  fscanf(f, "\n");
  // dim_entree
  fscanf(f, "%zu  %zu", &dim_entree[0], &dim_entree[1]);
  fscanf(f, "\n");
  // prof
  for (int l = 0; l < nbr_couches; l++) {
    fscanf(f, "%zu ", &prof[l]);
  }

  reseau r = initialiser_reseau(nbr_couches, nbr_lots, types, dim_poids,
                                dim_entree, prof);
  free(types);
  free(dim_poids);
  free(prof);

  // sauver biais et poids
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];

    if (cur->type == COUCHE_CONV || cur->type == COUCHE_TCONV) {
      fscanf(f, "\n");
      size_t k = cur->poids.dim[3];
      for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
        for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
          for (size_t u = 0; u < k; u++) {
            fscanf(f, "\n");
            for (size_t v = 0; v < k; v++) {
              float val;
              fscanf(f, "%f", &val);
              t_set(cur->poids, c_out, c_in, u, v, val);
            }
          }
          fscanf(f, "\n");
        }
        float val;
        fscanf(f, "%f", &val);
        t_set(cur->biais, 0, 0, 0, c_out, val);
      }
    }

    if (cur->type == COUCHE_DENSE) {
      fscanf(f, "\n");
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        for (size_t j = 0; j < cur->poids.dim[3]; ++j) {
          float val;
          fscanf(f, "%f", &val);
          t_set(cur->poids, 0, 0, i, j, val);
        }
        fscanf(f, "\n");
      }
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        float val;
        fscanf(f, "%f", &val);
        t_set(cur->biais, 0, 0, 0, i, val);
      }
    }
  }
  fclose(f);
  return r;

  // ADAM:
  for (size_t l = 0; l < r.nbr_couches; l++) {
    couche *cur = &r.c[l];

    if (cur->type == COUCHE_CONV || cur->type == COUCHE_TCONV) {
      fscanf(f, "\n");
      size_t k = cur->poids.dim[3];
      for (size_t c_out = 0; c_out < cur->poids.dim[0]; ++c_out) {
        for (size_t c_in = 0; c_in < cur->poids.dim[1]; ++c_in) {
          for (size_t u = 0; u < k; u++) {
            fscanf(f, "\n");
            for (size_t v = 0; v < k; v++) {
              float val;
              fscanf(f, "%f", &val);
              t_set(cur->m_p, c_out, c_in, u, v, val);
              t_set(cur->v_p, c_out, c_in, u, v, val);
            }
          }
          fscanf(f, "\n");
        }
        float val;
        fscanf(f, "%f", &val);
        t_set(cur->m_b, 0, 0, 0, c_out, val);
        t_set(cur->v_b, 0, 0, 0, c_out, val);
      }
    }
    if (cur->type == COUCHE_DENSE) {
      fscanf(f, "\n");
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        for (size_t j = 0; j < cur->poids.dim[3]; ++j) {
          float val;
          fscanf(f, "%f", &val);
          t_set(cur->m_p, 0, 0, i, j, val);
          t_set(cur->v_p, 0, 0, i, j, val);
        }
        fscanf(f, "\n");
      }
      for (size_t i = 0; i < cur->poids.dim[2]; ++i) {
        float val;
        fscanf(f, "%f", &val);
        t_set(cur->m_b, 0, 0, 0, i, val);
        t_set(cur->v_b, 0, 0, 0, i, val);
      }
    }
  }
  fclose(f);
  return r;
}

// fonctions de visualisation
void t_fprint(FILE *f, tenseur t) {
  fprintf(f, "Tenseur de dimensions [%zu, %zu, %zu, %zu]\n", t.dim[0], t.dim[1],
          t.dim[2], t.dim[3]);

  for (size_t i = 0; i < t.dim[0]; ++i) {
    for (size_t j = 0; j < t.dim[1]; ++j) {
      fprintf(f, "Canal %zu, Profondeur %zu:\n", i, j);
      for (size_t k = 0; k < t.dim[2]; ++k) {
        for (size_t l = 0; l < t.dim[3]; ++l) {
          fprintf(f, "%.2f ", t_get(t, i, j, k, l));
        }
        fputc('\n', f);
      }
      fputc('\n', f);
    }
  }
}

void ecrire_delta(reseau r, const char *nom_fichier) {
  FILE *f = fopen(nom_fichier, "w");
  if (!f) {
    fprintf(stderr, "ecrire_reseau_fichier : impossible d’ouvrir %s (%s)\n",
            nom_fichier, strerror(errno));
    return;
  }

  fprintf(f, "Réseau avec %zu couches, batch size %zu\n", r.nbr_couches,
          r.nbr_lots);

  for (size_t l = 0; l < r.nbr_couches; ++l) {
    fprintf(f, "\nCouche %zu:\n", l);
    fprintf(f, "  Activations (z):\n");
    t_fprint(f, r.delta[l]);
  }

  fclose(f);
}

void ecrire_reseau_fichier(reseau r, const char *nom_fichier) {
  FILE *f = fopen(nom_fichier, "w");
  if (!f) {
    fprintf(stderr, "ecrire_reseau_fichier : impossible d’ouvrir %s (%s)\n",
            nom_fichier, strerror(errno));
    return;
  }

  fprintf(f, "Réseau avec %zu couches, batch size %zu\n", r.nbr_couches,
          r.nbr_lots);

  for (size_t l = 0; l < r.nbr_couches; ++l) {
    couche *cur = &r.c[l];

    static const char *noms[] = {
        "ENTREE",  "CONV",   "TCONV",      "SE",       "APLAT",
        "DEAPLAT", "DENSE",  "ACT_RELU",   "ACT_SIGM", "ACT_TANH",
        "SOFTMAX", "SORTIE", "SORTIE_SIGM"};

    fprintf(f, "\nCouche %zu : %s\n", l, noms[cur->type]);

    fprintf(f, "  Activations (z):\n");
    t_fprint(f, cur->z);

    if (cur->type == COUCHE_CONV || cur->type == COUCHE_DENSE) {
      fprintf(f, "  Poids:\n");
      t_fprint(f, cur->poids);
      fprintf(f, "  Biais:\n");
      t_fprint(f, cur->biais);
    }
  }

  fclose(f);
}

void afficher_reseau(reseau r) {
  static const char *nom_type[] = {
      "ENTREE",  "CONV",   "TCONV",      "SE",       "APLAT",
      "DEAPLAT", "DENSE",  "ACT_RELU",   "ACT_SIGM", "ACT_TANH",
      "SOFTMAX", "SORTIE", "SORTIE_SIGM"};

  printf("\n%-3s | %-12s | %-16s | %-18s\n", "Id", "Type", "z (N,C,H,W)",
         "Poids/Biais");
  puts("---------------------------------------------------------------"
       "-----------------");

  for (size_t l = 0; l < r.nbr_couches; ++l) {
    couche *c = &r.c[l];

    /* Dimensions d’activation */
    char zbuf[32];
    snprintf(zbuf, sizeof zbuf, "(%zu,%zu,%zu,%zu)", c->z.dim[0], c->z.dim[1],
             c->z.dim[2], c->z.dim[3]);

    /* Dimensions des poids (ou « – » s’il n’y en a pas) */
    char pbuf[32] = "–";
    if (c->type == COUCHE_CONV || c->type == COUCHE_TCONV) {
      snprintf(pbuf, sizeof pbuf, "(%zu,%zu,%zu,%zu)", c->poids.dim[0],
               c->poids.dim[1], c->poids.dim[2], c->poids.dim[3]);
    } else if (c->type == COUCHE_DENSE) {
      snprintf(pbuf, sizeof pbuf, "(%zu,%zu,%zu,%zu)", c->poids.dim[0],
               c->poids.dim[1], c->poids.dim[2], c->poids.dim[3]);
    }

    printf("%-3zu | %-12s | %-16s | %-18s\n", l, nom_type[c->type], zbuf, pbuf);
  }
  putchar('\n');
}
