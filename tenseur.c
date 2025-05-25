#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *cont;   // contenu
  size_t dim[4]; // dimension
} tenseur;

tenseur t_init(size_t d1, size_t d2, size_t d3, size_t d4) {
  tenseur t = {.cont = calloc(d1 * d2 * d3 * d4, sizeof(float)),
               .dim = {d1, d2, d3, d4}};
  return t;
}

void t_liberer(tenseur t) { free(t.cont); }

void t_set(tenseur t, size_t x1, size_t x2, size_t x3, size_t x4, float val) {
  assert(x1 < t.dim[0] && x2 < t.dim[1] && x3 < t.dim[2] && x4 < t.dim[3]);
  t.cont[x1 * t.dim[1] * t.dim[2] * t.dim[3] + x2 * t.dim[2] * t.dim[3] +
         x3 * t.dim[3] + x4] = val;
}

float t_get(tenseur t, size_t x1, size_t x2, size_t x3, size_t x4) {
  assert(x1 < t.dim[0] && x2 < t.dim[1] && x3 < t.dim[2] && x4 < t.dim[3]);
  return t.cont[x1 * t.dim[1] * t.dim[2] * t.dim[3] + x2 * t.dim[2] * t.dim[3] +
                x3 * t.dim[3] + x4];
}

void t_copier(tenseur t_in, tenseur t_out) {
  assert(t_in.dim[0] == t_out.dim[0] && t_in.dim[1] == t_out.dim[1] &&
         t_in.dim[2] == t_out.dim[2] && t_in.dim[3] == t_out.dim[3]);
  for (int x1 = 0; x1 < t_in.dim[0]; x1++) {
    for (int x2 = 0; x2 < t_in.dim[1]; x2++) {
      for (int x3 = 0; x3 < t_in.dim[2]; x3++) {
        for (int x4 = 0; x4 < t_in.dim[3]; x4++) {
          float val = t_get(t_in, x1, x2, x3, x4);
          t_set(t_out, x1, x2, x3, x4, val);
        }
      }
    }
  }
}

void t_afficher(tenseur t) {
  printf("Tenseur de dimensions [%zu, %zu, %zu, %zu]\n", t.dim[0], t.dim[1],
         t.dim[2], t.dim[3]);
  for (size_t i = 0; i < t.dim[0]; i++) {
    for (size_t j = 0; j < t.dim[1]; j++) {
      printf("Canal %zu, Profondeur %zu:\n", i, j);
      for (size_t k = 0; k < t.dim[2]; k++) {
        for (size_t l = 0; l < t.dim[3]; l++) {
          printf("%.2f ", t.cont[i * t.dim[1] * t.dim[2] * t.dim[3] +
                                 j * t.dim[2] * t.dim[3] + k * t.dim[3] + l]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}