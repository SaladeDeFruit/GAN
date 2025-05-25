#include "mnist.h"
#include "tenseur.h"
#include <stdio.h>

int lire_ligne_mnist(FILE *f, float *pixel) {
  if (feof(f)) {
    return -1;
  }
  int label;
  fscanf(f, "%d,", &label);

  for (int i = 0; i < TAILLE_IMAGE * TAILLE_IMAGE; ++i) {
    int p;
    char c;
    fscanf(f, "%d%c", &p, &c);
    if (pixel != NULL) {
      pixel[i] = ((float)p) / 127.5f - 1;
    }
  }
  return label;
}

void afficher_chiffre(float *pixel) {
  printf("\n");
  for (int i = 0; i < TAILLE_IMAGE; ++i) {
    for (int j = 0; j < TAILLE_IMAGE; ++j) {
      printf("%c", pixel[i * TAILLE_IMAGE + j] > 0.5 ? '#' : ' ');
    }
    printf("\n");
  }
  printf("\n");
}

static void copier_image_vers_tenseur(tenseur z, float *pixels, size_t n) {
  for (size_t i = 0; i < TAILLE_IMAGE; ++i)
    for (size_t j = 0; j < TAILLE_IMAGE; ++j)
      t_set(z, n, 0, i, j, pixels[i * TAILLE_IMAGE + j]);
}