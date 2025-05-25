#include "reseau.h"
#include "tenseur.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external/stb_image_write.h"

void generer_bruit(tenseur t, int n) {
  for (size_t p = 0; p < t.dim[1]; ++p) {
    for (size_t i = 0; i < t.dim[2]; ++i) {
      for (size_t j = 0; j < t.dim[3]; ++j) {
        float val = 2.f * ((float)rand() / RAND_MAX) - 1.f; /* U(-1,1) */
        t_set(t, n, p, i, j, val);
      }
    }
  }
}

/* PGM ASCII très simple : P2 WIDTH HEIGHT 255 puis les niveaux 0-255 */
static int save_png(tenseur img, size_t n, const char *path) {
  const int H = (int)img.dim[2], W = (int)img.dim[3];
  unsigned char *buf = malloc(W * H);
  for (int i = 0; i < H; ++i)
    for (int j = 0; j < W; ++j) {
      float p = t_get(img, n, 0, i, j);
      if (p < -1)
        p = -1;
      if (p > 1)
        p = 1;
      buf[i * W + j] = (unsigned char)(p * 127.5f + 127.5f);
    }
  int ok = stbi_write_png(path, W, H, 1, buf, W);
  free(buf);
  return ok ? 0 : -1;
}

/* ----------------------------------------------------------- */
/*  Génère nb_imgs images avec le modèle sauvegardé            */
/*  chaque fichier est prefix_0000.pgm, prefix_0001.pgm …      */
/* ----------------------------------------------------------- */
void generer_images(const char *modele, size_t nb_imgs, const char *prefix) {
  /* 1. Charge le réseau générateur */
  reseau G = charger_reseau(modele);
  if (!G.c) {
    fprintf(stderr, "Impossible de charger %s\n", modele);
    return;
  }

  /* 2. Graines aléatoires pour le bruit */
  srand((unsigned)time(NULL));

  size_t out_layer = G.nbr_couches - 1; /* couche image */
  tenseur out = G.c[out_layer].z;       /* (N,1,28,28)  */

  /* 3. Boucle de génération */
  for (size_t k = 0; k < nb_imgs; ++k) {

    /* a. Met un vecteur bruit dans l’index 0 uniquement */
    generer_bruit(G.c[0].z, 0);

    /* b. Propagation avant pour n = 0 uniquement        */
    propagation_avant(G, 1);

    /* c. Sauvegarde en PGM                              */
    char filename[256];
    snprintf(filename, sizeof filename, "%s_%04zu.png", prefix, k);
    if (save_png(out, 0, filename) == 0)
      printf("Image %zu sauvegardée dans %s\n", k, filename);
  }

  liberer_reseau(G);
}

// int main() {
//   generer_images("generateur.cnn", 10, "./generation/img");
//   return 0;
// }