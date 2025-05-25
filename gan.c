#include "generate.h"
#include "mnist.c"
#include "reseau.h"
#include "tenseur.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// gcc ./gan.c ./reseau.c ./generate.c ./tenseur.c -o train -O3 -lm
// -march=native -ffast-math

/* --- Générateur --- */
#define BATCH_SIZE 32 /* taille du mini‑batch */
#define G_COUCHES 10
size_t G_nbr_couches = G_COUCHES;

TypeCouche Gtypes[G_COUCHES] = {COUCHE_ENTREE,  COUCHE_DENSE, COUCHE_ACT_RELU,
                                COUCHE_DEAPLAT, COUCHE_TCONV, COUCHE_ACT_RELU,
                                COUCHE_TCONV,   COUCHE_CONV,  COUCHE_ACT_TANH,
                                COUCHE_SORTIE};

/* 0 où il n’y a pas de poids */
size_t G_dim_poids[G_COUCHES] = {0, 128 * 6 * 6, 0, 0, 4, 0, 4, 3, 0, 0};

/* Bruit 100×1 */
size_t G_dim_entree[2] = {100, 1};

/* Canal (prof) de chaque couche */
size_t G_prof[G_COUCHES] = {1, 1, 1, 128, 64, 64, 32, 1, 1, 1};

/* --- Discriminateur --- */
#define D_COUCHES 11
size_t D_nbr_couches = D_COUCHES;

TypeCouche Dtypes[D_COUCHES] = {
    COUCHE_ENTREE,   COUCHE_CONV,  COUCHE_ACT_RELU,   COUCHE_CONV,
    COUCHE_ACT_RELU, COUCHE_SE,    COUCHE_APLAT,      COUCHE_DENSE,
    COUCHE_ACT_RELU, COUCHE_DENSE, COUCHE_SORTIE_SIGM};

/* kernel sizes pour chaque CONV ; 0 sinon */
size_t D_dim_poids[D_COUCHES] = {0, 5, 0, 5, 0, 0, 0, 256, 0, 1, 0};

/* entrée 28 × 28 */
size_t D_dim_entree[2] = {28, 28};

/* canaux (prof) — cf. tableau plus haut */
size_t D_prof[D_COUCHES] = {1, 64, 64, 128, 128, 128, 128, 1, 1, 1, 1};

/* -----------------------------------------------------------
 *  Affiche la topologie du réseau sous forme de petit tableau
 *  Exemple :
 *      Id | Type          |   z (N,C,H,W)   |   Poids (Cout,Cin,H,W)
 *      ------------------------------------------------------------
 *       0 | ENTREE        |  (64,1,100,1)
 *       1 | DENSE         |  (64,1,256,1)   |  (1,1,256,100)
 *       2 | ACT_RELU      |  (64,1,256,1)
 *       … etc.
 * ----------------------------------------------------------- */

#define TAILLE_IMAGE 28
#define NB_EPOCHS 2

int main() {
  /* --- Génerateur --- */
  srand((unsigned)time(NULL));
  reseau G = initialiser_reseau(G_nbr_couches, BATCH_SIZE, Gtypes, G_dim_poids,
                                G_dim_entree, G_prof);
  init_poids_biais(G);
  // reseau G = charger_reseau("generateur.cnn");
  // afficher_reseau(G);
  /* --- Discriminateur --- */
  reseau D = initialiser_reseau(D_nbr_couches, BATCH_SIZE, Dtypes, D_dim_poids,
                                D_dim_entree, D_prof);
  init_poids_biais(D);

  // reseau D = charger_reseau("detecteur.cnn");
  // afficher_reseau(D);

  const float ALPHA = 0.001f;
  const float BETA1 = 0.9f;
  const float BETA2 = 0.999f;

  //   1)  // ----- phase Discriminateur -----
  FILE *graphf = fopen("graph.txt", "w");
  if (!graphf) {
    perror("Ouverture graph.txt");
    return EXIT_FAILURE;
  }

  /* Pointeurs pour la rétro‑prop (structure : y_att[lot][classe]) */
  float **y_D = malloc(BATCH_SIZE * sizeof(y_D));
  float **y = malloc(BATCH_SIZE * sizeof(y));
  for (int i = 0; i < BATCH_SIZE; i++) {
    y_D[i] = calloc(1, sizeof(float));
    y[i] = calloc(1, sizeof(float));
  }

  /* Buffers pour images */
  float batch_pixels[TAILLE_IMAGE * TAILLE_IMAGE];

  /* ------------------- Entraînement ------------------------*/
  int saut_fichier = 1267;
  FILE *trainf = fopen(MNIST_T, "r");

  if (!trainf) {
    perror("Ouverture train");
    liberer_reseau(G);
    liberer_reseau(D);
    fclose(graphf);
    return EXIT_FAILURE;
  }

  for (int i = 0; i < saut_fichier; ++i) {
    lire_ligne_mnist(trainf, NULL);
  }

  for (size_t epoch = 0; epoch < NB_EPOCHS; ++epoch) {
    rewind(trainf);
    size_t batch_idx = 0, total = saut_fichier, b_total = 0, b_correct = 0;
    int label = 0;
    int t = 0;
    int t_gen = 1;
    int c = 0;

    while (label >= 0) {
      t++;

      if (batch_idx == BATCH_SIZE) {
        propagation_avant(D, -1);

        for (int n = 0; n < BATCH_SIZE; n++) {
          float val = t_get(D.c[D_nbr_couches - 1].z, n, 0, 0, 0);
          printf("%f - %f\n", val, y_D[n][0]);
          if (val >= 0.5 && fabs(y_D[n][0] - 0.9) <= 0.1 ||
              val <= 0.5 && fabs(y_D[n][0] - 0.1) <= 0.1) {
            b_correct += 1;
          }
        }
        printf("%zu - score D:%%%f\n", total, (float)b_correct / b_total * 100);
        b_total = 0;
        b_correct = 0;

        retropropagation(D, ALPHA, BETA1, BETA2, (float **)y_D, NULL);
        batch_idx = 0; /* nouveau mini‑batch */
        if (t >= t_gen * BATCH_SIZE) {
          c++;
          // printf("G\n");
          if (c % 10 == 0) {
            c = 0;
            sauver_reseau(D, "detecteur.cnn");
            sauver_reseau(G, "generateur.cnn");
            char prefix[25];
            snprintf(prefix, sizeof prefix, "./generation/%zu", total);

            generer_images("generateur.cnn", 5, prefix);
          }
          D.gel = true;
          t = 0;
          for (int k = 0; k < t_gen * 2; k++) {
            for (int n = 0; n < BATCH_SIZE; n++) {
              generer_bruit(G.c[0].z, n);
              y[n][0] = 0.9;
            }

            propagation_avant(G, -1);

            // on place l'image obtenue dans D:
            for (int n = 0; n < BATCH_SIZE; n++) {
              for (int i = 0; i < TAILLE_IMAGE; i++) {
                for (int j = 0; j < TAILLE_IMAGE; j++) {
                  float val = t_get(G.c[G_nbr_couches - 1].z, n, 0, i, j);
                  t_set(D.c[0].z, n, 0, i, j, val);
                }
              }
            }

            propagation_avant(D, -1);
            retropropagation(D, ALPHA, BETA1, BETA2, y, NULL);
            // on retropropague dans G avec le delta de D
            retropropagation(G, ALPHA, BETA1, BETA2, NULL, &D.delta[0]);
          }
          D.gel = false;
        }
      }
      int alea = (rand() % 2);
      if (alea == 1) {
        // cas vraie image
        label = lire_ligne_mnist(trainf, batch_pixels);
        if (label < 0) {
          break;
        }
        copier_image_vers_tenseur(D.c[0].z, batch_pixels, batch_idx);
        y_D[batch_idx][0] = 0.9;
        batch_idx++;
        total++;
        b_total++;
      } else {
        // cas fausse image
        generer_bruit(G.c[0].z, 0);
        propagation_avant(G, 1);
        for (int i = 0; i < TAILLE_IMAGE; i++) {
          for (int j = 0; j < TAILLE_IMAGE; j++) {
            float val = t_get(G.c[G_nbr_couches - 1].z, 0, 0, i, j);
            t_set(D.c[0].z, batch_idx, 0, i, j, val);
          }
        }
        y_D[batch_idx][0] = 0.1;
        batch_idx++;
        // total++;
        b_total++;
      }
    }
    printf("\nFin epoch %zu – %zu images traitées\n", epoch + 1, total);
  }
  fclose(trainf);

  for (int i = 0; i < BATCH_SIZE; i++) {
    free(y_D[i]);
    free(y[i]);
  }
  free(y_D);
  free(y);

  liberer_reseau(G);
  liberer_reseau(D);
  return 0;
}