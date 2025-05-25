#include "tenseur.h"
#include <stdio.h>
#define MNIST_T "./MNIST/mnist_train.csv"
#define MNIST_TEST "./MNIST/mnist_test.csv"
#define TAILLE_IMAGE 28

int lire_ligne_mnist(FILE *f, float *pixel);

int lire_ligne_lettres(FILE *f, float *pixel);

void afficher_chiffre(float *pixel);
static void copier_image_vers_tenseur(tenseur z, float *pixels, size_t n);