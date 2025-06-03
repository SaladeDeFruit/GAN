#include "tenseur.h"
#include <stdio.h>
#define MNIST_T "./MNIST/mnist_train.csv"
#define MNIST_T0 "./MNIST/mnist0.csv"
#define MNIST_T1 "./MNIST/mnist1.csv"
#define MNIST_T2 "./MNIST/mnist2.csv"
#define MNIST_T3 "./MNIST/mnist3.csv"
#define MNIST_T4 "./MNIST/mnist4.csv"
#define MNIST_T5 "./MNIST/mnist5.csv"
#define MNIST_T6 "./MNIST/mnist6.csv"
#define MNIST_T7 "./MNIST/mnist7.csv"
#define MNIST_T8 "./MNIST/mnist8.csv"
#define MNIST_T9 "./MNIST/mnist9.csv"
#define MNIST_TEST "./MNIST/mnist_test.csv"
#define TAILLE_IMAGE 28

int lire_ligne_mnist(FILE *f, float *pixel);

int lire_ligne_lettres(FILE *f, float *pixel);

void afficher_chiffre(float *pixel);
static void copier_image_vers_tenseur(tenseur z, float *pixels, size_t n);
