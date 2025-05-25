#ifndef GENERATE_H
#define GENERATE_H

#include "reseau.h"
#include "tenseur.h"

void generer_bruit(tenseur t, int n);
static int save_png(tenseur img, size_t n, const char *path);

void generer_images(const char *modele, size_t nb_imgs, const char *prefix);


#endif