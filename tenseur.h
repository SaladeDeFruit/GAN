#ifndef TENSEUR_H
#define TENSEUR_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *cont;   // contenu
  size_t dim[4]; // dimension
} tenseur;

/**
 * Initialise un tenseur 4D avec les dimensions spécifiées
 * @param d1, d2, d3, d4 Les dimensions du tenseur
 * @return Un tenseur initialisé avec des zéros
 */
tenseur t_init(size_t d1, size_t d2, size_t d3, size_t d4);

/**
 * Libère la mémoire allouée pour un tenseur
 * @param t Le tenseur à libérer
 */
void t_liberer(tenseur t);

/**
 * Définit une valeur à une position donnée du tenseur
 * @param t Le tenseur
 * @param x1, x2, x3, x4 Les indices
 * @param val La valeur à définir
 */
void t_set(tenseur t, size_t x1, size_t x2, size_t x3, size_t x4, float val);

/**
 * Récupère une valeur à une position donnée du tenseur
 * @param t Le tenseur
 * @param x1, x2, x3, x4 Les indices
 * @return La valeur à la position spécifiée
 */
float t_get(tenseur t, size_t x1, size_t x2, size_t x3, size_t x4);

/**
 * Copie le contenu d'un tenseur vers un autre
 * @param t_in Le tenseur source
 * @param t_out Le tenseur destination (doit avoir les mêmes dimensions)
 */
void t_copier(tenseur t_in, tenseur t_out);

/**
 * Affiche le contenu du tenseur de manière formatée
 * @param t Le tenseur à afficher
 */
void t_afficher(tenseur t);

#endif // TENSEUR_H