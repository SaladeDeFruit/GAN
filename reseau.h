#ifndef RESEAU_H
#define RESEAU_H

#define EPS 1e-8f

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tenseur.h"

/**
 * @enum TypeCouche
 * @brief Types de couches dans un réseau de neurones.
 */
typedef enum {
    COUCHE_ENTREE,      // Couche d'entrée
    COUCHE_CONV,        // Couche convolutionnelle
    COUCHE_TCONV,       // Couche transposée convolutionnelle
    COUCHE_SE,          // Couche de sous-échantillonnage (max pooling)
    COUCHE_APLAT,       // Couche d'aplatissement
    COUCHE_DEAPLAT,     // Couche de désaplatissement
    COUCHE_DENSE,       // Couche dense (entièrement connectée)
    COUCHE_ACT_RELU,    // Couche d'activation ReLU
    COUCHE_ACT_SIGM,    // Couche d'activation sigmoïde
    COUCHE_ACT_TANH,    // Couche d'activation Tanh
    COUCHE_SOFTMAX,     // Couche Softmax
    COUCHE_SORTIE,      // Couche de sortie
    COUCHE_SORTIE_SIGM  // Couche de sortie sigmoïde
} TypeCouche;

/**
 * @struct Couche
 * @brief Structure représentant une couche du réseau avec ses activations,
 *        poids et biais.
 */
typedef struct Couche {
    TypeCouche type;    // Type de la couche
    tenseur z;          // Tenseur des activations
    tenseur poids;      // Tenseur des poids (si applicable)
    tenseur biais;      // Tenseur des biais (si applicable)
    tenseur m_p;        // Tenseur de moment des poids (Adam optimizer)
    tenseur v_p;        // Tenseur de vitesse des poids (Adam optimizer)
    tenseur m_b;        // Tenseur de moment des biais (Adam optimizer)
    tenseur v_b;        // Tenseur de vitesse des biais (Adam optimizer)
} couche;

/**
 * @struct reseau
 * @brief Structure représentant un réseau de neurones avec plusieurs couches.
 */
typedef struct {
    size_t nbr_couches;     // Nombre de couches
    size_t nbr_lots;        // Taille du lot (batch size)
    int *t;                 // Tableau des threads
    couche *c;              // Tableau de couches
    pthread_t *threads;     // Threads pour parallélisation
    tenseur *delta;         // Tenseurs de gradients
    bool gel;               // Bloque l'apprentissage si true
} reseau;

/* ========================================================================== */
/* FONCTIONS DE GESTION DU RÉSEAU                                            */
/* ========================================================================== */

/**
 * @brief Initialise un réseau de neurones avec les types de couches et
 *        dimensions spécifiés.
 * @param nbr_couches Nombre de couches.
 * @param nbr_lots Taille du lot.
 * @param types Tableau des types de couches.
 * @param dim_poids Tableau des dimensions des poids (taille noyau pour CONV,
 *                  neurones pour DENSE).
 * @param dim_entree Dimensions de l'entrée [hauteur, largeur].
 * @param prof Tableau des profondeurs (canaux) pour chaque couche.
 * @return Réseau initialisé.
 */
reseau initialiser_reseau(size_t nbr_couches, size_t nbr_lots,
                          TypeCouche *types, size_t *dim_poids,
                          size_t dim_entree[2], size_t *prof);

/**
 * @brief Libère la mémoire allouée pour un réseau.
 * @param r Réseau à libérer.
 */
void liberer_reseau(reseau r);

/**
 * @brief Affiche les informations du réseau.
 * @param r Réseau à afficher.
 */
void afficher_reseau(reseau r);

/* ========================================================================== */
/* FONCTIONS D'ACTIVATION                                                     */
/* ========================================================================== */

/**
 * @brief Fonction d'activation ReLU avec fuite (Leaky ReLU).
 * @param x Valeur d'entrée.
 * @return Valeur après application de Leaky ReLU.
 */
float f_act_leaky_relu(float x);

/**
 * @brief Dérivée de la fonction d'activation ReLU avec fuite.
 * @param x Valeur d'entrée.
 * @return Dérivée de Leaky ReLU en x.
 */
float df_act_leaky_relu(float x);

/* ========================================================================== */
/* FONCTIONS DE PROPAGATION                                                   */
/* ========================================================================== */

/**
 * @brief Fonction auxiliaire pour la propagation avant (thread).
 * @param param Paramètres pour le thread.
 * @return NULL.
 */
void *propagation_avant_aux(void *param);

/**
 * @brief Effectue la propagation avant dans le réseau.
 * @param r Réseau à utiliser.
 * @param nmb_cmp Nombre de composants à traiter.
 */
void propagation_avant(reseau r, int nmb_cmp);

/**
 * @brief Fonction auxiliaire pour la rétropropagation (thread).
 * @param arg Arguments pour le thread.
 * @return NULL.
 */
void *retropropagation_aux(void *arg);

/**
 * @brief Effectue la rétropropagation pour calculer les gradients et mettre à
 *        jour les poids et biais.
 * @param r Réseau à entraîner.
 * @param alpha Taux d'apprentissage.
 * @param beta1 Paramètre beta1 pour Adam optimizer.
 * @param beta2 Paramètre beta2 pour Adam optimizer.
 * @param y_attendu Tableau des sorties attendues pour chaque lot.
 * @param delta Tableau de tenseurs d'erreurs (gradients).
 */
void retropropagation(reseau r, float alpha, float beta1, float beta2,
                      float **y_attendu, tenseur *delta);

/* ========================================================================== */
/* FONCTIONS DE GESTION DES POIDS ET BIAIS                                   */
/* ========================================================================== */

/**
 * @brief Génère une valeur aléatoire pour l'initialisation des poids/biais.
 * @param x Amplitude maximale.
 * @return Valeur aléatoire dans [-x, x].
 */
float distrib_alea(float x);

/**
 * @brief Initialise les poids et biais du réseau avec des valeurs aléatoires.
 * @param r Réseau à initialiser.
 */
void init_poids_biais(reseau r);

/**
 * @brief Met à jour les poids et biais du réseau en utilisant l'optimiseur Adam.
 * @param r Réseau à modifier.
 * @param alpha Taux d'apprentissage.
 * @param beta1 Paramètre de décroissance exponentielle pour le premier moment.
 * @param beta2 Paramètre de décroissance exponentielle pour le second moment.
 */
void mettre_a_jour_poids_biais_moments(reseau r, float alpha, float beta1,
                                       float beta2);

/* ========================================================================== */
/* FONCTIONS DE GESTION DES ERREURS/GRADIENTS                                */
/* ========================================================================== */

/**
 * @brief Initialise un tableau de tenseurs pour stocker les erreurs
 *        (gradients).
 * @param r Réseau concerné.
 * @return Pointeur vers le tableau de tenseurs d'erreurs.
 */
tenseur *init_erreur(reseau r);

/**
 * @brief Libère la mémoire allouée pour les tenseurs d'erreurs.
 * @param r Réseau concerné.
 * @param delta Tableau de tenseurs d'erreurs.
 */
void liberer_erreur(reseau r, tenseur *delta);

/* ========================================================================== */
/* FONCTIONS D'ENTRÉE/SORTIE                                                  */
/* ========================================================================== */

/**
 * @brief Affiche un tenseur dans un fichier.
 * @param f Fichier de sortie.
 * @param t Tenseur à afficher.
 */
void t_fprint(FILE *f, tenseur t);

/**
 * @brief Écrit le réseau dans un fichier.
 * @param r Réseau à écrire.
 * @param nom_fichier Nom du fichier de destination.
 */
void ecrire_reseau_fichier(reseau r, const char *nom_fichier);

/**
 * @brief Écrit les gradients (delta) dans un fichier.
 * @param r Réseau concerné.
 * @param nom_fichier Nom du fichier de destination.
 */
void ecrire_delta(reseau r, const char *nom_fichier);

/**
 * @brief Sauvegarde le réseau dans un fichier.
 * @param r Réseau à sauvegarder.
 * @param nom_fichier Nom du fichier de destination.
 */
void sauver_reseau(reseau r, const char *nom_fichier);

/**
 * @brief Charge un réseau depuis un fichier.
 * @param nom_fichier Nom du fichier source.
 * @return Réseau chargé.
 */
reseau charger_reseau(const char *nom_fichier);

#endif // RESEAU_H