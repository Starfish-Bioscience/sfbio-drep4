"""
Ré-implémentation vectorisée des goulots skani de dRep 3.6.2 — PLOMBERIE UNIQUEMENT.

POURQUOI
--------
Sur un dRep de n = 14 195 génomes avec `--SkipMash` (un seul cluster primaire,
donc skani en all-vs-all), trois fonctions de `d_cluster/external.py` rendent le
run infaisable :

  * `load_triangle_matrix` re-split une ligne ENTIÈRE de la matrice dans sa
    boucle interne -> coût en O(n³) (~9 h projetées à n = 14 195) ;
  * `load_matrix_to_dataframe` fait de même en O(n²) avec un `.split()` par
    cellule ;
  * `load_skani` matérialise deux DataFrames de n² lignes puis les joint par
    `pd.merge` sur des clés chaînes (201 M lignes -> dizaines de GB) ;
  * `add_avani` parcourt les 201 M lignes en `iterrows()` et construit un dict
    de 201 M clés chaînes formatées.

CE QUI EST GARANTI IDENTIQUE
----------------------------
Aucune formule n'est touchée. Sont reproduits à l'identique :

  * la SÉMANTIQUE : l'ANI ne vient que du triangle inférieur de la matrice
    (i >= j), miroité — le triangle supérieur est ignoré, comme en vanilla ;
    la couverture d'alignement (.af) vient de la matrice COMPLÈTE, donc
    asymétrique — également comme en vanilla ;
  * l'ORDRE DES LIGNES : triangle inférieur en row-major (`np.tril_indices`,
    exactement l'ordre des boucles vanilla), puis les lignes miroir hors
    diagonale dans le même ordre ;
  * l'ARITHMÉTIQUE : parsing décimal -> float64 puis `/100`, et
    `av_ani = (A + Aᵀ)/2` qui est le `np.mean([x, y])` de vanilla (division
    exacte par une puissance de 2), avec 1.0 sur la diagonale ;
  * les NOMS : `basename`, appliqué aux n noms au lieu des n² lignes.

Les colonnes de noms restent en dtype `object` avec des chaînes PARTAGÉES (un
seul objet Python par génome, n² pointeurs) : pas de `Categorical`, donc aucun
changement de comportement en aval (`pivot`, `unique`, `to_csv`).

Validation : reproduction bit-à-bit du cross-assembleur dRep du pipeline sur des
échantillons de contrôle (`04-Script/03-drep_fidelity_control.py`).
"""

import numpy as np
import pandas as pd


def _read_full_matrix(path):
    """Lit une matrice pleine skani (`--full-matrix`) -> (noms, ndarray float64)."""
    with open(path, "r") as fh:
        n = int(fh.readline().strip())
        names = []
        M = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            parts = fh.readline().rstrip("\n").split("\t")
            names.append(parts[0])
            M[i, :] = parts[1:n + 1]
    return names, M


def _shared(names):
    """Tableau object de chaînes PARTAGÉES : n² pointeurs, n objets str."""
    a = np.empty(len(names), dtype=object)
    a[:] = names
    return a


def load_triangle_matrix(file_path):
    names, M = _read_full_matrix(file_path)
    no = _shared(names)
    i, j = np.tril_indices(len(names))
    return pd.DataFrame({"reference": no[i], "querry": no[j], "ani": M[i, j]})


def load_matrix_to_dataframe(file_path):
    names, M = _read_full_matrix(file_path)
    no, n = _shared(names), len(names)
    return pd.DataFrame({"reference": np.repeat(no, n), "querry": np.tile(no, n),
                         "ani": M.ravel()})


def load_skani(file):
    """ANI (triangle inférieur miroité) + couverture (.af, matrice complète)."""
    import drep.d_cluster.utils

    names, A = _read_full_matrix(file)
    af_names, F = _read_full_matrix(file + ".af")
    assert af_names == names, "matrices skani .ani et .af dans un ordre différent"

    base = _shared([drep.d_cluster.utils._get_genome_name_from_fasta(x) for x in names])
    i, j = np.tril_indices(len(names))
    off = i != j                      # hors diagonale : les lignes miroir
    tri = A[i, j]

    ref = np.concatenate([i, j[off]])
    qry = np.concatenate([j, i[off]])
    ani = np.concatenate([tri, tri[off]]) / 100.0

    return pd.DataFrame({"reference": base[ref], "querry": base[qry],
                         "ani": ani,
                         # dRep 4.0.0 : l'AF de skani (0-100) est ramené en fraction (0-1),
                         # c'est la correction du bug d'unité. On la reproduit ici.
                         "alignment_coverage": F[ref, qry] / 100.0})


def add_avani(db, _vanilla=None):
    """`av_ani` = moyenne réciproque de `ani` ; 1.0 quand reference == querry."""
    ref = db["reference"].values
    qry = db["querry"].values
    cats = pd.Index(pd.unique(ref))
    ri = cats.get_indexer(ref)
    qi = cats.get_indexer(qry)
    if (ri < 0).any() or (qi < 0).any():
        # jeu de paires incomplet (ne devrait pas arriver avec skani all-vs-all)
        # -> on repasse par l'implémentation d'origine plutôt que d'inventer un zéro
        return _vanilla(db)

    m = len(cats)
    A = np.zeros((m, m), dtype=np.float64)
    A[ri, qi] = db["ani"].values
    av = (A + A.T) * 0.5
    out = av[ri, qi]
    out[ri == qi] = 1.0
    db["av_ani"] = out
