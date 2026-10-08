Correction HMI dynamique — validation du 8 octobre 2026

Cette validation remplace les conclusions HMI_hourly/HMI_SYNC du premier
rapport `longitude_validation.md`. La convention des autres produits n'est
pas modifiée par cette correction.

Les mêmes FITS ont servi à toutes les comparaisons : les quatre cartes
`hmi.synoptic_hourly_20260831_222400.fits`, `20260831_232400`,
`20260901_002400`, `20260901_012400`, et
`hmi.mrdailysynframe_720s_nrt.20260901_002400_TAI.data.fits`.
Les métadonnées et mesures complètes figurent dans
[le résultat JSON](hmi_dynamic_longitude_validation.json).

Les deux produits ont `CONTENT='Update Synoptic MAP (Mr)'`, `CRPIX1=1800`
et `CDELT1=-0.1`. Leur CONTENT ne satisfait pas le sélecteur
`HMISynopticMap` de SunPy, qui attend « carrington synoptic chart ».
De plus, sa correction `abs(CDELT1)` est conditionnée par
`CUNIT2='Sine Latitude'`. Ici CUNIT2 vaut `deg` pour hourly et `sin(deg)`
pour SYNC. Voir le [code officiel SunPy](https://github.com/sunpy/sunpy/blob/main/sunpy/map/sources/sdo.py).

Le [producteur JSOC NRT](https://github.com/JSOC-SDP/proj/blob/main/mag/synop/apps/mrmlosdailysynframe_nrt.c)
et le [producteur définitif](https://github.com/JSOC-SDP/proj/blob/main/mag/synop/apps/mrmlosdailysynframe.c)
placent la longitude physique `clog0-hwd` dans la première colonne.
Ils écrivent `LON_LAST=360*crn-clog0+hwd`, et `LON_FRST` désigne
la dernière colonne. Ces nombres sont des temps Carrington.
La conversion physique correspondant aux colonnes stockées est donc :

```python
longitude[i] = -(CRVAL1 + (i + 1 - CRPIX1) * CDELT1) % 360
```

Cela inverse le signe de la référence et du pas, sans retourner Br.
La concordance des deux extrémités avec `-LON_LAST` et `-LON_FRST`
est vérifiée à 1e-6 degré (écart maximal SYNC : environ 2.07e-7 degré,
lié aux décimales différentes des mots-clés).
La page descriptive JSOC n'étant pas accessible pendant cet audit,
le code officiel de production et les en-têtes constituent les références.

| Fichier | Ancienne inversion de la séquence WCS | Hypothèse abs(CDELT1) seule | Longitude physique de Br[:,0] |
|---|---:|---:|---:|
| hourly 2026-08-31 22:24 | 88.800012° | 88.900012° | 271.299988° |
| SYNC 2026-09-01 00:24 | 89.899988° | 89.999988° | 270.200012° |

Une vérification indépendante utilise les structures du champ sur les lignes
200:1200 et colonnes 1500:3300 : le meilleur déplacement entre cartes horaires
est successivement +6, +5 et +6 colonnes. Les erreurs quadratiques moyennes
sont 0.00265, 0.00270 et 0.00280, contre environ 219–221 pour les déplacements
voisins. Cette évolution confirme le signe de la longitude physique.

Résultats :

- Toutes les colonnes natives des cinq fichiers correspondent exactement
  aux colonnes Br attendues, avec le traitement historique des NaN.
- À résolution native, le champ après rotation Stonyhurst est exactement
  identique au pipeline antérieur au refactoring, pour les cinq fichiers.
- Avec redimensionnement à 360×720, cette égalité exacte est obtenue pour
  hourly 00:24, hourly 01:24 et SYNC. Elle ne l'est pas pour hourly 22:24
  et 23:24 : écarts maximaux respectifs 838.14 et 987.62 dans les valeurs
  du champ, pour des différences de premiers centres physiques après rotation
  de -0.4° et -0.5°. Le nouveau pipeline utilise les centres physiques après
  normalisation et redimensionnement, tandis que l'ancien utilise une
  origine arrondie et un axe commençant à zéro. L'égalité historique n'est
  donc pas un critère validé dans ces deux cas. La géométrie redimensionnée
  et son Br sont en revanche contrôlés contre une construction indépendante
  dans `validate_magnetogram_longitude.py`.
- Le stencil réel des quatre cartes horaires passe en modes historique et
  Carrington, avec et sans redimensionnement. Le champ interpolé et sa
  variante linéaire sont finis ; la rotation conserve leurs valeurs.
- Les différences de grille inférieures à 0.001 pixel sont acceptées
  sans interpolation spatiale supplémentaire, après alignement périodique
  des colonnes. Le décalage de 0.000024° = 0.00024 pixel ne bloque plus.
  Les coordonnées du premier fichier servent de référence au stencil.
- 177 tests automatisés passent. L'audit réel général donne 78 cas PASS,
  aucun FAIL, sur les coordonnées et champs natifs/redimensionnés ; il
  contrôle aussi les aires et flux avant/après rotation. Ce dernier audit
  ne relance pas les filtres numériques ni les téléchargements de stencils.

Reproduction : `tests/validate_hmi_dynamic_longitude.py` accepte les chemins
`--cache`, `--sync`, `--baseline` (anciens readers.py et longitude.py) et
`--output`. Aucun FITS original n'est modifié. Le premier rapport est conservé
comme historique ; ses validations HMI basées sur l'ancienne interprétation
du WCS et son rejet du stencil hourly sont obsolètes.
