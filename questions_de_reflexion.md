# Module 1: Configuration du Workflow GitHub Actions
## Questions theoriques de reflexion
### Question 1

**Quelle est l'utilité de `enable-cache: true` ? Quel est son impact direct sur l'empreinte carbone et le coût computationnel ?**

`enable-cache: true` active le mécanisme de cache d'`uv` dans GitHub Actions. Lors des exécutions suivantes du pipeline, les dépendances déjà téléchargées sont réutilisées au lieu d'être téléchargées et réinstallées à chaque fois.

Les principaux avantages sont :

- réduction du temps d'exécution du pipeline ;
- diminution du trafic réseau ;
- réduction de la charge des serveurs hébergeant les paquets ;
- accélération des validations de code.

Sur le plan environnemental, cette optimisation diminue la consommation d'énergie liée aux téléchargements et aux installations répétitives. Elle contribue ainsi à réduire l'empreinte carbone de l'infrastructure de recherche tout en diminuant les coûts computationnels associés aux exécutions fréquentes des pipelines CI/CD.



### Question 2: Déploiement Continu(CD) et Gestion des Artéfacts

**Pourquoi utiliser `mypy --strict` avant de lancer des simulations longues ?**

L'option `--strict` de MyPy effectue une vérification statique approfondie des types avant l'exécution du programme.

Cette étape permet de détecter précocement des erreurs telles que :

- incompatibilités entre types numériques ;
- mauvaise manipulation de tableaux NumPy ou tenseurs PyTorch ;
- valeurs pouvant être `None` alors qu'elles sont supposées contenir des données ;
- erreurs dans les signatures des fonctions.

# Module 2: Matrice de tests multi-environnements (Matrix Build)
### Exercice 2.2
```
strategy:
      matrix:
        python-version:
          - "3.10"
          - "3.11"
          - "3.12"
```
to use it 
```
- name: Set up Python
        run: uv python install ${{ matrix.python-version }}
```

# Module 4: Maîtrise de NumPy et Ingestion de Données
### C-contiguous (Row-major)
- Les éléments d'une même ligne sont stockés de manière contiguë en mémoire.
- Les parcours ligne par ligne sont très rapides.
- Les réductions suivant les lignes (`axis=1`) bénéficient d'une excellente localité mémoire et d'une meilleure utilisation du cache CPU.

### F-contiguous (Column-major)
- Les éléments d'une même colonne sont contigus.
- Les parcours colonne par colonne sont plus efficaces.
- Les réductions suivant les colonnes (`axis=0`) exploitent mieux le cache processeur.

Le choix du layout mémoire influence donc directement les performances des calculs numériques intensifs.

### Vue vs Copie
Une opération de slicing comme
```
view = X[:, :10]
```
ne crée pas un nouveau tableau : elle renvoie une **vue** partageant la même mémoire que le tableau d'origine.
En revanche,
```
copy = X[:, :10].copy()
```
alloue une nouvelle zone mémoire indépendante.
La propriété `.base` permet de distinguer ces deux cas :

- `view.base is X` → il s'agit d'une vue ;
- `copy.base is None` → il s'agit d'une copie.

### Pourquoi utiliser l'API Lazy de Polars ?
L'API Lazy ne charge pas immédiatement les données en mémoire. Elle construit un plan d'exécution optimisé qui n'est évalué qu'au moment de l'appel à `collect()`. Cette approche permet de filtrer et sélectionner uniquement les colonnes et lignes nécessaires avant le chargement effectif, réduisant ainsi l'utilisation de la RAM et améliorant les performances sur de très grands fichiers CSV ou Parquet.

### Pourquoi éviter les boucles `for` ?
Les boucles Python interprétées sont relativement lentes pour les calculs numériques. En utilisant le **broadcasting** de NumPy, les opérations sont effectuées directement dans des routines compilées en C, ce qui permet de traiter simultanément l'ensemble des éléments des tableaux sans boucle explicite. Cette vectorisation réduit considérablement le temps d'exécution et exploite efficacement les optimisations matérielles (cache CPU et instructions SIMD).


# Module 5: Stabilité, Conditionnement et Analyse des Erreurs
### 1. Nombre de conditionnement

Le nombre de conditionnement
$κ(A)=∥A∥⋅∥A−1∥$
mesure la sensibilité de la solution d'un système linéaire aux erreurs de calcul ou aux perturbations des données. Une matrice de Hilbert est notoirement mal conditionnée : lorsque sa dimension augmente, son nombre de conditionnement croît très rapidement, ce qui rend la résolution numériquement instable.

### 2. Comparaison des précisions
En comparant `float16`, `float32` et `float64`, on observe généralement que :

- **float16** présente les erreurs de reconstruction les plus importantes en raison de son faible nombre de bits pour la mantisse ;
- **float32** offre un compromis entre précision et performances ;
- **float64** fournit les résultats les plus précis et est souvent privilégié pour les simulations scientifiques.

À mesure que la taille de la matrice augmente, les erreurs deviennent plus marquées, notamment pour les matrices mal conditionnées.

### 3. Propagation des erreurs
Une faible perturbation du second membre, par exemple de l'ordre de $10^{−7}$, peut entraîner une variation beaucoup plus importante de la solution lorsque la matrice est mal conditionnée. Cette amplification est approximativement bornée par :
$$
\frac{∥x∥}{∥δx∥}​≤κ(A)\frac{∥b∥}{∥δb∥​}
$$

Ainsi, plus le nombre de conditionnement est élevé, plus les erreurs d'entrée sont amplifiées dans la solution.

### 4. Pourquoi ne pas utiliser `==` ?

Les nombres à virgule flottante sont représentés de manière approximative selon la norme *IEEE 754*. Des erreurs d'arrondi apparaissent lors des opérations arithmétiques, ce qui rend les comparaisons exactes (`==`) peu fiables.

Il est donc préférable d'utiliser `np.isclose()` ou `np.allclose()`, qui considèrent deux valeurs comme égales si leur différence reste inférieure à des tolérances absolue (`atol`) et relative (`rtol`) adaptées à la précision utilisée. Cette approche permet une validation robuste des résultats numériques.

# Module 6
### Diagnostic de Profiling

Example of a profiling report to include:

|Function|Time (%)|
|---|---|
|`local_filter()`|98 %|
|inner nested loops|96 %|
|memory allocation|2 %|

The profiling performed with **cProfile** identifies the nested loops implementing the local filtering operator as the main performance bottleneck. The measurements obtained with `timeit` confirm that the majority of the execution time is spent repeatedly traversing the two-dimensional grid.

### Analyse des performances

Applying

```
@njit(parallel=True, fastmath=True)
```

provides three optimizations:

- **JIT compilation** converts Python code into optimized native machine code.
- **parallel=True** distributes the outer loop across multiple CPU cores using `prange`.
- **fastmath=True** allows additional compiler optimizations by relaxing certain floating-point constraints.

In practice, these optimizations can reduce execution time by a factor ranging from **5× to 30×**, depending on the processor, the size of the grid, and the number of available cores.

### Risques liés à `fastmath=True`

The `fastmath=True` option allows the compiler to reorder floating-point operations and apply aggressive mathematical optimizations. While this generally improves performance, it may alter numerical results because it no longer strictly enforces the IEEE 754 standard.

Potential consequences include:

- changes in the order of floating-point operations;
- different rounding behavior;
- reduced reproducibility across platforms;
- slight differences in numerical results due to non-associativity of floating-point arithmetic.

Therefore, `fastmath=True` is appropriate when small numerical deviations are acceptable in exchange for higher performance, but it should be used cautiously in applications requiring strict numerical reproducibility.

### Parallélisation de haut niveau

The parameter sweep evaluates 100 combinations of the coefficients ccc and ν\nuν. Since each simulation is independent, the workload is naturally parallelizable. The implementation with **Joblib** distributes the simulations across all available CPU cores (`n_jobs=-1`), reducing the overall execution time.

Increasing the number of workers generally decreases the execution time until the available hardware resources are fully utilized. Beyond that point, the performance gains diminish due to scheduling overhead and competition for shared resources such as memory bandwidth.

# Module 7
## Architecture et Différentiation Automatique sous PyTorch / JAX
### 1. Architecture du MLP

Le réseau implémenté est un perceptron multicouche (MLP) constitué d'une couche d'entrée recevant les variables physiques $(x,t)$, de plusieurs couches cachées utilisant la fonction d'activation `Tanh`, et d'une couche de sortie produisant l'approximation u^(x,t). Les fonctions d'activation lisses sont particulièrement adaptées aux PINNs, car elles facilitent le calcul des dérivées d'ordre élevé par différentiation automatique.

### 2. Perte physique

La perte physique est calculée grâce au moteur de différentiation automatique (`torch.autograd.grad`). Celui-ci permet d'obtenir successivement les dérivées $$
∂\hat{u}/∂t,\space \space ∂\hat{u}/∂x \space \space et \space \space ∂^{2}\hat{u}/∂x^{2}
$$
 qui sont ensuite injectées dans le résidu de l'équation d'advection-diffusion. La perte correspond à la moyenne du carré de ce résidu sur l'ensemble des points de collocation.

### 3. Perte totale

La fonction de coût finale est définie par :

$$
L = L_{\text{physics}} + L_{\text{boundary}}
$$

où $L_{\text{physics}}$ ​ impose le respect de l'équation différentielle et $L_{\text{boundary}}$  garantit le respect des conditions aux limites (ou des données expérimentales). Cette combinaison permet au réseau d'apprendre une solution qui satisfait simultanément les contraintes physiques et les observations disponibles.

---

## Exécution Multi-GPU

La fonction `get_device()` détecte automatiquement le meilleur accélérateur disponible :

- **CUDA** pour les GPU NVIDIA ;
- **MPS** pour les processeurs Apple Silicon ;
- **CPU** si aucun accélérateur matériel n'est disponible.

Le modèle et les tenseurs sont ensuite transférés vers ce périphérique à l'aide de `model.to(device)` et `tensor.to(device)`.

Pour exploiter plusieurs GPU, PyTorch propose notamment `torch.nn.DataParallel` ou, de manière plus performante et évolutive, `DistributedDataParallel` (DDP), qui répartit les données entre plusieurs GPU et synchronise les gradients à chaque itération. Pour des simulations scientifiques de grande taille, ces approches peuvent être complétées par des bibliothèques HPC telles que PETSc afin de distribuer les calculs sur plusieurs nœuds d'un cluster et d'améliorer le passage à l'échelle.
