# ============================================================
#  ANALYSE DES DONNÉES ÉTUDIANTS — Maths & Portugais
#  Source : UCI Student Performance Dataset
#  Merge : 382 élèves communs aux deux matières
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# ── 0. Chargement des données ────────────────────────────────
d1 = pd.read_csv(r"1- SY09\Projet\SY09\data\student-mat.csv", sep=",")
d2 = pd.read_csv(r"1- SY09\Projet\SY09\data\student-por.csv", sep=",")

cles_merge = ["school", "sex", "age", "address", "famsize", "Pstatus",
              "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

d3 = pd.merge(d1, d2, on=cles_merge, suffixes=(".x", ".y"))

print(f"Dimensions d3 : {d3.shape[0]} lignes x {d3.shape[1]} colonnes")
# Suffixe .x = Maths | .y = Portugais

"""
# ── 1. STATISTIQUES DESCRIPTIVES ────────────────────────────
print("\n=== STATISTIQUES DESCRIPTIVES ===")
cols_desc = ["age", "G1.x", "G2.x", "G3.x", "G1.y", "G2.y", "G3.y",
             "absences.x", "absences.y", "studytime.x", "studytime.y"]
print(d3[cols_desc].describe())

# Distribution du sexe et de l'école
print(d3["sex"].value_counts())
print(d3["school"].value_counts())
print(d3["sex"].value_counts(normalize=True) * 100)


# ── 2. ANALYSE DES NOTES FINALES ────────────────────────────

# Histogramme des notes finales
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(d3["G3.x"], bins=10, color="#4E79A7", edgecolor="white")
axes[0].axvline(d3["G3.x"].mean(), color="red", linewidth=2, linestyle="--")
axes[0].set_title("Notes finales – Mathématiques")
axes[0].set_xlabel("Note (0–20)")
axes[0].set_ylabel("Nombre d'élèves")
axes[0].legend([f"Moyenne = {d3['G3.x'].mean():.1f}"], loc="upper right")

axes[1].hist(d3["G3.y"], bins=10, color="#F28E2B", edgecolor="white")
axes[1].axvline(d3["G3.y"].mean(), color="red", linewidth=2, linestyle="--")
axes[1].set_title("Notes finales – Portugais")
axes[1].set_xlabel("Note (0–20)")
axes[1].set_ylabel("Nombre d'élèves")
axes[1].legend([f"Moyenne = {d3['G3.y'].mean():.1f}"], loc="upper right")

plt.tight_layout()
#plt.show()

# Corrélation maths vs portugais
print("\nCorrélation notes finales Maths/Portugais :")
corr_result = stats.pearsonr(d3["G3.x"], d3["G3.y"])
print(f"r = {corr_result[0]:.4f}, p-value = {corr_result[1]:.4e}")

colors = d3["sex"].map({"F": "#E15759", "M": "#4E79A7"})
plt.figure(figsize=(7, 6))
plt.scatter(d3["G3.x"], d3["G3.y"], c=colors, alpha=0.6)
# Droite de régression
m, b = np.polyfit(d3["G3.x"], d3["G3.y"], 1)
x_line = np.linspace(d3["G3.x"].min(), d3["G3.x"].max(), 100)
plt.plot(x_line, m * x_line + b, color="black", linewidth=2)
plt.title("Corrélation notes : Maths vs Portugais")
plt.xlabel("Note finale Maths (G3)")
plt.ylabel("Note finale Portugais (G3)")
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#E15759', label='Filles'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#4E79A7', label='Garçons')]
plt.legend(handles=legend_elements, loc="upper left")
plt.tight_layout()
#plt.show()


# ── 3. EFFET DU SEXE SUR LES NOTES ─────────────────────────
print("\n=== TEST SEXE vs NOTES ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

groups_sex = [d3[d3["sex"] == s]["G3.x"].values for s in ["F", "M"]]
axes[0].boxplot(groups_sex, labels=["F", "M"],
                patch_artist=True,
                boxprops=dict(facecolor="#E15759"),
                medianprops=dict(color="grey"))
# Colorer F et M différemment
bp = axes[0].boxplot(groups_sex, labels=["F", "M"], patch_artist=True)
bp["boxes"][0].set_facecolor("#E15759")
bp["boxes"][1].set_facecolor("#4E79A7")
axes[0].set_title("Notes Maths selon le sexe")
axes[0].set_xlabel("Sexe")
axes[0].set_ylabel("Note finale")

groups_sex_por = [d3[d3["sex"] == s]["G3.y"].values for s in ["F", "M"]]
bp2 = axes[1].boxplot(groups_sex_por, labels=["F", "M"], patch_artist=True)
bp2["boxes"][0].set_facecolor("#E15759")
bp2["boxes"][1].set_facecolor("#4E79A7")
axes[1].set_title("Notes Portugais selon le sexe")
axes[1].set_xlabel("Sexe")
axes[1].set_ylabel("Note finale")

plt.tight_layout()
#plt.show()

# Test de Wilcoxon / Mann-Whitney (non-paramétrique, équivalent R)
print("Test Maths (F vs M) :")
stat, pval = stats.mannwhitneyu(d3[d3["sex"] == "F"]["G3.x"],
                                 d3[d3["sex"] == "M"]["G3.x"])
print(f"W = {stat:.1f}, p-value = {pval:.4f}")

print("Test Portugais (F vs M) :")
stat, pval = stats.mannwhitneyu(d3[d3["sex"] == "F"]["G3.y"],
                                 d3[d3["sex"] == "M"]["G3.y"])
print(f"W = {stat:.1f}, p-value = {pval:.4f}")

# Moyennes par sexe
print(d3.groupby("sex")["G3.x"].mean())
print(d3.groupby("sex")["G3.y"].mean())


# ── 4. EFFET DU TEMPS D'ÉTUDE SUR LES NOTES ─────────────────
print("\n=== TEMPS D'ÉTUDE vs NOTES ===")
# studytime : 1=<2h, 2=2-5h, 3=5-10h, 4=>10h

def make_gradient_colors(base_hex_light, base_hex_dark, n):
    cmap = mcolors.LinearSegmentedColormap.from_list("grad", [base_hex_light, base_hex_dark])
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]

colors_mat = make_gradient_colors("#AEC6E8", "#4E79A7", 4)
colors_por = make_gradient_colors("#FFCF90", "#F28E2B", 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

groups_mat = [d3[d3["studytime.x"] == t]["G3.x"].values for t in sorted(d3["studytime.x"].unique())]
bp = axes[0].boxplot(groups_mat, labels=sorted(d3["studytime.x"].unique()), patch_artist=True)
for patch, color in zip(bp["boxes"], colors_mat):
    patch.set_facecolor(color)
axes[0].set_title("Maths : notes selon le temps d'étude")
axes[0].set_xlabel("Temps d'étude (1=<2h … 4=>10h)")
axes[0].set_ylabel("Note finale")

groups_por = [d3[d3["studytime.y"] == t]["G3.y"].values for t in sorted(d3["studytime.y"].unique())]
bp2 = axes[1].boxplot(groups_por, labels=sorted(d3["studytime.y"].unique()), patch_artist=True)
for patch, color in zip(bp2["boxes"], colors_por):
    patch.set_facecolor(color)
axes[1].set_title("Portugais : notes selon le temps d'étude")
axes[1].set_xlabel("Temps d'étude (1=<2h … 4=>10h)")
axes[1].set_ylabel("Note finale")

plt.tight_layout()
#plt.show()

# ANOVA
print("ANOVA Maths ~ studytime :")
model_anova_mat = ols("Q('G3.x') ~ C(Q('studytime.x'))", data=d3).fit()
print(anova_lm(model_anova_mat))

print("ANOVA Portugais ~ studytime :")
model_anova_por = ols("Q('G3.y') ~ C(Q('studytime.y'))", data=d3).fit()
print(anova_lm(model_anova_por))


# ── 5. CONSOMMATION D'ALCOOL ────────────────────────────────
print("\n=== ALCOOL vs NOTES ===")
# Dalc = consommation en semaine (1=très faible … 5=très élevée)
# Walc = consommation le week-end

d3["alcool_moy"] = (d3["Dalc.x"] + d3["Walc.x"]) / 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

jitter_dalc = d3["Dalc.x"] + np.random.uniform(-0.2, 0.2, size=len(d3))
axes[0].scatter(jitter_dalc, d3["G3.x"], color="#76B7B2", alpha=0.6)
m, b = np.polyfit(d3["Dalc.x"], d3["G3.x"], 1)
x_r = np.linspace(d3["Dalc.x"].min(), d3["Dalc.x"].max(), 100)
axes[0].plot(x_r, m * x_r + b, color="red", linewidth=2)
axes[0].set_title("Alcool semaine vs note Maths")
axes[0].set_xlabel("Dalc (1=faible, 5=élevé)")
axes[0].set_ylabel("G3 Maths")

jitter_walc = d3["Walc.x"] + np.random.uniform(-0.2, 0.2, size=len(d3))
axes[1].scatter(jitter_walc, d3["G3.x"], color="#59A14F", alpha=0.6)
m2, b2 = np.polyfit(d3["Walc.x"], d3["G3.x"], 1)
x_r2 = np.linspace(d3["Walc.x"].min(), d3["Walc.x"].max(), 100)
axes[1].plot(x_r2, m2 * x_r2 + b2, color="red", linewidth=2)
axes[1].set_title("Alcool week-end vs note Maths")
axes[1].set_xlabel("Walc (1=faible, 5=élevé)")
axes[1].set_ylabel("G3 Maths")

plt.tight_layout()
#plt.show()

print(f"Corrélation Dalc/G3 Maths : {d3['Dalc.x'].corr(d3['G3.x']):.4f}")
print(f"Corrélation Walc/G3 Maths : {d3['Walc.x'].corr(d3['G3.x']):.4f}")


# ── 6. ABSENTÉISME ──────────────────────────────────────────
print("\n=== ABSENCES vs NOTES ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(d3["absences.x"], d3["G3.x"], color="#4E79A7", alpha=0.6)
m, b = np.polyfit(d3["absences.x"], d3["G3.x"], 1)
x_r = np.linspace(d3["absences.x"].min(), d3["absences.x"].max(), 100)
axes[0].plot(x_r, m * x_r + b, color="red", linewidth=2)
axes[0].set_title("Absences vs Note Maths")
axes[0].set_xlabel("Nombre d'absences")
axes[0].set_ylabel("Note finale")

axes[1].scatter(d3["absences.y"], d3["G3.y"], color="#F28E2B", alpha=0.6)
m2, b2 = np.polyfit(d3["absences.y"], d3["G3.y"], 1)
x_r2 = np.linspace(d3["absences.y"].min(), d3["absences.y"].max(), 100)
axes[1].plot(x_r2, m2 * x_r2 + b2, color="red", linewidth=2)
axes[1].set_title("Absences vs Note Portugais")
axes[1].set_xlabel("Nombre d'absences")
axes[1].set_ylabel("Note finale")

plt.tight_layout()
#plt.show()

print(f"Corrélation absences/G3 Maths    : {d3['absences.x'].corr(d3['G3.x']):.4f}")
print(f"Corrélation absences/G3 Portugais: {d3['absences.y'].corr(d3['G3.y']):.4f}")


# ── 7. ÉDUCATION PARENTALE ──────────────────────────────────
print("\n=== ÉDUCATION PARENTALE vs NOTES ===")
# Medu / Fedu : 0=aucune … 4=supérieur

print(f"Corrélation Medu/G3 Maths    : {d3['Medu'].corr(d3['G3.x']):.4f}")
print(f"Corrélation Fedu/G3 Maths    : {d3['Fedu'].corr(d3['G3.x']):.4f}")
print(f"Corrélation Medu/G3 Portugais: {d3['Medu'].corr(d3['G3.y']):.4f}")
print(f"Corrélation Fedu/G3 Portugais: {d3['Fedu'].corr(d3['G3.y']):.4f}")

colors_medu = make_gradient_colors("#D4E5F5", "#4E79A7", 5)
colors_fedu = make_gradient_colors("#D4E5F5", "#2E5F8A", 5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

groups_medu = [d3[d3["Medu"] == lvl]["G3.x"].values for lvl in sorted(d3["Medu"].unique())]
bp = axes[0].boxplot(groups_medu, labels=sorted(d3["Medu"].unique()), patch_artist=True)
for patch, color in zip(bp["boxes"], colors_medu):
    patch.set_facecolor(color)
axes[0].set_title("Éducation mère vs Maths")
axes[0].set_xlabel("Niveau (0=aucun … 4=supérieur)")
axes[0].set_ylabel("Note finale")

groups_fedu = [d3[d3["Fedu"] == lvl]["G3.x"].values for lvl in sorted(d3["Fedu"].unique())]
bp2 = axes[1].boxplot(groups_fedu, labels=sorted(d3["Fedu"].unique()), patch_artist=True)
for patch, color in zip(bp2["boxes"], colors_fedu):
    patch.set_facecolor(color)
axes[1].set_title("Éducation père vs Maths")
axes[1].set_xlabel("Niveau (0=aucun … 4=supérieur)")
axes[1].set_ylabel("Note finale")

plt.tight_layout()
#plt.show()


# ── 8. RÉGRESSION LINÉAIRE MULTIPLE ─────────────────────────
print("\n=== RÉGRESSION LINÉAIRE — Note finale Maths ===")

# Encoder les variables catégorielles
d3["sex_bin"] = (d3["sex"] == "M").astype(int)
d3["internet_bin"] = (d3["internet"] == "yes").astype(int)
d3["romantic_x_bin"] = (d3["romantic.x"] == "yes").astype(int)
d3["romantic_y_bin"] = (d3["romantic.y"] == "yes").astype(int)

formula_maths = ("Q('G3.x') ~ age + sex_bin + Medu + Fedu + "
                 "Q('studytime.x') + Q('failures.x') + Q('absences.x') + "
                 "Q('Dalc.x') + Q('Walc.x') + internet_bin + romantic_x_bin")
modele_maths = smf.ols(formula_maths, data=d3).fit()
print(modele_maths.summary())

print("\n=== RÉGRESSION LINÉAIRE — Note finale Portugais ===")
formula_por = ("Q('G3.y') ~ age + sex_bin + Medu + Fedu + "
               "Q('studytime.y') + Q('failures.y') + Q('absences.y') + "
               "Q('Dalc.y') + Q('Walc.y') + internet_bin + romantic_y_bin")
modele_por = smf.ols(formula_por, data=d3).fit()
print(modele_por.summary())


# ── 9. ÉVOLUTION DES NOTES (G1 → G2 → G3) ──────────────────
print("\n=== PROGRESSION TRIMESTRIELLE ===")

moy_maths = [d3["G1.x"].mean(), d3["G2.x"].mean(), d3["G3.x"].mean()]
moy_por   = [d3["G1.y"].mean(), d3["G2.y"].mean(), d3["G3.y"].mean()]

plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3], moy_maths, "o-", color="#4E79A7", label="Mathématiques")
plt.plot([1, 2, 3], moy_por,   "o-", color="#F28E2B", label="Portugais")
plt.xticks([1, 2, 3], ["T1 (G1)", "T2 (G2)", "T3 (G3)"])
plt.ylim(8, 15)
plt.title("Évolution des moyennes trimestrielles")
plt.xlabel("Trimestre")
plt.ylabel("Moyenne de classe")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
#plt.show()


# ── 10. PROFIL DE RÉUSSITE ──────────────────────────────────
# Créer une variable de réussite globale (G3 >= 10 dans les deux)
d3["reussite"] = np.where((d3["G3.x"] >= 10) & (d3["G3.y"] >= 10), "Succès", "Échec")

print("\nRépartition réussite :")
print(d3["reussite"].value_counts())

# Profil moyen par groupe
print("\nProfil moyen – Succès vs Échec :")
cols_profil = ["age", "Medu", "Fedu", "studytime.x", "failures.x",
               "absences.x", "Dalc.x", "Walc.x", "G3.x", "G3.y"]
print(d3.groupby("reussite")[cols_profil].mean().round(2))

"""
# ── 11. ACP ────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n=== ACP ===")

vars_acp = [
    "age",
    "Medu",
    "Fedu",
    "studytime.x",
    "failures.x",
    "absences.x",
    "Dalc.x",
    "Walc.x",
    "G3.x"
]

X = d3[vars_acp]

X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Variance expliquée :")
print(np.round(pca.explained_variance_ratio_, 4))

plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(pca.explained_variance_ratio_)+1),
    np.cumsum(pca.explained_variance_ratio_),
    marker="o"
)
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulée expliquée")
plt.title("ACP - Variance expliquée cumulée")
plt.grid()
#plt.show()
loadings = pca.components_.T

plt.figure(figsize=(8,8))

for i, var in enumerate(vars_acp):
    plt.arrow(
        0,
        0,
        loadings[i,0],
        loadings[i,1],
        head_width=0.03
    )

    plt.text(
        loadings[i,0]*1.1,
        loadings[i,1]*1.1,
        var
    )

plt.axhline(0,color="grey")
plt.axvline(0,color="grey")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ACP - Cercle des corrélations")
plt.grid()
#plt.show()

##### --- KPV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Variables utilisées pour le clustering
X_cluster = d3[["G3.x", "G3.y"]]

# Standardisation
X_scaled = StandardScaler().fit_transform(X_cluster)

# K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

d3["cluster"] = kmeans.fit_predict(X_scaled)

# Moyennes des clusters
print(
    d3.groupby("cluster")[["G3.x", "G3.y"]]
      .mean()
      .sort_values("G3.x")
)

# Visualisation
plt.figure(figsize=(8,6))

couleurs = ["red", "orange", "green"]

for i in range(3):
    subset = d3[d3["cluster"] == i]

    plt.scatter(
        subset["G3.x"],
        subset["G3.y"],
        color=couleurs[i],
        label=f"Cluster {i}",
        alpha=0.7
    )

plt.xlabel("G3 Maths")
plt.ylabel("G3 Portugais")
plt.title("Classification K-Means (3 clusters)")
plt.legend()
plt.grid(True)
plt.show()

moyennes = (
    d3.groupby("cluster")[["G3.x", "G3.y"]]
      .mean()
      .sort_values("G3.x")
)

print(moyennes)