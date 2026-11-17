"""
Régression Logistique — Student Performance Dataset
Cible : pass/fail (G3 >= 10)
Compare avec RF et XGBoost pour avoir une vue complète

>>> VARIANTE ACP <<<
Au lieu des variables d'origine, les modèles sont entraînés sur les
AXES FACTORIELS d'une Analyse en Composantes Principales.
Pipeline : encodage → StandardScaler → PCA  (ajustés sur le TRAIN seul).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE    = 0.2
THRESHOLD    = 10

# Nombre de composantes ACP à conserver :
#   - un float dans ]0, 1[  → seuil de variance cumulée à atteindre (ex. 0.90)
#   - un int                → nombre fixe de composantes
N_COMPONENTS = 0.90

# ─────────────────────────────────────────────
# 1. Chargement & prétraitement
# ─────────────────────────────────────────────

def load_and_preprocess(path="student-mat.csv"):
    df = pd.read_csv(path)

    # Encodage binaire
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])

    # One-hot pour le reste
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df["pass"] = (df["G3"] >= THRESHOLD).astype(int)
    feature_cols = [c for c in df.columns if c not in ("G3", "pass", "G1", "G2")]

    X = df[feature_cols]
    y = df["pass"]
    return X, y, feature_cols


# ─────────────────────────────────────────────
# 1bis. ACP : standardisation + PCA (fit sur le TRAIN seulement)
# ─────────────────────────────────────────────

def apply_pca(X_train, X_test, n_components, feature_cols):
    """Standardise puis projette les données sur les axes factoriels.

    Le scaler et l'ACP sont ajustés UNIQUEMENT sur X_train pour éviter
    toute fuite d'information du jeu de test vers le modèle.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)

    k = pca.n_components_
    pc_names = [f"PC{i+1}" for i in range(k)]
    var_cum = pca.explained_variance_ratio_.sum()

    print(f"  ACP : {len(feature_cols)} variables → {k} composantes "
          f"({var_cum:.1%} de variance conservée)")
    print(f"  Variance des 5 premiers axes : "
          + "  ".join(f"PC{i+1}={r:.1%}" for i, r in
                      enumerate(pca.explained_variance_ratio_[:5])))

    Xtr_p = pd.DataFrame(Xtr_p, columns=pc_names, index=X_train.index)
    Xte_p = pd.DataFrame(Xte_p, columns=pc_names, index=X_test.index)
    return Xtr_p, Xte_p, pca, pc_names


# ─────────────────────────────────────────────
# 2. Entraînement des modèles
# ─────────────────────────────────────────────

def train_all(X_train, y_train):
    # Régression logistique dans un pipeline avec standardisation.
    # (Les axes ACP sont déjà centrés ; on standardise chaque composante
    #  à variance unitaire, ce qui aide la régularisation L2.)
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ))
    ])
    logreg.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_train, y_train)

    return logreg, rf, xgb


# ─────────────────────────────────────────────
# 3. Visualisations
# ─────────────────────────────────────────────

def plot_all(logreg, rf, xgb, X_train, X_test, y_train, y_test, feature_cols):
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Régression Logistique vs Random Forest vs XGBoost\n"
                 "(Student Performance — pass/fail — entrées = axes ACP)",
                 fontsize=15, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    models = [
        ("Logistic Regression", logreg, "#9C27B0"),
        ("Random Forest",       rf,     "#2196F3"),
        ("XGBoost",             xgb,    "#FF5722"),
    ]

    # ── (A) Matrices de confusion ──
    for i, (name, model, color) in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        acc = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.set_title(f"{name}\nAccuracy={acc:.2%}", color=color)

    # ── (B) Courbes ROC ──
    ax = fig.add_subplot(gs[1, 0])
    for name, model, color in models:
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Courbes ROC"); ax.legend(fontsize=8)

    # ── (C) Courbes Precision-Recall ──
    ax = fig.add_subplot(gs[1, 1])
    for name, model, color in models:
        prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall"); ax.legend(fontsize=8)

    # ── (D) Cross-validation F1 ──
    ax = fig.add_subplot(gs[1, 2])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_data = {}
    for name, model, color in models:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_data[name.replace(" ", "\n")] = scores
    cv_df = pd.DataFrame(cv_data)
    bp = cv_df.boxplot(ax=ax, patch_artist=True, return_type="dict")
    colors_list = ["#CE93D8", "#90CAF9", "#FFAB91"]
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
    ax.set_title("Cross-validation F1 (5-fold)")
    ax.set_ylabel("F1 score")
    ax.tick_params(axis="x", labelsize=8)

    # ── (E) Coefficients Logistic Regression — Top 15 (sur les axes ACP) ──
    ax = fig.add_subplot(gs[2, 0])
    clf_step = logreg.named_steps["clf"]
    coefs = pd.Series(clf_step.coef_[0], index=feature_cols)
    top_pos = coefs.nlargest(8)
    top_neg = coefs.nsmallest(7)
    top = pd.concat([top_neg, top_pos]).sort_values()
    colors_coef = ["#EF5350" if v < 0 else "#42A5F5" for v in top.values]
    top.plot(kind="barh", ax=ax, color=colors_coef, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title("Coefficients Logistic Reg. — axes ACP\n"
                 "(rouge=risque échec, bleu=chance réussite)")
    ax.set_xlabel("Coefficient (log-odds)")

    # ── (F) Importance des axes ACP — Random Forest ──
    ax = fig.add_subplot(gs[2, 1])
    imp = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(15).sort_values()
    imp.plot(kind="barh", ax=ax, color="#2196F3", alpha=0.85)
    ax.set_title("Importance des axes ACP\nRandom Forest (top 15)")
    ax.set_xlabel("Importance")

    # ── (G) Comparaison métriques synthèse ──
    ax = fig.add_subplot(gs[2, 2])
    summary = {}
    for name, model, color in models:
        y_pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import f1_score, precision_score, recall_score
        summary[name] = {
            "F1":        f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "AUC":       auc(*roc_curve(y_test, prob)[:2]),
        }
    summary_df = pd.DataFrame(summary).T
    x = np.arange(len(summary_df))
    width = 0.2
    metrics = ["F1", "Precision", "Recall", "AUC"]
    metric_colors = ["#AB47BC", "#29B6F6", "#66BB6A", "#FFA726"]
    for j, (metric, mc) in enumerate(zip(metrics, metric_colors)):
        ax.bar(x + j * width, summary_df[metric], width, label=metric, color=mc, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([n.split()[0] for n in summary_df.index], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparaison des métriques\n(classe Pass)")
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")

    plt.savefig("student_logistic_acp_results.png", dpi=150, bbox_inches="tight")
    print("\n  Figure sauvegardée : student_logistic_acp_results.png")


# ─────────────────────────────────────────────
# 4. Analyse des coefficients (interprétabilité)
# ─────────────────────────────────────────────

def print_top_coefficients(logreg, feature_cols, n=10):
    clf = logreg.named_steps["clf"]
    coefs = pd.Series(clf.coef_[0], index=feature_cols).sort_values()
    print("\n── Top axes ACP associés au RISQUE D'ÉCHEC (coefficients négatifs) ──")
    print(coefs.head(n).to_string())
    print("\n── Top axes ACP associés à la RÉUSSITE (coefficients positifs) ──")
    print(coefs.tail(n).iloc[::-1].to_string())


def print_pca_loadings(pca, original_features, pc_names, top_axes=3, n=6):
    """Pour interpréter les axes : variables qui pèsent le plus sur chaque PC."""
    print("\n── Interprétation des principaux axes ACP "
          "(variables de plus fort poids) ──")
    loadings = pd.DataFrame(pca.components_, columns=original_features, index=pc_names)
    for pc in pc_names[:top_axes]:
        row = loadings.loc[pc]
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
        contenu = "  ".join(f"{v:+.2f}·{name}" for name, v in top.items())
        print(f"  {pc} : {contenu}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  RÉGRESSION LOGISTIQUE + RF + XGBoost  (entrées = axes ACP)")
    print(f"  Cible : pass/fail (G3 >= {THRESHOLD})")
    print("=" * 55)

    X, y, original_features = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain : {len(X_train)}  |  Test : {len(X_test)}")
    print(f"Pass: {y_test.sum()}  Fail: {(y_test==0).sum()}")

    # ── ACP : on remplace les variables par les axes factoriels ──
    print("\n── Analyse en Composantes Principales ──")
    X_train, X_test, pca, pc_names = apply_pca(
        X_train, X_test, N_COMPONENTS, original_features
    )
    feature_cols = pc_names   # désormais les "features" sont les composantes

    print("\n── Entraînement ──")
    logreg, rf, xgb = train_all(X_train, y_train)

    print("\n── Métriques Classification ──")
    for name, model in [("Logistic Regression", logreg), ("Random Forest", rf), ("XGBoost", xgb)]:
        print(f"\n  {name}")
        print(classification_report(y_test, model.predict(X_test), target_names=["Fail", "Pass"]))

    print_top_coefficients(logreg, feature_cols)
    print_pca_loadings(pca, original_features, pc_names)

    print("\n── Visualisations ──")
    plot_all(logreg, rf, xgb, X_train, X_test, y_train, y_test, feature_cols)
    print("\nFin du script.")
