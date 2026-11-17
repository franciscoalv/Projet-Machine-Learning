# ============================================================
# ANALYSE DISCRIMINANTE — Dataset "student-mat.csv"
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_FILENAME = "student-mat.csv"  # change to student-mat.csv or student-por.csv
TEST_SIZE = 0.2
N_PCA_COMPONENTS = 7


def load_data():
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    path = parent_dir / "data" / DATA_FILENAME
    df = pd.read_csv(path, sep=",")
    return df


def prepare_data(df):
    # Variable cible : réussite en maths (G3 >= 10)
    df = df.copy()
    df["passed"] = np.where(df["G3"] >= 10, 1, 0)

    numeric_features = [
        "age", "Medu",  "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout","Walc", "health", "absences"
    ]

    categorical_features = [
        "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
        "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    ]

    X = df[numeric_features + categorical_features]
    y = df["passed"]

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    print(f"\n=== Résultats {name} ===")
    print(f"Accuracy test : {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "accuracy_test": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": report,
    }




def summarize_target(y):
    counts = y.value_counts().sort_index()
    proportions = y.value_counts(normalize=True).sort_index() * 100
    print("\nRépartition de la cible :")
    for label, count in counts.items():
        print(f"  {label} : {count} ({proportions[label]:.1f} %)" )


def main():
    df = load_data()

    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    X, y, numeric_features, categorical_features = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    # ============================================================
    # Recherche du meilleur reg_param pour QDA
    # ============================================================

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    preprocessor = build_pipeline(numeric_features, categorical_features)

    reg_params = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    print("\n=== Recherche du meilleur reg_param ===")

    best_score = 0
    best_reg = None

    for reg in reg_params:

        qda_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=N_PCA_COMPONENTS)),
                ("classifier", QuadraticDiscriminantAnalysis(reg_param=reg))
            ]
        )

        scores = cross_val_score(
            qda_pipeline,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy"
        )

        mean_score = scores.mean()

        print(
            f"reg_param = {reg:<4} | "
            f"Accuracy CV = {mean_score:.4f} "
            f"(± {scores.std():.4f})"
        )

        if mean_score > best_score:
            best_score = mean_score
            best_reg = reg

    print("\n=== Meilleur paramètre ===")
    print(f"reg_param optimal = {best_reg}")
    print(f"Accuracy CV moyenne = {best_score:.4f}")

    print(f"Train/test : {len(X_train)} / {len(X_test)} ({int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%)")
    print(f"Distribution classes train : {y_train.value_counts().to_dict()}")
    print(f"Distribution classes test  : {y_test.value_counts().to_dict()}")

    preprocessor = build_pipeline(numeric_features, categorical_features)

    lda = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=N_PCA_COMPONENTS)),
            ("classifier", LinearDiscriminantAnalysis())
        ]
    )

    qda = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=N_PCA_COMPONENTS)),
            (
                "classifier",
                QuadraticDiscriminantAnalysis(reg_param=best_reg), 
            ),
        ]
    )

    print("\n=== Vérifications ===")
    summarize_target(y)

    results = []
    results.append(evaluate_model(lda, X_train, X_test, y_train, y_test, "LDA"))
    results.append(evaluate_model(qda, X_train, X_test, y_train, y_test, "QDA"))

    used_features = numeric_features + categorical_features
    dropped_features = [col for col in df.columns if col not in used_features + ["passed"]]

    save_analysis_figure(results, y_test, best_reg, used_features, dropped_features, N_PCA_COMPONENTS)

    print("\nNote : LDA est une analyse discriminante linéaire, QDA est une version non-linéaire.")
    print("Si QDA échoue à cause de colinéarité, le paramètre reg_param réduit l’impact des variables parfaitement liées.")

def save_analysis_figure(results, y_test, best_reg, used_features, dropped_features, n_pca_components, filename="analyse_discriminante_report.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, res in enumerate(results[:2]):
        cm = res["confusion_matrix"]
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{res['name']} - Confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)

    ax = axes[2]
    names = [res["name"] for res in results[:2]]
    acc_test = [res["accuracy_test"] for res in results[:2]]
    ax.bar(np.arange(len(names)), acc_test, width=0.5, color=["#4E79A7", "#F28E2B"], label="Test accuracy")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy sur le test set")
    ax.legend()

    ax = axes[3]
    ax.axis("off")

    def format_feature_lines(title, features):
        lines = [title]
        if not features:
            return lines + ["  Aucune"]
        wrapped_lines = textwrap.wrap(
            ", ".join(features),
            width=75,
            subsequent_indent="  "
        )
        return lines + ["  " + wrapped_lines[0]] + [line for line in wrapped_lines[1:]]

    summary_text = [
        f"Test set : {len(y_test)} observations",
        f"Cible 0 / 1 : {y_test.value_counts().get(0, 0)} / {y_test.value_counts().get(1, 0)}",
        "",
        "Classes :",
        "  0 = échec (G3 < 10)",
        "  1 = réussite (G3 >= 10)",
        "",
        f"Split : entraînement/test = {int((1 - TEST_SIZE) * 100)}% / {int(TEST_SIZE * 100)}%",
        "Pas de validation croisée, uniquement entraînement/test",
        f"Meilleur reg_param pour QDA : {best_reg}",
        f"PCA components gardés : {n_pca_components}",
        "",
    ]
    summary_text += format_feature_lines("Variables originales utilisées :", used_features)
    summary_text.append("")
    summary_text += format_feature_lines("Variables abandonnées :", dropped_features)
    summary_text.append("")
    for res in results[:2]:
        summary_text.append(f"{res['name']}")
        summary_text.append(f"  Accuracy test : {res['accuracy_test']:.4f}")
        summary_text.append(f"  Precision(0) : {res['classification_report']['0']['precision']:.4f}")
        summary_text.append(f"  Recall(0) : {res['classification_report']['0']['recall']:.4f}")
        summary_text.append(f"  Precision(1) : {res['classification_report']['1']['precision']:.4f}")
        summary_text.append(f"  Recall(1) : {res['classification_report']['1']['recall']:.4f}")
        summary_text.append("")

    ax.text(0, 1, "\n".join(summary_text), family="monospace", fontsize=10, va="top")
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nRapport image sauvegardé : {filename}")



if __name__ == "__main__":
    main()
