import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATASET_PATH = r"SisFall_dataset"
RANDOM_STATE = 42
BALANCEAR_CLASES = True

CLASES = {
    "F13": 0,
    "F14": 1,
    "F15": 2,
    "D07": 3,
    "D08": 3,
    "D09": 3,
    "D10": 3,
}

NOMBRES_CLASES = [
    "F13 (caída adelante)",
    "F14 (caída atrás)",
    "F15 (caída lateral)",
    "Sentado quieto"
]

CODIGOS_SENTADO = {"D07", "D08", "D09", "D10"}

ACC_SCALE = (2 * 16) / (2 ** 13)
GYRO_SCALE = (2 * 2000) / (2 ** 16)


def leer_archivo(filepath):
    filas = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            linea = linea.strip().rstrip(";")
            if not linea:
                continue

            try:
                valores = [int(v.strip()) for v in linea.split(",")]
                if len(valores) == 9:
                    filas.append(valores[:6])
            except ValueError:
                pass

    if not filas:
        return None

    df = pd.DataFrame(filas, columns=["ax1", "ay1", "az1", "gx", "gy", "gz"])

    df[["ax1", "ay1", "az1"]] *= ACC_SCALE
    df[["gx", "gy", "gz"]] *= GYRO_SCALE

    return df


def obtener_sujeto(filepath):
    nombre = os.path.splitext(os.path.basename(filepath))[0]

    for parte in nombre.split("_"):
        if parte.startswith(("SA", "SE")):
            return parte

    return os.path.basename(os.path.dirname(filepath))


def recortar_sentado(df, inicio=0.35, fin=0.65):
    n = len(df)
    i0 = int(inicio * n)
    i1 = int(fin * n)

    if i1 - i0 < 50:
        return df

    return df.iloc[i0:i1].reset_index(drop=True)


def extraer_caracteristicas(df):
    caracteristicas = []

    for col in df.columns:
        s = df[col].values

        caracteristicas += [
            np.mean(s),
            np.std(s),
            np.min(s),
            np.max(s),
            np.max(s) - np.min(s),
            np.mean(s ** 2),
        ]

    return caracteristicas


def balancear_dataset(X, y, grupos):
    rng = np.random.default_rng(RANDOM_STATE)
    clases = np.unique(y)
    min_muestras = min(np.sum(y == c) for c in clases)

    indices = []

    for c in clases:
        idx = np.where(y == c)[0]
        indices.extend(rng.choice(idx, min_muestras, replace=False))

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices], grupos[indices]


def cargar_dataset(path):
    X, y, grupos = [], [], []
    conteo_codigos = {k: 0 for k in CLASES}

    archivos = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)

    if not archivos:
        print(f"\n[ERROR] No se encontraron archivos .txt en: {path}")
        return None, None, None

    for filepath in archivos:
        codigo = os.path.basename(filepath)[:3]

        if codigo not in CLASES:
            continue

        df = leer_archivo(filepath)

        if df is None or len(df) < 50:
            continue

        if codigo in CODIGOS_SENTADO:
            df = recortar_sentado(df)

        X.append(extraer_caracteristicas(df))
        y.append(CLASES[codigo])
        grupos.append(obtener_sujeto(filepath))
        conteo_codigos[codigo] += 1

    X = np.array(X)
    y = np.array(y)
    grupos = np.array(grupos)

    print("\n── Archivos cargados por código ───────────────────")
    for codigo, cantidad in conteo_codigos.items():
        print(f"  {codigo}: {cantidad}")

    print("\n── Muestras por clase antes de balancear ──────────")
    for i, nombre in enumerate(NOMBRES_CLASES):
        print(f"  {nombre}: {np.sum(y == i)}")

    if BALANCEAR_CLASES:
        X, y, grupos = balancear_dataset(X, y, grupos)

        print("\n── Muestras por clase después de balancear ────────")
        for i, nombre in enumerate(NOMBRES_CLASES):
            print(f"  {nombre}: {np.sum(y == i)}")

    return X, y, grupos


def evaluar_modelo(nombre, y_test, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    labels = list(range(len(NOMBRES_CLASES)))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\n{'=' * 70}")
    print(f"  {nombre}")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {acc * 100:.2f}%")

    aciertos = np.trace(cm)
    total = np.sum(cm)

    print(f"  Aciertos correctos: {aciertos}/{total}")
    print(f"  Accuracy calculado desde matriz: {(aciertos / total) * 100:.2f}%\n")

    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=NOMBRES_CLASES,
        zero_division=0
    ))

    print("  Matriz de confusión:")
    encabezado = " " * 26 + "".join([f"{n[:12]:>14}" for n in NOMBRES_CLASES])
    print(encabezado)

    for i, fila in enumerate(cm):
        valores = "".join([f"{v:>14}" for v in fila])
        print(f"  {NOMBRES_CLASES[i][:24]:24} {valores}")

    return acc, cm


def graficar_resultados(matrices, nombres, accuracies):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Matrices de confusión — validación por sujeto", fontsize=13)

    for ax, cm, nombre in zip(axes, matrices, nombres):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=NOMBRES_CLASES,
            yticklabels=NOMBRES_CLASES
        )

        ax.set_title(nombre)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig("matrices_confusion.png", dpi=120, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    barras = ax.bar(nombres, [a * 100 for a in accuracies], width=0.5)

    for barra, acc in zip(barras, accuracies):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.5,
            f"{acc * 100:.2f}%",
            ha="center",
            va="bottom"
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparación de modelos — validación por sujeto")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparacion_modelos.png", dpi=120, bbox_inches="tight")
    plt.show()


def main():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    print("\n════════════════════════════════════════════════════════════════")
    print("  Clasificación SisFall — validación por sujeto")
    print("════════════════════════════════════════════════════════════════")

    X, y, grupos = cargar_dataset(DATASET_PATH)

    if X is None or len(X) == 0:
        return

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.3,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=grupos))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    grupos_train = grupos[train_idx]
    grupos_test = grupos[test_idx]

    sujetos_comunes = set(grupos_train) & set(grupos_test)

    print("\n── División por sujeto ────────────────────────────")
    print(f"  Sujetos en train: {len(set(grupos_train))}")
    print(f"  Sujetos en test:  {len(set(grupos_test))}")
    print(f"  Sujetos repetidos entre train y test: {len(sujetos_comunes)}")
    print(f"  Muestras train: {len(X_train)}")
    print(f"  Muestras test:  {len(X_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    modelos = [
        (
            "Árbol de Decisión",
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            X_train,
            X_test
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            X_train,
            X_test
        ),
        (
            "Red Neuronal MLP",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE
            ),
            X_train_sc,
            X_test_sc
        ),
    ]

    accuracies = []
    matrices = []
    nombres = []
    modelos_entrenados = {}

    for nombre, modelo, Xtr, Xte in modelos:
        modelo.fit(Xtr, y_train)
        y_pred = modelo.predict(Xte)

        acc, cm = evaluar_modelo(nombre, y_test, y_pred)

        accuracies.append(acc)
        matrices.append(cm)
        nombres.append(nombre)
        modelos_entrenados[nombre] = modelo

    mejor_idx = int(np.argmax(accuracies))

    print("\n════════════════════════════════════════════════════════════════")
    print("  Resumen final")
    print("════════════════════════════════════════════════════════════════")

    for nombre, acc in zip(nombres, accuracies):
        print(f"  {nombre:<22} {acc * 100:6.2f}%")

    print(f"\n  Mejor modelo: {nombres[mejor_idx]} ({accuracies[mejor_idx] * 100:.2f}%)")

    joblib.dump(
        {
            "arbol_decision": modelos_entrenados["Árbol de Decisión"],
            "random_forest": modelos_entrenados["Random Forest"],
            "mlp": modelos_entrenados["Red Neuronal MLP"],
            "scaler": scaler,
            "nombres_clases": NOMBRES_CLASES,
        },
        "modelos_sisfall_sentado.pkl"
    )

    print("\n  Modelos guardados en: modelos_sisfall_sentado.pkl")

    try:
        graficar_resultados(matrices, nombres, accuracies)
        print("  Figuras guardadas: matrices_confusion.png y comparacion_modelos.png")
    except Exception as e:
        print(f"\n  [Aviso] No se pudieron generar gráficas: {e}")

    print("\n  ¡Listo!\n")


if __name__ == "__main__":
    main()