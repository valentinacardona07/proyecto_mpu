import os, glob, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATASET_PATH = "SisFall_dataset"
MODEL_FILE = "modelos_sisfall_sentado.pkl"
RANDOM_STATE = 42
FS = 200
WINDOW_SECONDS = 3
WIN = int(FS * WINDOW_SECONDS)

CLASES = {"F13": 0, "F14": 1, "F15": 2, "D07": 3, "D08": 3, "D09": 3, "D10": 3}
NOMBRES_CLASES = ["F13 (caída adelante)", "F14 (caída atrás)", "F15 (caída lateral)", "Sentado quieto"]
SENTADO = {"D07", "D08", "D09", "D10"}

ACC_SCALE = 32 / (2 ** 13)
GYRO_SCALE = 4000 / (2 ** 16)


def leer_archivo(path):
    filas = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            try:
                v = [int(x.strip()) for x in linea.strip().rstrip(";").split(",")]
                if len(v) == 9:
                    filas.append(v[:6])
            except ValueError:
                pass

    if not filas:
        return None

    df = pd.DataFrame(filas, columns=["ax", "ay", "az", "gx", "gy", "gz"])
    df[["ax", "ay", "az"]] *= ACC_SCALE
    df[["gx", "gy", "gz"]] *= GYRO_SCALE
    return df


def sujeto(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for p in name.split("_"):
        if p.startswith(("SA", "SE")):
            return p
    return os.path.basename(os.path.dirname(path))


def tomar_ventana(df, codigo):
    if len(df) < WIN:
        return None

    if codigo in SENTADO:
        centro = len(df) // 2
    else:
        acc = np.linalg.norm(df[["ax", "ay", "az"]].values, axis=1)
        centro = int(np.argmax(acc))

    i0 = max(0, min(len(df) - WIN, centro - WIN // 2))
    return df.iloc[i0:i0 + WIN].reset_index(drop=True)


def extraer_caracteristicas(data):
    x = np.asarray(data, dtype=float)
    acc_mag = np.linalg.norm(x[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)
    x = np.column_stack([x, acc_mag, gyro_mag])

    feats = []
    for s in x.T:
        feats += [s.mean(), s.std(), s.min(), s.max(), s.max() - s.min(), np.sqrt(np.mean(s * s))]
    return feats


def balancear(X, y, g):
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(np.sum(y == c) for c in np.unique(y))
    idx = []

    for c in np.unique(y):
        idx += list(rng.choice(np.where(y == c)[0], n, replace=False))

    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx], g[idx]


def cargar_dataset():
    X, y, g = [], [], []
    conteo = {k: 0 for k in CLASES}

    for path in sorted(glob.glob(os.path.join(DATASET_PATH, "**", "*.txt"), recursive=True)):
        codigo = os.path.basename(path)[:3]
        if codigo not in CLASES:
            continue

        df = leer_archivo(path)
        if df is None:
            continue

        v = tomar_ventana(df, codigo)
        if v is None:
            continue

        X.append(extraer_caracteristicas(v.values))
        y.append(CLASES[codigo])
        g.append(sujeto(path))
        conteo[codigo] += 1

    X, y, g = np.array(X), np.array(y), np.array(g)

    print("\nArchivos cargados:")
    for k, v in conteo.items():
        print(f"{k}: {v}")

    print("\nMuestras antes de balancear:")
    for i, n in enumerate(NOMBRES_CLASES):
        print(f"{n}: {np.sum(y == i)}")

    X, y, g = balancear(X, y, g)

    print("\nMuestras después de balancear:")
    for i, n in enumerate(NOMBRES_CLASES):
        print(f"{n}: {np.sum(y == i)}")

    return X, y, g


def evaluar(nombre, y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print(f"\n{'=' * 60}\n{nombre}\n{'=' * 60}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=NOMBRES_CLASES, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))


def main():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    X, y, g = cargar_dataset()

    split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(X, y, groups=g))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    modelos = {
        "arbol_decision": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
            alpha=1e-3, max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    modelos["arbol_decision"].fit(X_train, y_train)
    modelos["random_forest"].fit(X_train, y_train)
    modelos["mlp"].fit(X_train_sc, y_train)

    evaluar("Árbol de decisión", y_test, modelos["arbol_decision"].predict(X_test))
    evaluar("Random Forest", y_test, modelos["random_forest"].predict(X_test))
    evaluar("MLP", y_test, modelos["mlp"].predict(X_test_sc))

    joblib.dump({
        "arbol_decision": modelos["arbol_decision"],
        "random_forest": modelos["random_forest"],
        "mlp": modelos["mlp"],
        "scaler": scaler,
        "nombres_clases": NOMBRES_CLASES,
        "feature_count": X.shape[1],
        "fs": FS,
        "window_seconds": WINDOW_SECONDS,
    }, MODEL_FILE)

    print(f"\nModelos guardados en: {MODEL_FILE}")


if __name__ == "__main__":
    main()