import os, glob, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

DATASET_PATH = "SisFall_dataset"
RANDOM_STATE = 42

CLASES = {"F13": 0, "F14": 1, "F15": 2, "D07": 3, "D08": 3, "D09": 3, "D10": 3}
NOMBRES_CLASES = ["F13 (adelante)", "F14 (atrás)", "F15 (lateral)", "Sentado quieto"]
CODIGOS_SENTADO = {"D07", "D08", "D09", "D10"}

ACC_SCALE = (2 * 16) / (2 ** 13)
GYRO_SCALE = (2 * 2000) / (2 ** 16)

def leer_archivo(filepath):
    filas = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            linea = linea.strip().rstrip(";")
            if not linea: continue
            try:
                valores = [int(v.strip()) for v in linea.split(",")]
                if len(valores) == 9: filas.append(valores[:6])
            except ValueError: pass
    if not filas: return None
    df = pd.DataFrame(filas, columns=["ax1", "ay1", "az1", "gx", "gy", "gz"])
    df[["ax1", "ay1", "az1"]] *= ACC_SCALE
    df[["gx", "gy", "gz"]] *= GYRO_SCALE
    return df

def extraer_caracteristicas(df):
    caracteristicas = []
    for col in df.columns:
        s = df[col].values
        caracteristicas.extend([np.mean(s), np.std(s), np.min(s), np.max(s), np.max(s) - np.min(s), np.mean(s ** 2)])
    return caracteristicas

def cargar_dataset(path):
    X, y, grupos = [], [], []
    archivos = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
    for filepath in archivos:
        codigo = os.path.basename(filepath)[:3]
        if codigo not in CLASES: continue
        df = leer_archivo(filepath)
        if df is None or len(df) < 50: continue
        if codigo in CODIGOS_SENTADO:
            n = len(df)
            df = df.iloc[int(0.35 * n):int(0.65 * n)].reset_index(drop=True)
        X.append(extraer_caracteristicas(df))
        y.append(CLASES[codigo])
        
        nombre = os.path.splitext(os.path.basename(filepath))[0]
        sujeto = next((p for p in nombre.split("_") if p.startswith(("SA", "SE"))), os.path.basename(os.path.dirname(filepath)))
        grupos.append(sujeto)
    return np.array(X), np.array(y), np.array(grupos)

def main():
    print("[INFO] Cargando y procesando dataset...")
    X, y, grupos = cargar_dataset(DATASET_PATH)
    if X is None or len(X) == 0: return

    # Balancear clases
    rng = np.random.default_rng(RANDOM_STATE)
    min_m = min(np.sum(y == c) for c in np.unique(y))
    indices = np.hstack([rng.choice(np.where(y == c)[0], min_m, replace=False) for c in np.unique(y)])
    X, y, grupos = X[indices], y[indices], grupos[indices]

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE).split(X, y, groups=grupos))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    modelos = {
        "arbol_decision": DecisionTreeClassifier(max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.001, max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
    }

    print("\n[INFO] Entrenando modelos (todos con datos normalizados)...")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_sc, y_train) 
        acc = accuracy_score(y_test, modelo.predict(X_test_sc))
        print(f"  {nombre:<15}: {acc * 100:.2f}%")

    joblib.dump({"modelos": modelos, "scaler": scaler, "nombres_clases": NOMBRES_CLASES}, "modelos_sisfall_sentado.pkl")
    print("\n[INFO] Modelos y Scaler guardados en: modelos_sisfall_sentado.pkl")

if __name__ == "__main__": 
    main()