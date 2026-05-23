import os
import glob
import sys
import io
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Configurar la codificación de salida a UTF-8 para evitar errores de codificación en consolas de Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Configuración ──────────────────────────────────────────────────────────────

# Ruta de la carpeta raíz del SisFall en tu PC
DATASET_PATH = r"SisFall_dataset"

# Clases que vamos a clasificar (incluye las 3 caídas y D07 mapeado a Sentado)
CLASES = {
    "F13": 0, 
    "F14": 1, 
    "F15": 2,
    # Códigos de SisFall relacionados con estar sentado o sentarse (Grupo Caso B)
    "D07": 3,
    "D08": 3,
    "D09": 3,
    "D10": 3,
    "D11": 3,
    "D12": 3,
    "D13": 3
}
NOMBRES_CLASES = ["F13 (adelante)", "F14 (atrás)", "F15 (lateral)", "Sentado"]

# Factores de conversión para usar solo señales equivalentes al MPU6050
# Acelerómetro ADXL345 del dataset SisFall: ±16g, 13 bits (para ax1, ay1, az1)
ACC_SCALE = (2 * 16) / (2 ** 13)
# Giroscopio ITG3200 del dataset SisFall: ±2000°/s, 16 bits (para gx, gy, gz)
GYRO_SCALE = (2 * 2000) / (2 ** 16)


# ── 1. Carga y conversión de datos ────────────────────────────────────────────

def leer_archivo(filepath):
    """Lee un archivo .txt del SisFall y retorna un DataFrame con 6 columnas."""
    filas = []

    with open(filepath, "r") as f:
        for linea in f:
            linea = linea.strip().rstrip(";")

            if not linea:
                continue

            try:
                valores = [int(v.strip()) for v in linea.split(",")]

                # SisFall tiene 9 columnas, pero solo usamos las primeras 6:
                # ax1, ay1, az1, gx, gy, gz
                if len(valores) == 9:
                    filas.append(valores[:6])

            except ValueError:
                continue

    if not filas:
        return None

    columnas = ["ax1", "ay1", "az1", "gx", "gy", "gz"]
    df = pd.DataFrame(filas, columns=columnas)

    # Convertir acelerómetro ADXL345 a g
    df["ax1"] *= ACC_SCALE
    df["ay1"] *= ACC_SCALE
    df["az1"] *= ACC_SCALE

    # Convertir giroscopio ITG3200 a °/s
    df["gx"] *= GYRO_SCALE
    df["gy"] *= GYRO_SCALE
    df["gz"] *= GYRO_SCALE

    return df


def leer_archivo_usuario(filepath):
    """Lee el archivo CSV del usuario (MPU6050) y retorna un DataFrame con las 6 columnas de interés."""
    # Las columnas del CSV son: n, timestamp, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, temp_C
    df = pd.read_csv(filepath)
    df_new = df[["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]].copy()
    # Renombrar para que coincidan exactamente con el formato de SisFall
    df_new.columns = ["ax1", "ay1", "az1", "gx", "gy", "gz"]
    return df_new


def extraer_caracteristicas(df):
    """
    Extrae un vector de características estadísticas por señal.
    Para cada columna calcula: media, std, min, max, rango, energía.
    Total: 6 columnas × 6 estadísticos = 36 características.
    """
    caracteristicas = []
    for col in df.columns:
        serie = df[col].values
        caracteristicas += [
            np.mean(serie),
            np.std(serie),
            np.min(serie),
            np.max(serie),
            np.max(serie) - np.min(serie),          # rango
            np.mean(serie ** 2),                     # energía
        ]
    return caracteristicas


def cargar_dataset(path):
    """Recorre todas las carpetas y carga los archivos de las clases configuradas."""
    X, y = [], []
    archivos_encontrados = {k: 0 for k in CLASES}

    patron = os.path.join(path, "**", "*.txt")
    archivos = glob.glob(patron, recursive=True)

    if not archivos:
        print(f"\n[ERROR] No se encontraron archivos .txt en: {path}")
        print("Verifica que DATASET_PATH apunte a la carpeta raíz del SisFall.")
        return None, None

    for filepath in archivos:
        nombre = os.path.basename(filepath)
        codigo = nombre[:3]  # F13, F14, F15, D01, etc.

        if codigo not in CLASES:
            continue

        df = leer_archivo(filepath)
        if df is None or len(df) < 50:
            continue

        feats = extraer_caracteristicas(df)
        X.append(feats)
        y.append(CLASES[codigo])
        archivos_encontrados[codigo] += 1

    print("\n── Archivos cargados ──────────────────────────────")
    for k, v in archivos_encontrados.items():
        print(f"  {k}: {v} archivos")
    print(f"  Total muestras: {len(X)}")

    return np.array(X), np.array(y)


# ── 2. Entrenamiento y evaluación ─────────────────────────────────────────────

def evaluar_modelo(nombre, y_test, y_pred):
    """Imprime métricas y matriz de confusión para un modelo."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*52}")
    print(f"  {nombre}")
    print(f"{'='*52}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=NOMBRES_CLASES))

    cm = confusion_matrix(y_test, y_pred)
    print("  Matriz de confusión:")
    SHORT_CLASSES = ["F13", "F14", "F15", "Sentado"]
    header = "  " + f"{'':18}" + "".join([f"{c:>10}" for c in SHORT_CLASSES])
    print(header)
    for i, fila in enumerate(cm):
        row_str = f"  {NOMBRES_CLASES[i]:18}" + "".join([f"{val:>10}" for val in fila])
        print(row_str)

    return acc, cm


def graficar_matrices(matrices, nombres):
    """Genera una figura con las 3 matrices de confusión."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Matrices de confusión — Clasificación F13/F14/F15/Sentado", fontsize=13)

    labels = ["F13", "F14", "F15", "Sentado"]
    for ax, cm, nombre in zip(axes, matrices, nombres):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=labels,
            yticklabels=labels
        )
        ax.set_title(nombre, fontsize=11)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig("matrices_confusion_sentado.png", dpi=120, bbox_inches="tight")
    print("\n  Figura guardada: matrices_confusion_sentado.png")
    plt.show()


def graficar_comparacion(nombres, accuracies):
    """Gráfico de barras comparando accuracy de los 3 modelos."""
    import matplotlib.pyplot as plt

    colores = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(nombres, [a * 100 for a in accuracies], color=colores, width=0.5)

    for barra, acc in zip(barras, accuracies):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.5,
            f"{acc*100:.2f}%",
            ha="center", va="bottom", fontsize=11
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparación de modelos — SisFall con Clase Sentado")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparacion_modelos_sentado.png", dpi=120, bbox_inches="tight")
    print("  Figura guardada: comparacion_modelos_sentado.png")
    plt.show()


# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    print("\n══════════════════════════════════════════════════")
    print("  Clasificación SisFall — F13 / F14 / F15 / Sentado (D07-D13)")
    print("══════════════════════════════════════════════════")

    # 1. Cargar datos
    print(f"\nBuscando datos en: {DATASET_PATH}")
    X, y = cargar_dataset(DATASET_PATH)

    if X is None or len(X) == 0:
        return

    # 2. División train/test (70/30, estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

    # 3. Normalización (necesaria para MLP)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    accuracies = []
    matrices   = []
    nombres    = []

    # ── Modelo 1: Árbol de Decisión ───────────────────────────────────────────
    arbol = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )
    arbol.fit(X_train, y_train)
    y_pred_arbol = arbol.predict(X_test)
    acc, cm = evaluar_modelo("Árbol de Decisión (max_depth=5)", y_test, y_pred_arbol)
    accuracies.append(acc)
    matrices.append(cm)
    nombres.append("Árbol de Decisión")

    # ── Modelo 2: Random Forest ───────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc, cm = evaluar_modelo("Random Forest (50 árboles, max_depth=6)", y_test, y_pred_rf)
    accuracies.append(acc)
    matrices.append(cm)
    nombres.append("Random Forest")

    # ── Modelo 3: Red Neuronal (MLP) ──────────────────────────────────────────
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train_sc, y_train)
    y_pred_mlp = mlp.predict(X_test_sc)
    acc, cm = evaluar_modelo("Red Neuronal MLP (64→32)", y_test, y_pred_mlp)
    accuracies.append(acc)
    matrices.append(cm)
    nombres.append("Red Neuronal MLP")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════")
    print("  Resumen final")
    print("══════════════════════════════════════════════════")
    for nombre, acc in zip(nombres, accuracies):
        barra = "█" * int(acc * 30)
        print(f"  {nombre:<25} {acc*100:6.2f}%  {barra}")

    mejor_idx = int(np.argmax(accuracies))
    print(f"\n  Mejor modelo: {nombres[mejor_idx]} ({accuracies[mejor_idx]*100:.2f}%)")

    # ── Prueba con datos de usuario (.csv) ────────────────────────────────────
    archivos_csv = glob.glob("*.csv")
    if archivos_csv:
        print("\n══════════════════════════════════════════════════")
        print("  Pruebas con datos propios del usuario (.csv)")
        print("══════════════════════════════════════════════════")
        for ruta_usuario in archivos_csv:
            print(f"\n  Procesando: {os.path.basename(ruta_usuario)}")
            try:
                df_usr = leer_archivo_usuario(ruta_usuario)
                if df_usr is not None and len(df_usr) >= 30:
                    feats_usr = np.array([extraer_caracteristicas(df_usr)])
                    
                    # Predicciones
                    pred_arbol = arbol.predict(feats_usr)[0]
                    prob_arbol = arbol.predict_proba(feats_usr)[0]
                    
                    pred_rf = rf.predict(feats_usr)[0]
                    prob_rf = rf.predict_proba(feats_usr)[0]
                    
                    feats_usr_sc = scaler.transform(feats_usr)
                    pred_mlp = mlp.predict(feats_usr_sc)[0]
                    prob_mlp = mlp.predict_proba(feats_usr_sc)[0]
                    
                    print(f"    - Árbol de Decisión: {NOMBRES_CLASES[pred_arbol]} (Confianza: {prob_arbol[pred_arbol]*100:.2f}%)")
                    print(f"      Detalle: " + ", ".join([f"{NOMBRES_CLASES[i]}: {p*100:.1f}%" for i, p in enumerate(prob_arbol)]))
                    
                    print(f"    - Random Forest: {NOMBRES_CLASES[pred_rf]} (Confianza: {prob_rf[pred_rf]*100:.2f}%)")
                    print(f"      Detalle: " + ", ".join([f"{NOMBRES_CLASES[i]}: {p*100:.1f}%" for i, p in enumerate(prob_rf)]))
                    
                    print(f"    - Red Neuronal MLP: {NOMBRES_CLASES[pred_mlp]} (Confianza: {prob_mlp[pred_mlp]*100:.2f}%)")
                    print(f"      Detalle: " + ", ".join([f"{NOMBRES_CLASES[i]}: {p*100:.1f}%" for i, p in enumerate(prob_mlp)]))
                else:
                    print("    [ERROR] Muestras insuficientes (mínimo 30 requeridas).")
            except Exception as e:
                print(f"    [ERROR] No se pudo procesar el archivo: {e}")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    try:
        graficar_matrices(matrices, nombres)
        graficar_comparacion(nombres, accuracies)
    except Exception as e:
        print(f"\n  [Aviso] No se pudieron generar gráficas: {e}")

    print("\n  ¡Listo!\n")


if __name__ == "__main__":
    main()
