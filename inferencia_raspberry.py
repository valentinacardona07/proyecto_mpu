#!/usr/bin/env python3

import argparse
import time
from collections import deque

import joblib
import numpy as np
from smbus2 import SMBus

I2C_BUS = 1
MPU_ADDR = 0x68
MODEL_FILE = "modelos_sisfall_sentado.pkl"
FS = 200.0
WINDOW_SECONDS = 3.0
PREDICT_EVERY_SECONDS = 1
ACC_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4

# Umbral mínimo de confianza para que el MLP emita una predicción.
# Si la probabilidad máxima es menor que esto, se reporta "Sentado quieto".
MLP_CONFIDENCE_THRESHOLD = 0.60

REG_SMPLRT_DIV   = 0x19
REG_CONFIG        = 0x1A
REG_GYRO_CONFIG   = 0x1B
REG_ACCEL_CONFIG  = 0x1C
REG_ACCEL_XOUT_H  = 0x3B
REG_PWR_MGMT_1    = 0x6B
REG_WHO_AM_I      = 0x75

NOMBRES_CLASES_IDX = {
    0: "F13 (caída adelante)",
    1: "F14 (caída atrás)",
    2: "F15 (caída lateral)",
    3: "Sentado quieto",
}


def to_int16(msb, lsb):
    value = (msb << 8) | lsb
    return value - 65536 if value >= 32768 else value


def init_mpu(bus):
    who = bus.read_byte_data(MPU_ADDR, REG_WHO_AM_I)
    print(f"[INFO] WHO_AM_I = 0x{who:02X}")
    bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x00)
    time.sleep(0.1)
    bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x03)
    bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 4)
    bus.write_byte_data(MPU_ADDR, REG_GYRO_CONFIG, 0x18)
    bus.write_byte_data(MPU_ADDR, REG_ACCEL_CONFIG, 0x18)
    time.sleep(0.1)


def read_mpu_raw(bus):
    data = bus.read_i2c_block_data(MPU_ADDR, REG_ACCEL_XOUT_H, 14)
    ax = to_int16(data[0], data[1])
    ay = to_int16(data[2], data[3])
    az = to_int16(data[4], data[5])
    gx = to_int16(data[8], data[9])
    gy = to_int16(data[10], data[11])
    gz = to_int16(data[12], data[13])
    return ax, ay, az, gx, gy, gz


def calibrar_giroscopio(bus, muestras=300):
    print("[INFO] Calibrando giroscopio. Deja el sensor quieto...")
    gx_vals, gy_vals, gz_vals = [], [], []
    periodo = 1.0 / FS
    for _ in range(muestras):
        t0 = time.perf_counter()
        _, _, _, gx, gy, gz = read_mpu_raw(bus)
        gx_vals.append(gx / GYRO_LSB_PER_DPS)
        gy_vals.append(gy / GYRO_LSB_PER_DPS)
        gz_vals.append(gz / GYRO_LSB_PER_DPS)
        elapsed = time.perf_counter() - t0
        if elapsed < periodo:
            time.sleep(periodo - elapsed)
    offset = np.array([np.mean(gx_vals), np.mean(gy_vals), np.mean(gz_vals)])
    print(f"[INFO] Offset gyro: gx={offset[0]:.3f}, gy={offset[1]:.3f}, gz={offset[2]:.3f} °/s")
    return offset


def read_mpu_units(bus, gyro_offset):
    ax, ay, az, gx, gy, gz = read_mpu_raw(bus)
    ax /= ACC_LSB_PER_G
    ay /= ACC_LSB_PER_G
    az /= ACC_LSB_PER_G
    gx = gx / GYRO_LSB_PER_DPS - gyro_offset[0]
    gy = gy / GYRO_LSB_PER_DPS - gyro_offset[1]
    gz = gz / GYRO_LSB_PER_DPS - gyro_offset[2]
    return ax, ay, az, gx, gy, gz


def extraer_caracteristicas(ventana):
    datos = np.array(ventana, dtype=float)
    feats = []
    for i in range(datos.shape[1]):
        s = datos[:, i]
        feats += [
            np.mean(s),
            np.std(s),
            np.min(s),
            np.max(s),
            np.max(s) - np.min(s),
            np.mean(s ** 2),
        ]
    return np.array(feats, dtype=float).reshape(1, -1)


def predecir_mlp_con_confianza(modelos, features):
    """
    El MLP devuelve probabilidades por clase. Si la clase ganadora
    no supera MLP_CONFIDENCE_THRESHOLD, se asume 'Sentado quieto'
    porque el sensor está en un estado fuera de la distribución de caídas.
    """
    scaler = modelos["scaler"]
    mlp = modelos["mlp"]
    nombres_clases = modelos["nombres_clases"]
    features_sc = scaler.transform(features)
    probs = mlp.predict_proba(features_sc)[0]
    idx_max = int(np.argmax(probs))
    conf = probs[idx_max]
    if conf < MLP_CONFIDENCE_THRESHOLD:
        return nombres_clases[3], conf  # forzar "Sentado quieto"
    return nombres_clases[idx_max], conf


def predecir_arbol(modelos, features):
    nombres_clases = modelos["nombres_clases"]
    pred = modelos["arbol_decision"].predict(features)[0]
    return nombres_clases[int(pred)]


def predecir_rf(modelos, features):
    nombres_clases = modelos["nombres_clases"]
    pred = modelos["random_forest"].predict(features)[0]
    return nombres_clases[int(pred)]


def modo_diagnose(modelos, features):
    """
    Muestra las features en crudo, normalizadas, y las probabilidades
    del MLP para cada clase. Úsalo para entender por qué el MLP falla.
    """
    nombres_clases = modelos["nombres_clases"]
    scaler = modelos["scaler"]
    mlp = modelos["mlp"]

    cols = ["ax1", "ay1", "az1", "gx", "gy", "gz"]
    stat_names = ["mean", "std", "min", "max", "range", "mean_sq"]

    print("\n── Features brutas ────────────────────────────────")
    for i, col in enumerate(cols):
        base = i * 6
        vals = features[0, base:base + 6]
        vals_str = "  ".join(f"{s}={v:+.4f}" for s, v in zip(stat_names, vals))
        print(f"  {col}: {vals_str}")

    features_sc = scaler.transform(features)
    print("\n── Features normalizadas (entrada al MLP) ─────────")
    for i, col in enumerate(cols):
        base = i * 6
        vals = features_sc[0, base:base + 6]
        vals_str = "  ".join(f"{s}={v:+.4f}" for s, v in zip(stat_names, vals))
        print(f"  {col}: {vals_str}")

    probs = mlp.predict_proba(features_sc)[0]
    print("\n── Probabilidades MLP ─────────────────────────────")
    for nombre, prob in zip(nombres_clases, probs):
        barra = "█" * int(prob * 40)
        print(f"  {nombre:<25} {prob:.4f}  {barra}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inferencia SisFall con MPU6050 en Raspberry Pi")
    parser.add_argument("--model", choices=["arbol_decision", "random_forest", "mlp", "todos"],
                        default="todos")
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--debug", action="store_true",
                        help="Muestra ax/ay/az en tiempo real")
    parser.add_argument("--diagnose", action="store_true",
                        help="Muestra features y probabilidades del MLP en cada predicción")
    args = parser.parse_args()

    print(f"\n[INFO] Modelo: {args.model} | Ventana: {args.window:.1f}s | FS: {FS:.0f}Hz")
    if args.diagnose:
        print("[INFO] Modo diagnóstico activo — predicciones más lentas")

    modelos = joblib.load(MODEL_FILE)
    print(f"[INFO] Modelos cargados desde: {MODEL_FILE}")

    window_samples = int(args.window * FS)
    ventana = deque(maxlen=window_samples)
    periodo = 1.0 / FS

    with SMBus(I2C_BUS) as bus:
        init_mpu(bus)
        gyro_offset = calibrar_giroscopio(bus)

        print("\n[INFO] Iniciando lectura. Ctrl+C para detener.\n")
        next_predict_time = time.perf_counter() + args.window

        try:
            while True:
                t0 = time.perf_counter()
                muestra = read_mpu_units(bus, gyro_offset)
                ventana.append(muestra)

                if args.debug:
                    ax, ay, az = muestra[0], muestra[1], muestra[2]
                    mag = np.sqrt(ax**2 + ay**2 + az**2)
                    print(f"ax={ax:+.3f} ay={ay:+.3f} az={az:+.3f} |a|={mag:.3f}", end="\r")

                now = time.perf_counter()
                if len(ventana) == window_samples and now >= next_predict_time:
                    features = extraer_caracteristicas(ventana)

                    if args.diagnose:
                        modo_diagnose(modelos, features)
                    elif args.model == "todos":
                        arbol = predecir_arbol(modelos, features)
                        rf    = predecir_rf(modelos, features)
                        mlp_pred, conf = predecir_mlp_con_confianza(modelos, features)
                        print(f"Árbol: {arbol} | RF: {rf} | MLP: {mlp_pred} (conf={conf:.2f})")
                    elif args.model == "mlp":
                        mlp_pred, conf = predecir_mlp_con_confianza(modelos, features)
                        print(f"MLP: {mlp_pred} (conf={conf:.2f})")
                    elif args.model == "arbol_decision":
                        print(f"Predicción: {predecir_arbol(modelos, features)}")
                    elif args.model == "random_forest":
                        print(f"Predicción: {predecir_rf(modelos, features)}")

                    next_predict_time = now + PREDICT_EVERY_SECONDS

                elapsed = time.perf_counter() - t0
                if elapsed < periodo:
                    time.sleep(periodo - elapsed)

        except KeyboardInterrupt:
            print("\n\n[INFO] Programa detenido.")


if __name__ == "__main__":
    main()