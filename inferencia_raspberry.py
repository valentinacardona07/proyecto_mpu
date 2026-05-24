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

REG_SMPLRT_DIV = 0x19
REG_CONFIG     = 0x1A
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1   = 0x6B
REG_WHO_AM_I     = 0x75


def to_int16(msb, lsb):
    value = (msb << 8) | lsb
    return value - 65536 if value >= 32768 else value


def init_mpu(bus):
    who = bus.read_byte_data(MPU_ADDR, REG_WHO_AM_I)
    print(f"[INFO] WHO_AM_I = 0x{who:02X}")
    bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x00)
    time.sleep(0.1)
    bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x03)       # DLPF activo → base 1 kHz
    bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 4)      # 1000/(1+4) = 200 Hz
    bus.write_byte_data(MPU_ADDR, REG_GYRO_CONFIG, 0x18)  # ±2000 °/s
    bus.write_byte_data(MPU_ADDR, REG_ACCEL_CONFIG, 0x18) # ±16 g
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


def predecir_todos(modelos, features):
    scaler = modelos["scaler"]
    features_sc = scaler.transform(features)
    nombres_clases = modelos["nombres_clases"]
    arbol = modelos["arbol_decision"].predict(features)[0]
    rf    = modelos["random_forest"].predict(features)[0]
    mlp   = modelos["mlp"].predict(features_sc)[0]
    return {
        "Árbol": nombres_clases[int(arbol)],
        "Random Forest": nombres_clases[int(rf)],
        "MLP": nombres_clases[int(mlp)],
    }


def predecir(modelos, features, modelo_nombre):
    nombres_clases = modelos["nombres_clases"]
    if modelo_nombre == "mlp":
        features = modelos["scaler"].transform(features)
        pred = modelos["mlp"].predict(features)[0]
    elif modelo_nombre == "random_forest":
        pred = modelos["random_forest"].predict(features)[0]
    elif modelo_nombre == "arbol_decision":
        pred = modelos["arbol_decision"].predict(features)[0]
    else:
        raise ValueError(f"Modelo no válido: {modelo_nombre}")
    return nombres_clases[int(pred)]


def main():
    parser = argparse.ArgumentParser(description="Inferencia SisFall con MPU6050 en Raspberry Pi")
    parser.add_argument("--model", choices=["arbol_decision", "random_forest", "mlp", "todos"],
                        default="todos")
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n[INFO] Modelo: {args.model} | Ventana: {args.window:.1f}s | FS: {FS:.0f}Hz")

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
                    print(f"ax={ax:+.3f} ay={ay:+.3f} az={az:+.3f} |a|={np.sqrt(ax**2+ay**2+az**2):.3f}", end="\r")

                now = time.perf_counter()
                if len(ventana) == window_samples and now >= next_predict_time:
                    features = extraer_caracteristicas(ventana)
                    if args.model == "todos":
                        r = predecir_todos(modelos, features)
                        print(f"Árbol: {r['Árbol']} | RF: {r['Random Forest']} | MLP: {r['MLP']}")
                    else:
                        print(f"Predicción: {predecir(modelos, features, args.model)}")
                    next_predict_time = now + PREDICT_EVERY_SECONDS

                elapsed = time.perf_counter() - t0
                if elapsed < periodo:
                    time.sleep(periodo - elapsed)

        except KeyboardInterrupt:
            print("\n\n[INFO] Programa detenido.")


if __name__ == "__main__":
    main()