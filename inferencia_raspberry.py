#!/usr/bin/env python3

import argparse, time, joblib
from collections import deque
import numpy as np
from smbus2 import SMBus

I2C_BUS = 1
MPU_ADDR = 0x68
MODEL_FILE = "modelos_sisfall_sentado.pkl"

FS = 200.0
WINDOW_SECONDS = 3.0
PREDICT_EVERY_SECONDS = 1.0

ACC_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4

REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75


def int16(msb, lsb):
    v = (msb << 8) | lsb
    return v - 65536 if v >= 32768 else v


def init_mpu(bus):
    print(f"[INFO] WHO_AM_I = 0x{bus.read_byte_data(MPU_ADDR, REG_WHO_AM_I):02X}")
    bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x00)
    time.sleep(0.1)
    bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x03)
    bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 4)
    bus.write_byte_data(MPU_ADDR, REG_GYRO_CONFIG, 0x18)
    bus.write_byte_data(MPU_ADDR, REG_ACCEL_CONFIG, 0x18)
    time.sleep(0.1)


def leer_raw(bus):
    d = bus.read_i2c_block_data(MPU_ADDR, REG_ACCEL_XOUT_H, 14)
    return (
        int16(d[0], d[1]), int16(d[2], d[3]), int16(d[4], d[5]),
        int16(d[8], d[9]), int16(d[10], d[11]), int16(d[12], d[13])
    )


def corregir_ejes(ax, ay, az, gx, gy, gz):
    return ax, ay, az, gx, gy, gz


def calibrar_gyro(bus, n=300):
    print("[INFO] Calibrando giroscopio. Deja el sensor quieto...")
    vals = []

    for _ in range(n):
        _, _, _, gx, gy, gz = leer_raw(bus)
        vals.append([gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS])
        time.sleep(1.0 / FS)

    off = np.mean(vals, axis=0)
    print(f"[INFO] Offset gyro: {off[0]:.2f}, {off[1]:.2f}, {off[2]:.2f} °/s")
    return off


def leer_mpu(bus, off):
    ax, ay, az, gx, gy, gz = leer_raw(bus)

    ax /= ACC_LSB_PER_G
    ay /= ACC_LSB_PER_G
    az /= ACC_LSB_PER_G

    gx = gx / GYRO_LSB_PER_DPS - off[0]
    gy = gy / GYRO_LSB_PER_DPS - off[1]
    gz = gz / GYRO_LSB_PER_DPS - off[2]

    return corregir_ejes(ax, ay, az, gx, gy, gz)


def extraer_caracteristicas(ventana):
    x = np.asarray(ventana, dtype=float)
    acc_mag = np.linalg.norm(x[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)
    x = np.column_stack([x, acc_mag, gyro_mag])

    feats = []
    for s in x.T:
        feats += [s.mean(), s.std(), s.min(), s.max(), s.max() - s.min(), np.sqrt(np.mean(s * s))]
    return np.array(feats, dtype=float).reshape(1, -1)


def sensor_quieto(ventana):
    x = np.asarray(ventana, dtype=float)
    acc_mag = np.linalg.norm(x[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)

    return 0.85 <= acc_mag.mean() <= 1.15 and acc_mag.std() < 0.08 and gyro_mag.mean() < 8.0


def predecir(pack, features, nombre):
    if nombre == "mlp":
        features = pack["scaler"].transform(features)
    return pack["nombres_clases"][int(pack[nombre].predict(features)[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["arbol_decision", "random_forest", "mlp", "todos"], default="todos")
    parser.add_argument("--sin-regla", action="store_true", help="Desactiva la regla de sensor quieto")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pack = joblib.load(MODEL_FILE)
    n = int(WINDOW_SECONDS * FS)
    ventana = deque(maxlen=n)

    print("\nInferencia SisFall con MPU6050")
    print(f"Modelo: {args.model}")
    print(f"Ventana: {WINDOW_SECONDS:.1f} s")
    print(f"Archivo: {MODEL_FILE}\n")

    with SMBus(I2C_BUS) as bus:
        init_mpu(bus)
        off = calibrar_gyro(bus)

        next_sample = time.perf_counter()
        next_pred = time.perf_counter() + WINDOW_SECONDS

        try:
            while True:
                now = time.perf_counter()

                if now >= next_sample:
                    muestra = leer_mpu(bus, off)
                    ventana.append(muestra)

                    if args.debug:
                        ax, ay, az, gx, gy, gz = muestra
                        acc = (ax * ax + ay * ay + az * az) ** 0.5
                        gyro = (gx * gx + gy * gy + gz * gz) ** 0.5
                        print(f"|acc|={acc:.3f} g |gyro|={gyro:.2f} °/s", end="\r")

                    next_sample += 1.0 / FS

                if len(ventana) == n and now >= next_pred:
                    if not args.sin_regla and sensor_quieto(ventana):
                        if args.model == "todos":
                            print("Árbol: Sentado quieto | RF: Sentado quieto | MLP: Sentado quieto")
                        else:
                            print("Predicción: Sentado quieto")
                    else:
                        features = extraer_caracteristicas(ventana)

                        if features.shape[1] != pack.get("feature_count", features.shape[1]):
                            raise ValueError("El modelo fue entrenado con otro número de características.")

                        if args.model == "todos":
                            a = predecir(pack, features, "arbol_decision")
                            r = predecir(pack, features, "random_forest")
                            m = predecir(pack, features, "mlp")
                            print(f"Árbol: {a} | RF: {r} | MLP: {m}")
                        else:
                            print(f"Predicción: {predecir(pack, features, args.model)}")

                    next_pred = now + PREDICT_EVERY_SECONDS

                time.sleep(0.0005)

        except KeyboardInterrupt:
            print("\n[INFO] Programa detenido.")


if __name__ == "__main__":
    main()