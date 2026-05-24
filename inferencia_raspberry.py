#!/usr/bin/env python3
"""
inferencia_raspberry.py

Inferencia en tiempo real con MPU6050 y modelos entrenados con SisFall.

Este archivo NO entrena modelos, NO genera gráficas y NO guarda imágenes.
Solo:
1. Lee el MPU6050 por I2C.
2. Convierte datos crudos a g y grados/s.
3. Forma una ventana temporal.
4. Extrae las mismas características usadas en entrenamiento.
5. Carga modelos_sisfall_sentado.pkl.
6. Ejecuta predicción en consola.
"""

import argparse
import time
from collections import deque

import joblib
import numpy as np
from smbus2 import SMBus


# =========================================================
# Configuración general
# =========================================================

I2C_BUS = 1
MPU_ADDR = 0x68

MODEL_FILE = "modelos_sisfall_sentado.pkl"

FS = 200.0                 # SisFall fue grabado a 200 Hz
WINDOW_SECONDS = 3.0       # Ventana inicial para inferencia
PREDICT_EVERY_SECONDS = 0.5

ACC_LSB_PER_G = 2048.0     # MPU6050 en ±16 g
GYRO_LSB_PER_DPS = 16.4    # MPU6050 en ±2000 °/s


# =========================================================
# Registros MPU6050
# =========================================================

REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75


# =========================================================
# Funciones de bajo nivel: MPU6050
# =========================================================

def to_int16(msb, lsb):
    """Convierte dos bytes a entero con signo de 16 bits."""
    value = (msb << 8) | lsb
    if value >= 32768:
        value -= 65536
    return value


def init_mpu(bus):
    """
    Configura el MPU6050 para trabajar de forma compatible con SisFall:
    - acelerómetro en ±16 g
    - giroscopio en ±2000 °/s
    - frecuencia aproximada de 200 Hz
    """

    who = bus.read_byte_data(MPU_ADDR, REG_WHO_AM_I)
    print(f"[INFO] WHO_AM_I = 0x{who:02X}")

    # Despertar MPU6050
    bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x00)
    time.sleep(0.1)

    # DLPF activado. Con DLPF activo, la base de muestreo es 1 kHz.
    bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x03)

    # 1 kHz / (1 + 4) = 200 Hz
    bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 4)

    # Giroscopio ±2000 °/s
    bus.write_byte_data(MPU_ADDR, REG_GYRO_CONFIG, 0x18)

    # Acelerómetro ±16 g
    bus.write_byte_data(MPU_ADDR, REG_ACCEL_CONFIG, 0x18)

    time.sleep(0.1)


def read_mpu_raw(bus):
    """Lee acelerómetro y giroscopio crudos del MPU6050."""
    data = bus.read_i2c_block_data(MPU_ADDR, REG_ACCEL_XOUT_H, 14)

    ax = to_int16(data[0], data[1])
    ay = to_int16(data[2], data[3])
    az = to_int16(data[4], data[5])

    gx = to_int16(data[8], data[9])
    gy = to_int16(data[10], data[11])
    gz = to_int16(data[12], data[13])

    return ax, ay, az, gx, gy, gz


def read_mpu_units(bus, gyro_offset):
    """
    Lee el MPU6050 y convierte:
    - acelerómetro a g
    - giroscopio a grados/s
    """

    ax, ay, az, gx, gy, gz = read_mpu_raw(bus)

    ax = ax / ACC_LSB_PER_G
    ay = ay / ACC_LSB_PER_G
    az = az / ACC_LSB_PER_G

    gx = gx / GYRO_LSB_PER_DPS - gyro_offset[0]
    gy = gy / GYRO_LSB_PER_DPS - gyro_offset[1]
    gz = gz / GYRO_LSB_PER_DPS - gyro_offset[2]

    return corregir_ejes(ax, ay, az, gx, gy, gz)


def corregir_ejes(ax, ay, az, gx, gy, gz):
    """
    Ajusta los ejes del MPU6050 para parecerse al orden usado en entrenamiento.

    Tu modelo espera:
        ax1, ay1, az1, gx, gy, gz

    Por ahora se deja identidad:
        x_modelo = x_sensor
        y_modelo = y_sensor
        z_modelo = z_sensor

    Si al probar el sensor quieto ves que la gravedad no queda en el eje esperado,
    modifica esta función.
    """

    ax_modelo = ax
    ay_modelo = ay
    az_modelo = az

    gx_modelo = gx
    gy_modelo = gy
    gz_modelo = gz

    return ax_modelo, ay_modelo, az_modelo, gx_modelo, gy_modelo, gz_modelo


def calibrar_giroscopio(bus, muestras=300):
    """
    Calcula el offset del giroscopio con el sensor quieto.
    No se calibra el acelerómetro porque la gravedad debe conservarse.
    """

    print("[INFO] Calibrando giroscopio. Deja el sensor quieto...")

    gx_vals, gy_vals, gz_vals = [], [], []

    for _ in range(muestras):
        _, _, _, gx, gy, gz = read_mpu_raw(bus)

        gx_vals.append(gx / GYRO_LSB_PER_DPS)
        gy_vals.append(gy / GYRO_LSB_PER_DPS)
        gz_vals.append(gz / GYRO_LSB_PER_DPS)

        time.sleep(1.0 / FS)

    offset = np.array([
        np.mean(gx_vals),
        np.mean(gy_vals),
        np.mean(gz_vals)
    ])

    print(f"[INFO] Offset gyro: gx={offset[0]:.3f}, gy={offset[1]:.3f}, gz={offset[2]:.3f} °/s")

    return offset


# =========================================================
# Características: deben coincidir con tu entrenamiento
# =========================================================

def extraer_caracteristicas(ventana):
    """
    Extrae exactamente las mismas características usadas en tu código de entrenamiento.

    Orden de columnas:
        ax1, ay1, az1, gx, gy, gz

    Por cada columna:
        mean, std, min, max, range, mean_square
    """

    datos = np.array(ventana, dtype=float)
    caracteristicas = []

    for i in range(datos.shape[1]):
        s = datos[:, i]

        caracteristicas += [
            np.mean(s),
            np.std(s),
            np.min(s),
            np.max(s),
            np.max(s) - np.min(s),
            np.mean(s ** 2),
        ]

    return np.array(caracteristicas, dtype=float).reshape(1, -1)


# =========================================================
# Inferencia
# =========================================================

def predecir(modelos, features, modelo_nombre):
    """Ejecuta predicción usando el modelo seleccionado."""

    nombres_clases = modelos["nombres_clases"]

    if modelo_nombre == "arbol_decision":
        modelo = modelos["arbol_decision"]
        pred = modelo.predict(features)[0]

    elif modelo_nombre == "random_forest":
        modelo = modelos["random_forest"]
        pred = modelo.predict(features)[0]

    elif modelo_nombre == "mlp":
        modelo = modelos["mlp"]
        scaler = modelos["scaler"]
        features_sc = scaler.transform(features)
        pred = modelo.predict(features_sc)[0]

    else:
        raise ValueError("Modelo no válido.")

    return nombres_clases[int(pred)]


def predecir_todos(modelos, features):
    """Ejecuta los tres modelos y devuelve sus predicciones."""

    resultados = {}

    resultados["Árbol"] = predecir(modelos, features, "arbol_decision")
    resultados["Random Forest"] = predecir(modelos, features, "random_forest")
    resultados["MLP"] = predecir(modelos, features, "mlp")

    return resultados


def main():
    parser = argparse.ArgumentParser(description="Inferencia SisFall con MPU6050 en Raspberry Pi")

    parser.add_argument(
        "--model",
        choices=["arbol_decision", "random_forest", "mlp", "todos"],
        default="arbol_decision",
        help="Modelo a usar para inferencia"
    )

    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
        help="Tamaño de ventana en segundos"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra magnitud de aceleración para verificar escala"
    )

    args = parser.parse_args()

    print("\n════════════════════════════════════════════════════")
    print("  Inferencia en Raspberry Pi con MPU6050")
    print("════════════════════════════════════════════════════")
    print(f"[INFO] Modelo seleccionado: {args.model}")
    print(f"[INFO] Ventana: {args.window:.2f} s")
    print(f"[INFO] Frecuencia objetivo: {FS:.1f} Hz")

    modelos = joblib.load(MODEL_FILE)
    print(f"[INFO] Modelos cargados desde: {MODEL_FILE}")

    window_samples = int(args.window * FS)
    ventana = deque(maxlen=window_samples)

    with SMBus(I2C_BUS) as bus:
        init_mpu(bus)
        gyro_offset = calibrar_giroscopio(bus)

        print("\n[INFO] Iniciando lectura en tiempo real...")
        print("[INFO] Presiona Ctrl+C para detener.\n")

        next_sample_time = time.perf_counter()
        next_predict_time = time.perf_counter() + args.window

        try:
            while True:
                now = time.perf_counter()

                if now >= next_sample_time:
                    muestra = read_mpu_units(bus, gyro_offset)
                    ventana.append(muestra)

                    if args.debug:
                        ax, ay, az, gx, gy, gz = muestra
                        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
                        print(
                            f"ax={ax:+.3f} g  ay={ay:+.3f} g  az={az:+.3f} g  "
                            f"|acc|={acc_mag:.3f} g",
                            end="\r"
                        )

                    next_sample_time += 1.0 / FS

                if len(ventana) == window_samples and now >= next_predict_time:
                    features = extraer_caracteristicas(ventana)

                    if args.model == "todos":
                        resultados = predecir_todos(modelos, features)
                        print(
                            f"Árbol: {resultados['Árbol']} | "
                            f"RF: {resultados['Random Forest']} | "
                            f"MLP: {resultados['MLP']}"
                        )
                    else:
                        pred = predecir(modelos, features, args.model)
                        print(f"Predicción: {pred}")

                    next_predict_time = now + PREDICT_EVERY_SECONDS

                time.sleep(0.0005)

        except KeyboardInterrupt:
            print("\n\n[INFO] Programa detenido por el usuario.")


if __name__ == "__main__":
    main()