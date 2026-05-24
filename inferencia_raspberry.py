#!/usr/bin/env python3
"""
inferencia_raspberry.py

Inferencia en tiempo real con MPU6050 y modelos entrenados con SisFall.

Este programa:
1. Lee el MPU6050 por I2C.
2. Convierte aceleración a g y giroscopio a grados/s.
3. Forma una ventana temporal.
4. Detecta reposo físico.
5. Extrae las mismas características usadas en entrenamiento.
6. Ejecuta Árbol, Random Forest y/o MLP.
7. Muestra la predicción en consola.

No entrena modelos.
No genera gráficas.
No guarda imágenes.
"""

import argparse
import time
from collections import Counter, deque

import joblib
import numpy as np
from smbus2 import SMBus


# =========================================================
# Configuración general
# =========================================================

I2C_BUS = 1
MPU_ADDR = 0x68

MODEL_FILE = "modelos_sisfall_sentado.pkl"

FS = 200.0
WINDOW_SECONDS = 3.0
PREDICT_EVERY_SECONDS = 0.5

ACC_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4


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
# Funciones MPU6050
# =========================================================

def to_int16(msb, lsb):
    value = (msb << 8) | lsb

    if value >= 32768:
        value -= 65536

    return value


def init_mpu(bus):
    """
    Configura el MPU6050 en:
    - acelerómetro ±16 g
    - giroscopio ±2000 grados/s
    - frecuencia aproximada de 200 Hz
    """

    who = bus.read_byte_data(MPU_ADDR, REG_WHO_AM_I)
    print(f"[INFO] WHO_AM_I = 0x{who:02X}")

    if who not in (0x68, 0x70, 0x71, 0x72):
        print("[ADVERTENCIA] El WHO_AM_I no parece típico de MPU6050/MPU compatible.")

    # Despertar sensor
    bus.write_byte_data(MPU_ADDR, REG_PWR_MGMT_1, 0x00)
    time.sleep(0.1)

    # Filtro digital interno activado
    bus.write_byte_data(MPU_ADDR, REG_CONFIG, 0x03)

    # Con DLPF activo: base 1 kHz.
    # 1 kHz / (1 + 4) = 200 Hz
    bus.write_byte_data(MPU_ADDR, REG_SMPLRT_DIV, 4)

    # Giroscopio ±2000 °/s
    bus.write_byte_data(MPU_ADDR, REG_GYRO_CONFIG, 0x18)

    # Acelerómetro ±16 g
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
    """
    Calcula el offset del giroscopio con el sensor quieto.
    No se calibra el acelerómetro porque la gravedad debe conservarse.
    """

    print("[INFO] Calibrando giroscopio. Deja el sensor completamente quieto...")

    gx_vals = []
    gy_vals = []
    gz_vals = []

    for _ in range(muestras):
        _, _, _, gx, gy, gz = read_mpu_raw(bus)

        gx_vals.append(gx / GYRO_LSB_PER_DPS)
        gy_vals.append(gy / GYRO_LSB_PER_DPS)
        gz_vals.append(gz / GYRO_LSB_PER_DPS)

        time.sleep(1.0 / FS)

    offset = np.array([
        np.mean(gx_vals),
        np.mean(gy_vals),
        np.mean(gz_vals),
    ])

    print(
        f"[INFO] Offset gyro: "
        f"gx={offset[0]:.3f}, "
        f"gy={offset[1]:.3f}, "
        f"gz={offset[2]:.3f} °/s"
    )

    return offset


def corregir_ejes(ax, ay, az, gx, gy, gz):
    """
    Corrección de orientación de ejes.

    Por ahora se deja identidad:
        x_modelo = x_sensor
        y_modelo = y_sensor
        z_modelo = z_sensor

    Si después verificamos que tu MPU6050 está orientado distinto
    al sensor usado en SisFall, aquí se cambia el mapeo.
    """

    ax_modelo = ax
    ay_modelo = ay
    az_modelo = az

    gx_modelo = gx
    gy_modelo = gy
    gz_modelo = gz

    return ax_modelo, ay_modelo, az_modelo, gx_modelo, gy_modelo, gz_modelo


def read_mpu_units(bus, gyro_offset):
    ax, ay, az, gx, gy, gz = read_mpu_raw(bus)

    ax = ax / ACC_LSB_PER_G
    ay = ay / ACC_LSB_PER_G
    az = az / ACC_LSB_PER_G

    gx = gx / GYRO_LSB_PER_DPS - gyro_offset[0]
    gy = gy / GYRO_LSB_PER_DPS - gyro_offset[1]
    gz = gz / GYRO_LSB_PER_DPS - gyro_offset[2]

    return corregir_ejes(ax, ay, az, gx, gy, gz)


# =========================================================
# Reposo físico
# =========================================================

def calcular_estadisticas_movimiento(ventana):
    datos = np.array(ventana, dtype=float)

    ax = datos[:, 0]
    ay = datos[:, 1]
    az = datos[:, 2]
    gx = datos[:, 3]
    gy = datos[:, 4]
    gz = datos[:, 5]

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    acc_mean = np.mean(acc_mag)
    acc_std = np.std(acc_mag)
    gyro_mean = np.mean(gyro_mag)

    return acc_mean, acc_std, gyro_mean


def detectar_reposo(ventana):
    """
    Detecta si el sensor está quieto.

    Criterios:
    - La magnitud de aceleración está cerca de 1 g.
    - La variación de aceleración es baja.
    - La magnitud media del giroscopio es baja.
    """

    acc_mean, acc_std, gyro_mean = calcular_estadisticas_movimiento(ventana)

    esta_quieto = (
        0.85 <= acc_mean <= 1.15 and
        acc_std < 0.08 and
        gyro_mean < 8.0
    )

    return esta_quieto, acc_mean, acc_std, gyro_mean


# =========================================================
# Características del modelo
# =========================================================

def extraer_caracteristicas(ventana):
    """
    Extrae las mismas características usadas en entrenamiento.

    Orden:
        ax1, ay1, az1, gx, gy, gz

    Por cada señal:
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
# Predicción
# =========================================================

def obtener_etiqueta_sentado(modelos):
    nombres = modelos.get("nombres_clases", [])

    for nombre in nombres:
        if "Sentado" in nombre or "sentado" in nombre:
            return nombre

    return "Sentado quieto"


def predecir(modelos, features, modelo_nombre):
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
    return {
        "Árbol": predecir(modelos, features, "arbol_decision"),
        "RF": predecir(modelos, features, "random_forest"),
        "MLP": predecir(modelos, features, "mlp"),
    }


def decision_final(resultados):
    """
    Decide por mayoría.

    Si hay empate, se prioriza:
    1. Random Forest
    2. Árbol de Decisión
    3. MLP
    """

    votos = list(resultados.values())
    conteo = Counter(votos)

    clase_mas_votada, cantidad = conteo.most_common(1)[0]

    if cantidad >= 2:
        return clase_mas_votada

    return resultados["RF"]


# =========================================================
# Programa principal
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia SisFall con MPU6050 en Raspberry Pi"
    )

    parser.add_argument(
        "--model",
        choices=["arbol_decision", "random_forest", "mlp", "todos"],
        default="todos",
        help="Modelo a usar"
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
        help="Muestra estadísticas de aceleración y giroscopio"
    )

    args = parser.parse_args()

    print("\n════════════════════════════════════════════════════")
    print("  Inferencia en Raspberry Pi con MPU6050")
    print("════════════════════════════════════════════════════")
    print(f"[INFO] Archivo de modelos: {MODEL_FILE}")
    print(f"[INFO] Modelo seleccionado: {args.model}")
    print(f"[INFO] Ventana: {args.window:.2f} s")
    print(f"[INFO] Frecuencia objetivo: {FS:.1f} Hz")

    modelos = joblib.load(MODEL_FILE)
    etiqueta_sentado = obtener_etiqueta_sentado(modelos)

    print("[INFO] Modelos cargados correctamente.")
    print(f"[INFO] Etiqueta de reposo: {etiqueta_sentado}")

    window_samples = int(args.window * FS)
    ventana = deque(maxlen=window_samples)

    sample_period = 1.0 / FS

    with SMBus(I2C_BUS) as bus:
        init_mpu(bus)
        gyro_offset = calibrar_giroscopio(bus)

        print("\n[INFO] Iniciando inferencia en tiempo real...")
        print("[INFO] Presiona Ctrl+C para detener.\n")

        next_sample_time = time.perf_counter()
        next_predict_time = time.perf_counter() + args.window

        try:
            while True:
                now = time.perf_counter()

                if now >= next_sample_time:
                    muestra = read_mpu_units(bus, gyro_offset)
                    ventana.append(muestra)

                    next_sample_time += sample_period

                    if now - next_sample_time > sample_period:
                        next_sample_time = now + sample_period

                if len(ventana) == window_samples and now >= next_predict_time:
                    esta_quieto, acc_mean, acc_std, gyro_mean = detectar_reposo(ventana)

                    debug_text = ""

                    if args.debug:
                        debug_text = (
                            f" | acc_mean={acc_mean:.3f} g"
                            f" | acc_std={acc_std:.3f} g"
                            f" | gyro_mean={gyro_mean:.3f} °/s"
                        )

                    if esta_quieto:
                        if args.model == "todos":
                            print(
                                f"Árbol: {etiqueta_sentado} | "
                                f"RF: {etiqueta_sentado} | "
                                f"MLP: {etiqueta_sentado} | "
                                f"FINAL: {etiqueta_sentado}"
                                f"{debug_text}"
                            )
                        else:
                            print(f"Predicción: {etiqueta_sentado}{debug_text}")

                    else:
                        features = extraer_caracteristicas(ventana)

                        if args.model == "todos":
                            resultados = predecir_todos(modelos, features)
                            final = decision_final(resultados)

                            print(
                                f"Árbol: {resultados['Árbol']} | "
                                f"RF: {resultados['RF']} | "
                                f"MLP: {resultados['MLP']} | "
                                f"FINAL: {final}"
                                f"{debug_text}"
                            )
                        else:
                            pred = predecir(modelos, features, args.model)
                            print(f"Predicción: {pred}{debug_text}")

                    next_predict_time = now + PREDICT_EVERY_SECONDS

                time.sleep(0.0005)

        except KeyboardInterrupt:
            print("\n[INFO] Programa detenido por el usuario.")


if __name__ == "__main__":
    main()