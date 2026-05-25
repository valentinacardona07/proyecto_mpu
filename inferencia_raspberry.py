#!/usr/bin/env python3

import argparse
import time
from collections import deque

import joblib
import numpy as np
import RPi.GPIO as GPIO
from smbus2 import SMBus


# =========================================================
# Configuración general
# =========================================================

I2C_BUS = 1
MPU_ADDR = 0x68

MODEL_FILE = "modelos_sisfall_sentado.pkl"

FS = 200.0
WINDOW_SECONDS = 3.0
PREDICT_EVERY_SECONDS = 1.0

ACC_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4


# =========================================================
# LED RGB
# =========================================================
# Conexión recomendada:
# Rojo  -> GPIO17, pin físico 11, con resistencia de 220 Ω o 330 Ω
# Verde -> GPIO27, pin físico 13, con resistencia de 220 Ω o 330 Ω
# Azul  -> GPIO22, pin físico 15, con resistencia de 220 Ω o 330 Ω
# Común -> GND, pin físico 6, si el LED es de cátodo común

PIN_R = 17
PIN_G = 27
PIN_B = 22

# False: LED RGB de cátodo común.
# True: LED RGB de ánodo común.
LED_COMMON_ANODE = False


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
# LED RGB
# =========================================================

def setup_led_rgb():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(PIN_R, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_G, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_B, GPIO.OUT, initial=GPIO.LOW)

    apagar_led()


def escribir_rgb(r, g, b):
    if LED_COMMON_ANODE:
        r, g, b = not r, not g, not b

    GPIO.output(PIN_R, GPIO.HIGH if r else GPIO.LOW)
    GPIO.output(PIN_G, GPIO.HIGH if g else GPIO.LOW)
    GPIO.output(PIN_B, GPIO.HIGH if b else GPIO.LOW)


def apagar_led():
    escribir_rgb(False, False, False)


def actualizar_led_por_estado(estado):
    if "F13" in estado:
        escribir_rgb(True, False, False)        # Rojo: caída adelante
    elif "F14" in estado:
        escribir_rgb(False, False, True)        # Azul: caída atrás
    elif "F15" in estado:
        escribir_rgb(True, True, False)         # Amarillo: caída lateral
    elif "Sentado" in estado:
        escribir_rgb(False, True, False)        # Verde: sentado quieto
    else:
        apagar_led()


def estado_principal(resultados):
    votos = list(resultados.values())

    for estado in votos:
        if votos.count(estado) >= 2:
            return estado

    return resultados["Random Forest"]


# =========================================================
# Funciones de bajo nivel: MPU6050
# =========================================================

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


def corregir_ejes(ax, ay, az, gx, gy, gz):
    return ax, ay, az, gx, gy, gz


def read_mpu_units(bus, gyro_offset):
    ax, ay, az, gx, gy, gz = read_mpu_raw(bus)

    ax = ax / ACC_LSB_PER_G
    ay = ay / ACC_LSB_PER_G
    az = az / ACC_LSB_PER_G

    gx = gx / GYRO_LSB_PER_DPS - gyro_offset[0]
    gy = gy / GYRO_LSB_PER_DPS - gyro_offset[1]
    gz = gz / GYRO_LSB_PER_DPS - gyro_offset[2]

    return corregir_ejes(ax, ay, az, gx, gy, gz)


def calibrar_giroscopio(bus, muestras=300):
    print("[INFO] Calibrando giroscopio. Deja el sensor quieto...")

    valores = []

    for _ in range(muestras):
        _, _, _, gx, gy, gz = read_mpu_raw(bus)

        valores.append([
            gx / GYRO_LSB_PER_DPS,
            gy / GYRO_LSB_PER_DPS,
            gz / GYRO_LSB_PER_DPS
        ])

        time.sleep(1.0 / FS)

    offset = np.mean(valores, axis=0)

    print(
        f"[INFO] Offset gyro: "
        f"gx={offset[0]:.3f}, gy={offset[1]:.3f}, gz={offset[2]:.3f} °/s"
    )

    return offset


# =========================================================
# Características
# =========================================================

def extraer_caracteristicas(ventana):
    x = np.asarray(ventana, dtype=float)

    acc_mag = np.linalg.norm(x[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)

    x = np.column_stack([x, acc_mag, gyro_mag])

    caracteristicas = []

    for s in x.T:
        caracteristicas += [
            np.mean(s),
            np.std(s),
            np.min(s),
            np.max(s),
            np.max(s) - np.min(s),
            np.sqrt(np.mean(s * s)),
        ]

    return np.array(caracteristicas, dtype=float).reshape(1, -1)


def sensor_quieto(ventana):
    x = np.asarray(ventana, dtype=float)

    acc_mag = np.linalg.norm(x[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)

    return (
        0.85 <= np.mean(acc_mag) <= 1.15
        and np.std(acc_mag) < 0.08
        and np.mean(gyro_mag) < 8.0
    )


# =========================================================
# Inferencia
# =========================================================

def predecir(modelos, features, modelo_nombre):
    nombres_clases = modelos["nombres_clases"]

    if modelo_nombre == "arbol_decision":
        pred = modelos["arbol_decision"].predict(features)[0]

    elif modelo_nombre == "random_forest":
        pred = modelos["random_forest"].predict(features)[0]

    elif modelo_nombre == "mlp":
        features_sc = modelos["scaler"].transform(features)
        pred = modelos["mlp"].predict(features_sc)[0]

    else:
        raise ValueError("Modelo no válido.")

    return nombres_clases[int(pred)]


def predecir_todos(modelos, features):
    return {
        "Árbol": predecir(modelos, features, "arbol_decision"),
        "Random Forest": predecir(modelos, features, "random_forest"),
        "MLP": predecir(modelos, features, "mlp"),
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Inferencia SisFall con MPU6050 y LED RGB")

    parser.add_argument(
        "--model",
        choices=["arbol_decision", "random_forest", "mlp", "todos"],
        default="todos",
        help="Modelo a usar para inferencia"
    )

    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
        help="Tamaño de ventana en segundos"
    )

    parser.add_argument(
        "--sin-regla",
        action="store_true",
        help="Desactiva la regla de sensor quieto"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra magnitud de aceleración y giroscopio"
    )

    args = parser.parse_args()

    print("\n════════════════════════════════════════════════════")
    print("  Inferencia en Raspberry Pi con MPU6050 + LED RGB")
    print("════════════════════════════════════════════════════")
    print(f"[INFO] Modelo seleccionado: {args.model}")
    print(f"[INFO] Ventana: {args.window:.2f} s")
    print(f"[INFO] Frecuencia objetivo: {FS:.1f} Hz")

    modelos = joblib.load(MODEL_FILE)
    print(f"[INFO] Modelos cargados desde: {MODEL_FILE}")

    window_samples = int(args.window * FS)
    ventana = deque(maxlen=window_samples)

    setup_led_rgb()

    try:
        with SMBus(I2C_BUS) as bus:
            init_mpu(bus)
            gyro_offset = calibrar_giroscopio(bus)

            print("\n[INFO] Iniciando lectura en tiempo real...")
            print("[INFO] Presiona Ctrl+C para detener.\n")

            next_sample_time = time.perf_counter()
            next_predict_time = time.perf_counter() + args.window

            while True:
                now = time.perf_counter()

                if now >= next_sample_time:
                    muestra = read_mpu_units(bus, gyro_offset)
                    ventana.append(muestra)

                    if args.debug:
                        ax, ay, az, gx, gy, gz = muestra
                        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
                        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

                        print(
                            f"|acc|={acc_mag:.3f} g  "
                            f"|gyro|={gyro_mag:.2f} °/s",
                            end="\r"
                        )

                    next_sample_time += 1.0 / FS

                if len(ventana) == window_samples and now >= next_predict_time:
                    if not args.sin_regla and sensor_quieto(ventana):
                        estado = "Sentado quieto"
                        actualizar_led_por_estado(estado)

                        if args.model == "todos":
                            print(
                                "Árbol: Sentado quieto | "
                                "RF: Sentado quieto | "
                                "MLP: Sentado quieto | "
                                "LED: Sentado quieto"
                            )
                        else:
                            print("Predicción: Sentado quieto")

                    else:
                        features = extraer_caracteristicas(ventana)

                        if features.shape[1] != modelos.get("feature_count", features.shape[1]):
                            raise ValueError(
                                "El modelo fue entrenado con un número diferente de características."
                            )

                        if args.model == "todos":
                            resultados = predecir_todos(modelos, features)
                            estado = estado_principal(resultados)
                            actualizar_led_por_estado(estado)

                            print(
                                f"Árbol: {resultados['Árbol']} | "
                                f"RF: {resultados['Random Forest']} | "
                                f"MLP: {resultados['MLP']} | "
                                f"LED: {estado}"
                            )

                        else:
                            estado = predecir(modelos, features, args.model)
                            actualizar_led_por_estado(estado)
                            print(f"Predicción: {estado}")

                    next_predict_time = now + PREDICT_EVERY_SECONDS

                time.sleep(0.0005)

    except KeyboardInterrupt:
        print("\n\n[INFO] Programa detenido por el usuario.")

    finally:
        apagar_led()
        GPIO.cleanup()


if __name__ == "__main__":
    main()