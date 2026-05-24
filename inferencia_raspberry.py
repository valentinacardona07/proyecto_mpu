#!/usr/bin/env python3
import time, joblib, argparse
from collections import deque
import numpy as np
from smbus2 import SMBus

I2C_BUS, MPU_ADDR = 1, 0x68
FS, WINDOW_SECONDS, PREDICT_EVERY_SECONDS = 200.0, 3.0, 1
ACC_LSB_PER_G, GYRO_LSB_PER_DPS = 2048.0, 16.4

def to_int16(msb, lsb):
    v = (msb << 8) | lsb
    return v - 65536 if v >= 32768 else v

def init_mpu(bus):
    try:
        bus.write_byte_data(MPU_ADDR, 0x6B, 0x00) # Despertar sensor
        time.sleep(0.1)
        bus.write_byte_data(MPU_ADDR, 0x1A, 0x03) # DLPF habilitado
        bus.write_byte_data(MPU_ADDR, 0x19, 4)    # Frecuencia interna a 200 Hz
        bus.write_byte_data(MPU_ADDR, 0x1B, 0x18) # Gyro ±2000 °/s
        bus.write_byte_data(MPU_ADDR, 0x1C, 0x18) # Accel ±16 g
        time.sleep(0.1)
    except OSError:
        print("[ERROR] No se pudo inicializar el MPU6050. Verifica las conexiones físicas I2C.")

def read_mpu_units(bus, gyro_offset):
    try:
        d = bus.read_i2c_block_data(MPU_ADDR, 0x3B, 14)
        ax = to_int16(d[0], d[1]) / ACC_LSB_PER_G
        ay = to_int16(d[2], d[3]) / ACC_LSB_PER_G
        az = to_int16(d[4], d[5]) / ACC_LSB_PER_G
        gx = (to_int16(d[8], d[9]) / GYRO_LSB_PER_DPS) - gyro_offset[0]
        gy = (to_int16(d[10], d[11]) / GYRO_LSB_PER_DPS) - gyro_offset[1]
        gz = (to_int16(d[12], d[13]) / GYRO_LSB_PER_DPS) - gyro_offset[2]
        return ax, ay, az, gx, gy, gz
    except OSError:
        return None

def calibrar_giroscopio(bus, muestras=300):
    print("[INFO] Calibrando giroscopio. Mantén el dispositivo inmóvil...")
    g_vals = []
    for _ in range(muestras):
        d = read_mpu_units(bus, [0.0, 0.0, 0.0])
        if d: g_vals.append(d[3:6])
        time.sleep(1.0 / FS)
    if not g_vals:
        print("[WARN] Calibración fallida. Usando offset cero por defecto.")
        return np.array([0.0, 0.0, 0.0])
    offset = np.mean(g_vals, axis=0)
    print(f"[INFO] Calibración completada. Offsets: gx={offset[0]:.2f}, gy={offset[1]:.2f}, gz={offset[2]:.2f}")
    return offset

def extraer_caracteristicas(ventana):
    d = np.array(ventana, dtype=float)
    feats = []
    for i in range(d.shape[1]):
        s = d[:, i]
        feats.extend([np.mean(s), np.std(s), np.min(s), np.max(s), np.max(s) - np.min(s), np.mean(s ** 2)])
    return np.array(feats).reshape(1, -1)

def main():
    parser = argparse.ArgumentParser()
    # Retornamos a tus opciones originales de consola ('arbol', 'rf', 'mlp', 'todos')
    parser.add_argument("--model", choices=["arbol", "rf", "mlp", "todos"], default="todos")
    args = parser.parse_args()

    try:
        datos = joblib.load("modelos_sisfall_sentado.pkl")
        modelos, scaler, clases = datos["modelos"], datos["scaler"], datos["nombres_clases"]
    except FileNotFoundError:
        print("[ERROR] No se encontró 'modelos_sisfall_sentado.pkl'. Ejecuta primero sisfall_clasificacion.py.")
        return

    ventana = deque(maxlen=int(WINDOW_SECONDS * FS))

    with SMBus(I2C_BUS) as bus:
        init_mpu(bus)
        offset = calibrar_giroscopio(bus)
        print("\n[INFO] Inferencia activa. Presiona Ctrl+C para finalizar.\n")
        
        next_sample = time.perf_counter()
        next_predict = time.perf_counter() + WINDOW_SECONDS

        try:
            while True:
                now = time.perf_counter()
                
                # Anti-drift: si el tiempo real se desfasa demasiado por retraso del bus, resincronizar
                if now > next_sample + 0.5:
                    next_sample = now

                if now >= next_sample:
                    muestra = read_mpu_units(bus, offset)
                    if muestra: 
                        ventana.append(muestra)
                    next_sample += 1.0 / FS

                if len(ventana) == ventana.maxlen and now >= next_predict:
                    # Extracción y escalado robusto
                    f_sc = scaler.transform(extraer_caracteristicas(ventana))
                    
                    if args.model == "todos":
                        p_arb = clases[int(modelos["arbol"].predict(f_sc)[0])]
                        p_rf = clases[int(modelos["rf"].predict(f_sc)[0])]
                        p_mlp = clases[int(modelos["mlp"].predict(f_sc)[0])]
                        print(f"Árbol: {p_arb:22} | RF: {p_rf:22} | MLP: {p_mlp}")
                    else:
                        pred = clases[int(modelos[args.model].predict(f_sc)[0])]
                        print(f"Predicción ({args.model}): {pred}")
                    
                    next_predict = now + PREDICT_EVERY_SECONDS
                time.sleep(0.0005)
        except KeyboardInterrupt:
            print("\n[INFO] Ejecución interrumpida por el usuario. Saliendo de forma limpia.")

if __name__ == "__main__": 
    main()