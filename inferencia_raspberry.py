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
    bus.write_byte_data(MPU_ADDR, 0x6B, 0x00) # Despertar
    time.sleep(0.1)
    bus.write_byte_data(MPU_ADDR, 0x1A, 0x03) # DLPF activado
    bus.write_byte_data(MPU_ADDR, 0x19, 4)    # 200 Hz
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x18) # Giroscopio ±2000 °/s
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x18) # Acelerómetro ±16 g
    time.sleep(0.1)

def read_mpu_units(bus, gyro_offset):
    try:
        d = bus.read_i2c_block_data(MPU_ADDR, 0x3B, 14)
        ax, ay, az = to_int16(d[0], d[1])/ACC_LSB_PER_G, to_int16(d[2], d[3])/ACC_LSB_PER_G, to_int16(d[4], d[5])/ACC_LSB_PER_G
        gx, gy, gz = to_int16(d[8], d[9])/GYRO_LSB_PER_DPS, to_int16(d[10], d[11])/GYRO_LSB_PER_DPS, to_int16(d[12], d[13])/GYRO_LSB_PER_DPS
        # NOTA: Si tus ejes físicos no coinciden con SisFall, intercámbialos aquí
        return ax, ay, az, gx - gyro_offset[0], gy - gyro_offset[1], gz - gyro_offset[2]
    except OSError:
        return None # Evita que el programa crashee si falla el I2C

def calibrar_giroscopio(bus, muestras=300):
    print("[INFO] Calibrando giroscopio (deja el sensor quieto)...")
    g_vals = []
    for _ in range(muestras):
        d = read_mpu_units(bus, [0,0,0])
        if d: g_vals.append(d[3:6])
        time.sleep(1.0 / FS)
    offset = np.mean(g_vals, axis=0)
    print(f"[INFO] Offset gyro: gx={offset[0]:.3f}, gy={offset[1]:.3f}, gz={offset[2]:.3f} °/s")
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
    parser.add_argument("--model", choices=["arbol_decision", "random_forest", "mlp", "todos"], default="todos")
    args = parser.parse_args()

    datos = joblib.load("modelos_sisfall_sentado.pkl")
    modelos, scaler, clases = datos["modelos"], datos["scaler"], datos["nombres_clases"]
    ventana = deque(maxlen=int(WINDOW_SECONDS * FS))

    with SMBus(I2C_BUS) as bus:
        init_mpu(bus)
        offset = calibrar_giroscopio(bus)
        print("\n[INFO] Lectura en tiempo real. Presiona Ctrl+C para salir.")
        
        next_sample = time.perf_counter()
        next_predict = time.perf_counter() + WINDOW_SECONDS

        try:
            while True:
                now = time.perf_counter()
                if now >= next_sample:
                    muestra = read_mpu_units(bus, offset)
                    if muestra: ventana.append(muestra)
                    next_sample += 1.0 / FS

                if len(ventana) == ventana.maxlen and now >= next_predict:
                    # NORMALIZACIÓN GLOBAL PARA TODOS LOS MODELOS
                    f_sc = scaler.transform(extraer_caracteristicas(ventana))
                    
                    if args.model == "todos":
                        p_arb = clases[int(modelos["arbol_decision"].predict(f_sc)[0])]
                        p_rf = clases[int(modelos["random_forest"].predict(f_sc)[0])]
                        p_mlp = clases[int(modelos["mlp"].predict(f_sc)[0])]
                        print(f"Árbol: {p_arb:15} | RF: {p_rf:15} | MLP: {p_mlp}")
                    else:
                        print(f"Predicción: {clases[int(modelos[args.model].predict(f_sc)[0])]}")
                    
                    next_predict = now + PREDICT_EVERY_SECONDS
                time.sleep(0.0005)
        except KeyboardInterrupt:
            print("\n[INFO] Sistema detenido.")

if __name__ == "__main__": 
    main()