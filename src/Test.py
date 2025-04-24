import serial
import time

# Remplace par ton port (ex: "COM3" sous Windows ou "/dev/cu.usbmodem14101" sous macOS)
PORT = "/dev/cu.usbmodem101"
BAUDRATE = 9600

try:
    print("🔌 Connexion au port série...")
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Laisse le temps à l’Arduino de reset après ouverture

    print("✅ Connecté. Lecture des données série...\n")
    start = time.time()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            print(f"📡 {line}")

        if time.time() - start > 10:  # Stop après 10 secondes
            break

    ser.close()
    print("🛑 Fin de test.")
except Exception as e:
    print(f"❌ Erreur : {e}")
