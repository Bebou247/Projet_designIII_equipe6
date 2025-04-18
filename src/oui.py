import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 📁 Chemin vers le CSV
csv_path = Path(__file__).parent.parent / "data" / "Hauteur 1.csv"

# 📄 Chargement du fichier
df = pd.read_csv(csv_path)

# 🕒 Conversion du timestamp si présent
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    # sinon, on crée un axe temps artificiel
    df["timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="200ms")

# 🎯 Liste des thermistances à tracer
thermistances = [col for col in df.columns if col.startswith("R") and col != "R12"]

# 📈 Tracé
plt.figure(figsize=(12, 6))
for col in thermistances:
    plt.plot(df["timestamp"], df[col], label=col)

plt.xlabel("Temps")
plt.ylabel("Température (°C)")
plt.title("Évolution des températures des thermistances")
plt.legend(ncol=4)
plt.grid(True)
plt.tight_layout()
plt.show()
