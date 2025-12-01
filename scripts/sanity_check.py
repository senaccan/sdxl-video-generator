import torch, os, yaml, pathlib
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")

assert torch.cuda.is_available(), "CUDA GPU görülmüyor!"
print("CUDA device:", torch.cuda.get_device_name(0))

with open(ROOT/"configs/paths.yaml") as f:
    paths = yaml.safe_load(f)
need = [
    paths["sdxl"]["base"],
    paths["sdxl"]["refiner"],
]
for p in need:
    assert pathlib.Path(p).exists(), f"Model yolu eksik: {p}"
print("Model yolları OK.")

print("Sanity check geçti.")
