import os
import shutil
import math
from sklearn.model_selection import train_test_split
import glob

dir_origen = "archive/PotatoPlants"
dir_destino = "dataset"

if not os.path.exists(dir_destino):
    os.mkdir(dir_destino)

dir_destino = dir_destino+"/data/"

sub_dirs = os.listdir(dir_origen)

for sub_dir in sub_dirs:
    if not os.path.exists(os.path.join(dir_destino, sub_dir)):
        os.mkdir(os.path.join(dir_destino, sub_dir))

    imagenes = glob.glob(os.path.join(dir_origen, sub_dir, "*.JPG"))
    print(len(imagenes))

    _ , porcentaje_imagenes = train_test_split(imagenes, test_size=0.2, random_state=42)
    #
    for ruta_imagenes in porcentaje_imagenes:
        print("Train: ", ruta_imagenes)
        nombre = ruta_imagenes.split("/")[-1]
        shutil.copy(ruta_imagenes, os.path.join(dir_destino, sub_dir,nombre))