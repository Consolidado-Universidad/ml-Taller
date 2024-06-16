import os
import glob
from sklearn.model_selection import train_test_split
import json

def guardar_diccionario(dic, nombre_archivo, carpeta_destino="../rutas_dataset"):
    if not os.path.exists(carpeta_destino):
        os.mkdir(carpeta_destino)
    file = open(carpeta_destino+"/"+nombre_archivo+".json", "w")
    json.dump(dic, file)
    file.close()

dir_path = "/proyectos/taller_temporal/dataset/data/Potato___Early_blight/"

etiquetas = glob.glob(dir_path + "/*.json")
print(etiquetas)
ids = []
rutas = {}

idx = 0
for etiqueta in etiquetas:
    print(etiqueta)
    nombre = etiqueta.split("/")[-1].replace(".json",".JPG")
    print(os.path.join(dir_path,nombre))
    print({"imagen":os.path.join(dir_path,nombre),"etiqueta":etiqueta})
    rutas["id-"+str(idx)] = {"imagen":os.path.join(dir_path,nombre),"etiqueta":etiqueta}
    ids.append("id-"+str(idx))
    idx = idx + 1

train, test = train_test_split(ids, test_size=0.2, random_state=42)

dataset = {"train":train,"test":test}

guardar_diccionario(rutas, "rutas_dataset")
guardar_diccionario(dataset, "id_dataset")


