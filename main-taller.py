import os
from src.dataloader_taller import DataLoader
import json
from model.modelo import unet
from  keras import optimizers

import tensorflow as tf
from src.metricas import dice_coef, dice_loss
import numpy as np
import wandb
from wandb.keras import WandbCallback


def run(config):

    wandb.init(settings=wandb.Settings(start_method="thread",console='off'),config=config,mode=config["modo"],project=config["project"])

    modelo = unet(input_size=(224, 224, 1))
    optim = optimizers.Adam(config["lr"]) 

    # Compile el modelo con el optimizador Adam, la función de pérdida dice_loss y la métrica dice_coef

    #Imprima el resumen del modelo

    # Invoque el método fit del modelo con los generadores de entrenamiento y validación, y el número de épocas
    

    

if __name__  == "__main__":

    config = {
        "epocas":5,
        "batch_size":1,
        "lr":0.01,
        "modo":"online",
        "project":"taller_ml",
        "directorio_almacenado": "pesos"
    }
    
    if not os.path.exists(config["directorio_almacenado"]):
        os.mkdir(config["directorio_almacenado"])

    id_dataset = "rutas_dataset/id_dataset.json"
    rutas_dataset = "rutas_dataset/rutas_dataset.json"

    with open(rutas_dataset) as file:
        rutas_dataset = json.load(file)

    with open(id_dataset) as file:
        id_dataset = json.load(file)

    # Instance el DataLoader con el dataset de entrenamiento y validación para imagenes de 224x224 y de 1 canal

    run(config)