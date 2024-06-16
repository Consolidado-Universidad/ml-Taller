
from scipy.interpolate import RegularGridInterpolator
from matplotlib.font_manager import json_load
from scipy import io
import numpy as np
from scipy import io
import keras
import json
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt

class DataLoader(keras.utils.Sequence):
    def __init__(self, lista_ID, rutas_dataset, image_size, batch_size, cantidad_canales, shuffle=False):

        try:
            assert len(lista_ID) >= batch_size, "error!: batch size es mayor a la cantidad de datos disponibles"
        except AssertionError as e:
            msj = "dataloader line 19 {}".format(e)
            logging.error(msj)

        self.lista_IDs = lista_ID
        self.rutas_dataset = rutas_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.cantidad_canales = cantidad_canales
        self.on_epoch_end()

    # Metodo on_epoch_end:
    # 
    # - Propósito: Este método se llama al final de cada época de entrenamiento.
    # - Función: Se encarga de barajar los índices de los datos si se ha especificado el shuffle

    def on_epoch_end(self):
            #Genera indices para el batch
            self.indices = np.arange(len(self.lista_IDs))
            #if self.shuffle:
            #    np.random.shuffle(self.lista_IDs)

    # Metodo __ len __:
    # 
    # - Propósito: Este método define el número de batches por época.
    # 
    # - Función: Calcula y retorna el número de batches basándose en el tamaño total del dataset y el tamaño de batch definido.
    # 
    # #### - $batches = \left \lceil \frac{cantidad_de_datos}{batch-size}  \right \rceil$
    # 

    def __len__(self):
            # retorna la cantidad de batches por epoca o interacione
            try:
                return int(np.floor(len(self.lista_IDs) / self.batch_size))
            except Exception as e:
                logging.debug("__len__: {} {}".format(e, e.args))

    # Metodo __ getitem __:
    # 
    # - Propósito: Este método obtiene un batch de datos.
    # - Función: Selecciona los índices correspondientes al batch actual y llama a un método auxiliar para generar los datos del batch.

    def __getitem__(self,index): # index = numero de batch hasta __len__
            #indices seleccionados
            indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

            #Busqueda de lista de ids
            lista_IDs_temp = [self.lista_IDs[k] for k in indices]
            X, y = self.__data_generation(lista_IDs_temp)
            return X,y

    # Metodo __data_generation:
    # 
    # - Propósito: Este método se encarga de la generación de los datos del batch.
    # - Función: Carga y preprocesa las imágenes y sus correspondientes etiquetas/máscaras.

    def __data_generation(self,lista_ids_temp):
            X = np.empty((self.batch_size, *self.image_size, self.cantidad_canales))
            # Crear un array vacio de etiquetas Y


            #generacion de datos
            for i, ID in enumerate(lista_ids_temp):

                #[desarollar] carga de imagen y etiqueta
                
                #[desarollar] leer imagen en escala de grises y definir shape de salida con load_img 

                imagen = img_to_array(imagen) / 255.0  # Normalize to [0, 1]

                with open(ruta_etiqueta) as file:
                    etiqueta = json.load(file)

                imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

                mask = np.zeros(imagen_original.shape, dtype=np.uint8)

                #[desarollar] Dibujar poligonos de json en la mascara
                for indice in range(0,len(etiqueta["shapes"])):

                mask = cv2.resize(mask, (224,224))

                # [desarollar] expandir dimensiones de la mascara para que coincida con la imagen

                #[desarollar]  binarizar la mascara

                X[i,] = imagen
                Y[i,] = imagen


            return X,Y


if __name__ == "__main__":

    id_dataset = "../rutas_dataset/id_dataset.json"
    rutas_dataset = "../rutas_dataset/rutas_dataset.json"

    with open(rutas_dataset) as file:
        rutas_dataset = json.load(file)

    with open(id_dataset) as file:
        id_dataset = json.load(file)

    train_generator = DataLoader(id_dataset["train"], rutas_dataset, image_size=(224,224), batch_size=1, cantidad_canales=1,shuffle=False)

    for i in range(10):
        for X, y in train_generator:
            print(X.shape)
            print(y.shape)

            fig, ax = plt.subplots(1,2, figsize=(10,10))
            ax[0].imshow(X[0,:,:,0], cmap="gray")
            ax[1].imshow(y[0,:,:,0], cmap="gray")
            plt.show()
            plt.close()
            #break