import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils import resample
from tqdm import tqdm
import multiprocessing

def procesar_fila(fila):
    if np.all(np.isnan(fila)):
        return np.zeros_like(fila)  # Alternativa: np.nanmedian(df[precio_cols].values, axis=0)
    max_valor = np.nanmax(fila)
    return np.where(np.isnan(fila), max_valor, fila)

def calcular_inercia(k, data):
    kmeans = KMeans(n_clusters=k, random_state=20, n_init=10)
    kmeans.fit(data)
    return k, kmeans.inertia_

def calcular_silueta(k, data):
    labels = KMeans(n_clusters=k, random_state=20, n_init=10).fit_predict(data)
    return k, silhouette_score(data, labels)

# Función para reemplazar valores vacíos en las columnas de precios con el máximo de la fila
def reemplazar_precios(row, columnas_precios):
    max_precio = row[columnas_precios].max(skipna=True)
    if not pd.isna(max_precio):
        row[columnas_precios] = row[columnas_precios].fillna(max_precio)
    return row

# Función para reemplazar valores NaN en la columna "Puntaje"
def reemplazar_puntaje(row, valores_dict):
    if pd.isna(row["Puntaje"]):  # Si "Puntaje" está vacío o es NaN
        producto = row["Producto"]
        if producto in valores_dict:
            return valores_dict[producto]
    return row["Puntaje"]


if __name__ == "__main__":
    df = pd.read_csv("consolidado_sin_duplicados_agrupado.csv", encoding="utf-8", sep=",", engine="python", quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    df = df.replace('\n', ' ', regex=True)
    
    print(f"Dimensiones: {df.shape}")
    # Crear diccionario con el promedio de Puntaje agrupado por Producto
    valores_dict = df.groupby("Producto")["Puntaje"].mean().dropna().to_dict()

    # Definir las columnas de precios que queremos revisar
    columnas_precios = [
        "Precio sin descuento",
        "Precio descuento tienda",
        "Precio descuento aliados",
        "Precio otros"
    ]

    # Reemplazar valores de "Puntaje" de forma secuencial con tqdm para ver progreso
    df["Puntaje"] = [reemplazar_puntaje(row, valores_dict) for _, row in tqdm(df.iterrows(), total=len(df), desc="Corrigiendo 'Puntaje'")]
    print("Valores de 'Puntaje' vacíos han sido reemplazados con su promedio.")

    # Reemplazar valores de precios de forma secuencial con tqdm para ver progreso
    df = df.apply(lambda row: reemplazar_precios(row, columnas_precios), axis=1)
    print("Valores de precios vacíos han sido reemplazados con el mayor precio de la fila.")
  
    X_sample = df[['Precio sin descuento', 'Precio descuento aliados', 'Precio descuento tienda', 'Precio otros', 'Puntaje']]
    X = resample(X_sample, n_samples=300000, random_state=20)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("4")
    pca = PCA()
    pca.fit(X_scaled)
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    # Graficar varianza acumulada
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianza Explicada')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Selección del Número de Componentes en PCA')
    plt.legend()
    plt.grid()
    plt.show()

    encoding_dim = np.argmax(varianza_acumulada >= 0.95) + 1
    print("5")
    pca_final = PCA(n_components=encoding_dim)
    reduced_features = pca_final.fit_transform(X_scaled)
    print("6")
    K_range = range(2, 21)
    num_threads = os.cpu_count()
    print(f"Número de hilos disponibles: {num_threads}")
    
    results = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(calcular_inercia, k, reduced_features) for k in K_range}
        for future in tqdm(as_completed(futures), total=len(K_range), desc="Calculando KMeans"):
            results.append(future.result())
    
    K_values, distortions = zip(*sorted(results))
    plt.figure(figsize=(8, 4))
    plt.plot(K_values, distortions, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Distorsión (Inercia)')
    plt.title('Método del Codo')
    plt.grid()
    plt.show()
    
    sil_results = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures_silueta = {executor.submit(calcular_silueta, k, reduced_features) for k in K_range}
        for future in tqdm(as_completed(futures_silueta), total=len(K_range), desc="Calculando Silueta"):
            sil_results.append(future.result())
    
    K_sil_values, sil_scores = zip(*sorted(sil_results))
    plt.figure(figsize=(8, 4))
    plt.plot(K_sil_values, sil_scores, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Coeficiente de Silueta')
    plt.title('Análisis de Silueta')
    plt.grid()
    plt.show()
    

    
    
