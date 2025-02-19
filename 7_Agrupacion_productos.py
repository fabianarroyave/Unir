import os
import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from rapidfuzz import fuzz
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Descargar recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

# Función para calcular la similitud rápida
def calcular_similitud(texto1, texto2):
    return fuzz.ratio(texto1, texto2) / 100  # Normalizado entre 0 y 1

# Función de agrupación paralela optimizada
def procesar_par(i, df_chunk, umbral_similitud, umbral_precio):
    df_fila = df_chunk.iloc[i]
    grupos = []
    for j in range(i + 1, len(df_chunk)):
        df_comp = df_chunk.iloc[j]
        similitud = calcular_similitud(df_fila['Texto_Combinado'], df_comp['Texto_Combinado'])
        diferencia_precio = abs(df_fila['Precio_Normalizado'] - df_comp['Precio_Normalizado'])
        if similitud >= umbral_similitud and diferencia_precio <= umbral_precio:
            grupos.append(j)
    return (i, grupos)

# **Dividir DataFrame en partes para paralelización**
def dividir_dataframe(df, num_partes):
    chunk_size = len(df) // num_partes
    return [df.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_partes)]

def agrupar_productos_con_precios_mp(df, umbral_similitud=0.75, umbral_precio=0.15):
    # Asegurarse de que el índice es consecutivo
    df.reset_index(drop=True, inplace=True)
    
    df['Texto_Combinado'] = df['Producto'] + " " + df['Descripcion'] + df["Tipo"] + df["Marca"]
    scaler = MinMaxScaler()
    df['Precio_Normalizado'] = scaler.fit_transform(df[['Precio sin descuento']])
    df['Grupo'] = -1
    grupo_actual = 0
    num_workers = multiprocessing.cpu_count()
    df_chunks = dividir_dataframe(df, num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        resultados = executor.map(procesar_par, range(len(df)), df_chunks, 
                                  [umbral_similitud] * len(df), [umbral_precio] * len(df))
    for i, grupos in resultados:
        if df.loc[i, 'Grupo'] == -1:
            df.loc[i, 'Grupo'] = grupo_actual
            for j in grupos:
                df.loc[j, 'Grupo'] = grupo_actual
            grupo_actual += 1
    return df


def agrupar_palabras_similares_rapidfuzz(lista_palabras, umbral=80):
    grupos = []
    def comparar_palabra(palabra):
        for grupo in grupos:
            if fuzz.ratio(palabra, grupo[0]) >= umbral:
                grupo.append(palabra)
                return
        grupos.append([palabra])
    # Usar ThreadPoolExecutor en lugar de multiprocessing.Pool
    with ThreadPoolExecutor() as executor:
        executor.map(comparar_palabra, lista_palabras)
    return grupos

def clustering_pca_kmeans(df, n_components=2, k=9):
    try:
        #Definir las columnas categóricas que quieres codificar
        cat_cols = ["Producto", "Tipo"]
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        # Ajustar y transformar las columnas categóricas
        X_cat = encoder.fit_transform(df[cat_cols])
        scaler = StandardScaler()
        
        # Columnas numéricas 
        columnas = ["Precio sin descuento", "Precio descuento tienda", 
                    "Precio descuento aliados", "Puntaje", "Grupo", "Grupo_RapidFuzz"]
        
        if any(col not in df.columns for col in columnas):
            raise ValueError("Faltan columnas necesarias para el clustering.")
        
        if df[columnas].isna().sum().sum() > 0:
            print("Se encontraron valores NaN en las columnas de clustering. Se imputarán con la media.")
            df[columnas] = df[columnas].fillna(df[columnas].mean())
        
        # Extraer los valores numéricos y combinar  columnas
        X_num = df[columnas].values
        X_combined = np.concatenate([X_num, X_cat], axis=1)
        
        # Escalar los datos combinados
        datos_escalados = scaler.fit_transform(X_combined)
        if np.isnan(datos_escalados).any() or np.isinf(datos_escalados).any():
            print("Error: Hay valores NaN o Inf en los datos escalados. Se reemplazarán por la media.")
            datos_escalados = np.nan_to_num(datos_escalados, nan=np.nanmean(datos_escalados))
        
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(datos_escalados)
        
        # K-Means con los componentes reducidos
        kmeans = KMeans(n_clusters=k, random_state=20, n_init=15)
        df['Cluster'] = kmeans.fit_predict(reduced_features)
        
        return df
    except Exception as e:
        print(f"Error en clustering_pca_kmeans: {e}")
        return df

# Main
if __name__ == "__main__":
    df_consolidado_sin_duplicados = pd.read_csv("consolidado_sin_duplicados_agrupado.csv", encoding="utf-8")
    print(df_consolidado_sin_duplicados.shape)  
    columnas = ["Precio sin descuento", "Precio descuento tienda", 
                        "Precio descuento aliados", "Puntaje"]
    df_consolidado_sin_duplicados = df_consolidado_sin_duplicados.dropna(subset=columnas)
    print(df_consolidado_sin_duplicados.shape)  

    # Agrupar términos con multiprocesos
    df_consolidado_sin_duplicados_agrupados=df_consolidado_sin_duplicados
    palabras_unicas = df_consolidado_sin_duplicados_agrupados['Descripcion'].dropna().unique().tolist()
    print("6")
    grupos_rapidfuzz = agrupar_palabras_similares_rapidfuzz(palabras_unicas)
    print("7")
    df_consolidado_sin_duplicados_agrupados['Grupo_RapidFuzz'] = df_consolidado_sin_duplicados_agrupados['Descripcion'].apply(
        lambda x: next((i for i, grupo in enumerate(grupos_rapidfuzz) if x in grupo), -1))
    print("8")
    # Agrupar productos con paralelización
    df_consolidado_sin_duplicados_agrupados = agrupar_productos_con_precios_mp(df_consolidado_sin_duplicados_agrupados)
    print("9")
    # Aplicar clustering con PCA + KMeans
    df_consolidado_sin_duplicados_agrupados = clustering_pca_kmeans(df_consolidado_sin_duplicados_agrupados)   
    print("10")
    #Guardar archivo agrpados
    df_consolidado_sin_duplicados_agrupados.to_csv("Consolidado_sin_duplicados_agrupados.csv", index=False, encoding='utf-8')
    
    
