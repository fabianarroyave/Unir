import os
import shutil
import pandas as pd
import unicodedata
import nltk
import itertools
import re
import sweetviz as sv
import time
import time
from datetime import datetime
import numpy as np
from datetime import datetime
from io import StringIO
from nltk.corpus import stopwords
from babel.numbers import parse_decimal
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz, process
from unidecode import unidecode  
from gensim.models import Word2Vec
from tqdm import tqdm

# Inicializar el lematizador y las stopwords en español
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))


# Función para corregir caracteres mal codificados
def corregir_caracteres(texto):
    if not isinstance(texto, str):
        return texto
    # Normalizar caracteres Unicode
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return texto

# Función para limpiar y estandarizar texto
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return texto  # Si no es texto, devolver sin cambios
    # Convertir a minúsculas
    texto = texto.lower()
    # Corregir caracteres mal codificados
    texto = corregir_caracteres(texto)
    # Corregir caracteres mal codificados (como Ã± -> ñ)
    texto = unidecode(texto)
    # Eliminar caracteres especiales que no sean letras o números
    texto = re.sub(r'[^a-z0-9áéíóúñ ]', '', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r'\n\s+', ' ', texto).strip()
    # Lemmatización (simplificar palabras a su forma base)
    lemmatizer = WordNetLemmatizer()
    palabras = [
        lemmatizer.lemmatize(palabra) 
        for palabra in texto.split() 
        if palabra not in stopwords.words('spanish')
    ]
    # Unir las palabras procesadas
    return ' '.join(palabras)

# Función para procesar las columnas de precios
def procesar_columna_precios (valor):
    s = str(valor).strip()
    try:
        # Intentar interpretar el número usando el locale español (ej. 'es_CO')
        return float(parse_decimal(s, locale='es_CO'))
    except Exception:
        # Fallback: procesamiento manual
        # Conservar solo dígitos, puntos y comas
        s_clean = re.sub(r'[^0-9.,]', '', s)
        if ',' in s_clean:
            # Se asume que la coma es el separador decimal y los puntos son separadores de miles
            s_clean = s_clean.replace('.', '')
            s_clean = s_clean.replace(',', '.')
            try:
                return float(s_clean)
            except:
                return np.nan
        else:
            # No hay coma, se eliminan los puntos (que se asumen separadores de miles)
            s_clean = s_clean.replace('.', '')
            try:
                return float(s_clean)
            except:
                return np.nan


# Extraer el puntaje numérico (ejemplo: 4.8, 4.7, etc.)
def extraer_puntaje(puntaje):
    if pd.isna(puntaje):
        return None
    match = re.search(r'(\d+\.\d+|\d+)', str(puntaje))  # Busca el primer número con o sin decimales
    return float(match.group(1)) if match else None

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
    # Ruta de la carpeta donde están los archivos CSV
    ruta_carpeta = r"C:\Users\francesca.arroyave\Desktop\Fabian\Progrmas\Python\TFM\TFM_Arroyave_Fabian_Anibal"

### Lista para almacenar los DataFrames consolidados
    print("Lista para almacenar los DataFrames consolidados")
    # Inicializar variables
    dataframes = []
    filas_revision = pd.DataFrame()
    archivos_no_leidos = []
    total_registros = 0
    archivos_procesados = 0
    registros_duplicados_eliminados = 0
    archivos_sin_leer = 0
    # Obtener la lista de archivos en la carpeta
    archivos = [archivo for archivo in os.listdir(ruta_carpeta) if archivo.endswith('.csv')]
    # Consolidar los archivos
    for archivo in archivos:
        ruta_archivo = os.path.join(ruta_carpeta, archivo)
        try:
            # Intentar leer el archivo con separación por comas
            df = pd.read_csv(ruta_archivo, encoding="latin1", sep=",")
            archivos_procesados += 1
        except:
            try:
                # Si falla, intentar leer con separación por punto y coma
                df = pd.read_csv(ruta_archivo, encoding="utf-8", sep=";")  
                archivos_procesados += 1
            except Exception as e:
                # Registrar archivos que no pudieron ser leídos
                archivos_no_leidos.append((archivo, str(e)))
                archivos_sin_leer += 1
            continue  
        # Sumar la cantidad de registros del archivo actual
        cantidad_registros = len(df)
        total_registros += cantidad_registros
        # Filtrar filas con exactamente 13 campos
        filas_validas = df.shape[1] == 13
        if filas_validas:
            df_validas = df[df.apply(lambda row: row.count(), axis=1) == 13]
            df_invalidas = df[~df.index.isin(df_validas.index)]
        else:
            df_validas = pd.DataFrame()
            df_invalidas = df
        # Almacenar registros válidos e inválidos
        dataframes.append(df_validas)
        if not df_invalidas.empty:  # Verifica si df_invalidas no está vacío
            filas_revision = pd.concat([filas_revision, df_invalidas], ignore_index=True)
    # Imprimir archivos que no pudieron ser leídos
    if archivos_no_leidos:
        print("\nArchivos que no pudieron ser leídos:")
        for archivo, error in archivos_no_leidos:
            print(f"- {archivo}   : {error}")
    else:
        print("\nTodos los archivos fueron leídos exitosamente.")
    # Consolidar todos los DataFrames en uno solo
    df_consolidado = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    
    
    
### Validacion y limpieza de información
    print("Validacion y limpieza de información")
    df_consolidado = df_consolidado.replace('\n', ' ', regex=True)
    #Ajuste formato columna producto
    df_consolidado["Producto"] = df_consolidado["Producto"].replace("consolas","consola").replace("Celular","celular").replace("Consola","consola")
    #Ajuste de formato para fecha
    df_consolidado["Fecha"] = pd.to_datetime(df_consolidado["Fecha"], errors='coerce').dt.strftime('%Y-%m-%d')    
    # Copiar la columna original 'Puntaje' a 'Puntaje Completo' para preservar la información original
    df_consolidado['Puntaje Completo'] = df_consolidado['Puntaje'].astype(str)
    # Aplicar la función a la columna 'Puntaje'
    df_consolidado['Puntaje'] = df_consolidado['Puntaje Completo'].apply(extraer_puntaje)
    # Crear diccionario con el promedio de Puntaje agrupado por Producto
    valores_dict = df_consolidado.groupby("Producto")["Puntaje"].mean().dropna().to_dict()
    # Reemplazar valores de "Puntaje" de forma secuencial con tqdm para ver progreso
    df_consolidado["Puntaje"] = [reemplazar_puntaje(row, valores_dict) for _, row in tqdm(df_consolidado.iterrows(), total=len(df_consolidado), desc="Corrigiendo 'Puntaje'")]  
    
    # Aplicar la función a las columnas relevantes
    columnas_precios = ["Precio sin descuento", "Precio descuento tienda", "Precio descuento aliados", "Precio otros"]
    for columna in columnas_precios:
        if columna in df_consolidado.columns:  # Verificar si la columna existe
            df_consolidado[columna] = df_consolidado[columna].apply(procesar_columna_precios)
        else:
            print(f"Advertencia: La columna '{columna}' no se encuentra en el DataFrame.")
    # Aplicar la función a cada fila del DataFrame
    df_consolidado.loc[:, columnas_precios] = df_consolidado[columnas_precios].apply(reemplazar_precios, axis=1, args=(columnas_precios,))


    if not filas_revision.empty:
        filas_revision.to_csv("Z_filas_revision.csv", index=False, encoding="utf-8")
        print("\nArchivo 'filas_revision.csv' creado con los registros problemáticos.")
    else:
        print("\nNo hay registros problemáticos.")
    # Eliminar duplicados del DataFrame consolidado
    dim = df_consolidado.shape
    registros_duplicados_eliminados = df_consolidado.duplicated().sum()
    df_consolidado_sin_duplicados = df_consolidado.drop_duplicates()       
    df_consolidado_sin_duplicados =df_consolidado_sin_duplicados.drop("Imagen", axis=1)
    df_consolidado_sin_duplicados['Descripcion'] = df_consolidado_sin_duplicados['Nombre'].apply(limpiar_texto)
    
      
     
### Creación de dataframes adicionales
    print("Creación de dataframes adicionales")
    # Verificar si las columnas "Fecha" y "Tienda" existen en el DataFrame consolidado
    if "Fecha" in df_consolidado_sin_duplicados.columns and "Tienda" in df_consolidado_sin_duplicados.columns:
        # Agrupar por "Fecha" y "Tienda" y contar los registros por combinación
        df_tienda_fecha = (
            df_consolidado_sin_duplicados
            .groupby(["Fecha", "Tienda"])
            .size()
            .reset_index(name="Cantidad de elementos"))   
    else:
        print("El DataFrame no contiene las columnas necesarias 'Fecha' y/o 'Tienda'.")
    # Verificar si las columnas "Fecha" "Tienda" y "Nombre" existen en el DataFrame consolidado
    if "Fecha" in df_consolidado_sin_duplicados.columns and "Tienda" in df_consolidado_sin_duplicados.columns:
        # Agrupar por "Fecha" y "Tienda" y contar los registros por combinación
        df_tienda_fecha_2 = (
            df_consolidado_sin_duplicados
            .groupby(["Fecha", "Tienda", "Nombre"])
            .size()
            .reset_index(name="Cantidad de elementos"))   
    else:
        print("El DataFrame no contiene las columnas necesarias 'Fecha' y/o 'Tienda'.")
    # Asegúrate de que la columna 'Fecha' sea una cadena para poder manipularla
    df_consolidado_fecha = df_consolidado_sin_duplicados
    df_consolidado_fecha['Fecha'] = df_consolidado_fecha['Fecha'].astype(str)
    # Crear las columnas 'year', 'mes' y 'dia'
    df_consolidado_fecha['year'] = df_consolidado_fecha['Fecha'].str[:4]  # Primeros 4 valores
    df_consolidado_fecha['mes'] = df_consolidado_fecha['Fecha'].str[5:7]  # Valores 6 y 7
    df_consolidado_fecha['dia'] = df_consolidado_fecha['Fecha'].str[-2:]  # Últimos 2 valores
    # Convertir 'year', 'mes' y 'dia' a enteros (opcional, pero útil para comparaciones)
    df_consolidado_fecha['year'] = df_consolidado_fecha['year'].astype(int)
    df_consolidado_fecha['mes'] = df_consolidado_fecha['mes'].astype(int)
    df_consolidado_fecha['dia'] = df_consolidado_fecha['dia'].astype(int)
    # Filtrar el DataFrame según las condiciones
    cond1 = df_consolidado_fecha['year'] == 2025
    cond2 = (df_consolidado_fecha['mes'] == 12) & (df_consolidado_fecha['dia'] > 14)   
    # Condición adicional: al menos una columna de precios debe tener un valor mayor a 100,000
    cond3 = (
        (df_consolidado_fecha['Precio sin descuento'] > 100000) |
        (df_consolidado_fecha['Precio descuento tienda'] > 100000) |
        (df_consolidado_fecha['Precio descuento aliados'] > 100000) |
        (df_consolidado_fecha['Precio otros'] > 100000)
        )
    # Aplicar las condiciones al DataFrame
    df_filtrado = df_consolidado_fecha[(cond1 | cond2) & cond3]  # Combinar condiciones
    # Seleccionar columnas específicas
    consolidado_url = df_filtrado[["Tienda", "URL Producto", "Nombre"]]
    
    
        
### Resultados finales
    print("\nResumen del proceso:")
    print(f"Cantidad de archivos {len(archivos)}")
    print(f"- Archivos procesados: {archivos_procesados}")
    print(f"- Archivos no procesados: {archivos_sin_leer}")
    print(f"- Total de registros iniciales: {total_registros}")
    print(f"- Dimensiones del DataFrame consolidado:  {dim}")
    print(f"- Registros válidos consolidados: {len(df_consolidado)}")
    print(f"- Registros no válidos consolidados: {len(archivos_no_leidos)}")
    print(f"- Registros válidos sin duplicados: {len(df_consolidado_sin_duplicados)}")
    print(f"- Registros duplicados eliminados: {registros_duplicados_eliminados}")
    print(f"- Registros problemáticos: {len(filas_revision)}")   
    # Guardar resultados en archivos
    df_tienda_fecha.to_csv("Z_Consolidado_tienda_fecha.csv", index=False, encoding='utf-8')
    print(f"\nDimensiones del consolidado_fecha_tienda: {df_tienda_fecha.shape}")
    df_tienda_fecha_2.to_csv("Z_consolidado_tienda_fecha_nombre.csv", index=False, encoding='utf-8')
    print(f"\nDimensiones del consolidado_fecha_tienda_nombre: {df_tienda_fecha_2.shape}")
    df_consolidado.to_csv("Z_Consolidado_general.csv", index=False, encoding='utf-8')
    print(f"\nDimensiones del consolidado: {df_consolidado.shape}")
    df_consolidado_sin_duplicados.to_csv("Consolidado_sin_duplicados.csv", index=False, encoding='utf-8')
    print(f"\nDimensiones del consolidado sin duplicados: {df_consolidado_sin_duplicados.shape}")
    print(df_consolidado_sin_duplicados.head(3))
    consolidado_url.to_csv("Z_consolidado_url.csv", index=False, encoding='utf-8')
    print(f"\nDimensiones del consolidado con urls: {consolidado_url.shape}")


### EDA Exploración Generar informes con Sweetviz
    print("***************************************************************************************************")
    print("\nGenerando informes de Sweetviz...")
    # Informe para consolidado_total
    reporte_consolidado_total = sv.analyze(df_consolidado)
    reporte_consolidado_total.show_html("reporte_consolidado_total2.html")
    print("Informe 'reporte_consolidado_total.html' generado exitosamente.")
    # Informe para consolidado_sin_duplicados
    reporte_consolidado_sin_duplicados = sv.analyze(df_consolidado_sin_duplicados)
    reporte_consolidado_sin_duplicados.show_html("reporte_consolidado_sin_duplicados2.html")
    print("Informe 'reporte_consolidado_sin_duplicados.html' generado exitosamente.")

