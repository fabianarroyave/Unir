import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz, process
from unidecode import unidecode  
import nltk 
from gensim.models import Word2Vec
import itertools
import time
from datetime import datetime
import unicodedata

# Función para limpiar y estandarizar texto
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return texto  # Si no es texto, devolver sin cambios
    # Convertir a minúsculas
    texto = texto.lower()
    # Corregir caracteres mal codificados (como Ã± -> ñ)
    texto = unidecode(texto)
    # Eliminar caracteres especiales que no sean letras o números
    texto = re.sub(r'[^a-z0-9áéíóúñ ]', '', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Lemmatización (simplificar palabras a su forma base)
    lemmatizer = WordNetLemmatizer()
    palabras = [
        lemmatizer.lemmatize(palabra) 
        for palabra in texto.split() 
        if palabra not in stopwords.words('spanish')
    ]
    # Unir las palabras procesadas
    return ' '.join(palabras)


# Alternativa 1: Detección de palabras similares usando RapidFuzz
def agrupar_palabras_similares_rapidfuzz(lista_palabras, umbral=85):
    """
    Agrupa palabras similares utilizando RapidFuzz con un umbral de similitud.
    """
    grupos = []
    for palabra in lista_palabras:
        agregado = False
        for grupo in grupos:
            # Comparar la palabra con el primer elemento de cada grupo
            if fuzz.ratio(palabra, grupo[0]) >= umbral:
                grupo.append(palabra)
                agregado = True
                break
        if not agregado:
            grupos.append([palabra])
    return grupos

# Alternativa 2: Detección de palabras similares usando Word2Vec
def agrupar_palabras_similares_word2vec(lista_palabras, min_count=1, similitud_minima=0.7):
    """
    Agrupa palabras similares utilizando Word2Vec y una similitud mínima.
    """
    # Entrenar un modelo Word2Vec con las palabras
    model = Word2Vec(sentences=[[p] for p in lista_palabras], vector_size=150, window=5, min_count=min_count, workers=6)
    model.train([[p] for p in lista_palabras], total_examples=len(lista_palabras), epochs=15)
    
    grupos = []
    for palabra in lista_palabras:
        agregado = False
        for grupo in grupos:
            # Comprobar similitud con el primer elemento del grupo
            try:
                if model.wv.similarity(palabra, grupo[0]) >= similitud_minima:
                    grupo.append(palabra)
                    agregado = True
                    break
            except KeyError:
                continue
        if not agregado:
            grupos.append([palabra])
    return grupos

# Asociar las palabras limpias originales con sus grupos
def encontrar_grupo(palabra, grupos):
    for i, grupo in enumerate(grupos):
        if palabra in grupo:
            return i  # Retornar el índice del grupo
    return -1  # Si no se encuentra, retornar -1

def registrar_tiempo_df(ejecucion_tiempo):
    # Crear DataFrame con la nueva entrada
    nueva_entrada = pd.DataFrame({
        "Fecha de ejecución": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Tiempo de ejecución (segundos)": [ejecucion_tiempo]
    })
    # Verificar si el archivo ya existe
    try:
        # Cargar el archivo existente
        df_tiempo = pd.read_csv("tiempo.txt", sep="\t")
        # Agregar la nueva entrada al DataFrame existente
        df_tiempo = pd.concat([df_tiempo, nueva_entrada], ignore_index=True)
    except FileNotFoundError:
        # Si el archivo no existe, usar la nueva entrada como DataFrame inicial
        df_tiempo = nueva_entrada

    # Guardar el DataFrame actualizado en el archivo
    df_tiempo.to_csv("token.txt", sep="\t", index=False)
    print("El tiempo de ejecución ha sido registrado en tiempo.txt")
    
    
print('inicio')    
start_time = time.time()
# Cargar el CSV
df = pd.read_csv("Consolidado_scraping_especificaciones.csv", encoding="utf-8", delimiter=",")

# Aplicar la función de limpieza a las columnas "Propiedad" y "Valor"
df['propiedad_limpia'] = df['Propiedad'].apply(limpiar_texto)
df['valor_limpia'] = df['Valor'].apply(limpiar_texto)
# Guardar el DataFrame limpio a un nuevo archivo CSV
ruta_salida = "resultados_scraping_limpio.csv"
df.to_csv(ruta_salida, index=False, encoding='utf-8')
print("Limpieza y estandarización completada. Archivo guardado en:", ruta_salida)
# Lista de palabras limpias únicas
palabras_unicas = df['propiedad_limpia'].dropna().unique().tolist()
# Alternativa 1: Agrupar palabras similares con RapidFuzz
grupos_rapidfuzz = agrupar_palabras_similares_rapidfuzz(palabras_unicas)
# Alternativa 2: Agrupar palabras similares con Word2Vec
grupos_word2vec = agrupar_palabras_similares_word2vec(palabras_unicas)
# Asegurarse de que ambas listas tengan la misma longitud
max_length = max(len(grupos_rapidfuzz), len(grupos_word2vec))
# Rellenar las listas más cortas con cadenas vacías
grupos_rapidfuzz += [[]] * (max_length - len(grupos_rapidfuzz))
grupos_word2vec += [[]] * (max_length - len(grupos_word2vec))
# Crear el DataFrame con los grupos
df_grupos = pd.DataFrame({
    'Grupo_RapidFuzz': ['; '.join(grupo) for grupo in grupos_rapidfuzz],
    'Grupo_Word2Vec': ['; '.join(grupo) for grupo in grupos_word2vec]
})
df['Grupo_RapidFuzz'] = df['propiedad_limpia'].apply(lambda x: encontrar_grupo(x, grupos_rapidfuzz))
df['Grupo_Word2Vec'] = df['propiedad_limpia'].apply(lambda x: encontrar_grupo(x, grupos_word2vec))
# Guardar el DataFrame con los grupos a un archivo CSV
ruta_salida_grupos = "Consolidado_scraping_especificaciones_agrupados.csv"
df.to_csv(ruta_salida_grupos, index=False, encoding='utf-8')
print("Agrupación de palabras similares completada. Archivo guardado en:", ruta_salida_grupos)
end_time = time.time()
# Cálculo del tiempo total en segundos
execution_time = end_time - start_time
print(f"El tiempo de ejecución fue de {execution_time:.2f} segundos.")

