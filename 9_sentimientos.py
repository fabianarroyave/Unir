import re
import time
import pandas as pd
import numpy as np
import nltk
import torch
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from transformers import pipeline  # Para usar modelos preentrenados
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nrclex import NRCLex 

# =============================================================================
# DESCARGA DE RECURSOS Y CONFIGURACIÓN INICIAL
# =============================================================================

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Inicialización global de NLTK
lemmatizer = WordNetLemmatizer()
global_stop_words = set(stopwords.words('spanish'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# =============================================================================
# "LAZY LOADERS" DE LOS PIPELINES (se cargan solo una vez por proceso)
# =============================================================================

def get_sentiment_pipeline_bert():
    global sentiment_pipeline_bert
    try:
        return sentiment_pipeline_bert
    except NameError:
        sentiment_pipeline_bert = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment", 
            device=0 if device=="cuda" else -1
        )
        return sentiment_pipeline_bert

def get_emotion_pipeline():
    global emotion_pipeline
    try:
        return emotion_pipeline
    except NameError:
        try:
            emotion_pipeline = pipeline(
                "text-classification", 
                model="mrm8488/distilroberta-finetuned-emotion-spanish", 
                return_all_scores=False, 
                device=0 if device=="cuda" else -1
            )
        except Exception as e:
            print("Error cargando el modelo de emociones:", e)
            emotion_pipeline = None
        return emotion_pipeline

def get_vader_analyzer():
    global vader_analyzer
    try:
        return vader_analyzer
    except NameError:
        vader_analyzer = SentimentIntensityAnalyzer()
        return vader_analyzer

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO Y ANÁLISIS
# =============================================================================

def limpiar_texto(texto):
    """
    Limpia y estandariza el texto:
      - Convierte a minúsculas.
      - Normaliza caracteres (ej: 'Ã±' a 'n').
      - Elimina caracteres especiales.
      - Elimina espacios extras.
      - Lematiza cada palabra, ignorando stopwords.
    """
    if not isinstance(texto, str):
        return texto
    texto = texto.lower()
    texto = unidecode(texto)
    texto = re.sub(r'[^a-z0-9áéíóúñ ]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palabras = [lemmatizer.lemmatize(palabra) for palabra in texto.split() if palabra not in global_stop_words]
    return ' '.join(palabras)

def analizar_sentimiento_vader(texto):
    """
    Analiza el sentimiento usando VADER y devuelve "positivo", "negativo" o "neutral".
    """
    if not isinstance(texto, str) or not texto.strip():
        return "neutral"
    analyzer = get_vader_analyzer()
    score = analyzer.polarity_scores(texto)
    if score['compound'] >= 0.05:
        return "positivo"
    elif score['compound'] <= -0.05:
        return "negativo"
    else:
        return "neutral"

def analizar_sentimiento_bert(texto):
    """
    Analiza el sentimiento usando un modelo BERT multilingüe.
    Se trunca la entrada a 512 tokens para evitar errores.
    """
    if not isinstance(texto, str) or not texto.strip():
        return "neutral"
    tokens = word_tokenize(texto)
    if len(tokens) > 512:
        tokens = tokens[:512]
    texto_truncado = ' '.join(tokens)
    pipeline_bert = get_sentiment_pipeline_bert()
    try:
        resultado = pipeline_bert(texto_truncado)[0]
    except Exception as e:
        return "neutral"
    if resultado['label'] in ['4 stars', '5 stars']:
        return "positivo"
    elif resultado['label'] in ['1 star', '2 stars']:
        return "negativo"
    else:
        return "neutral"

def analizar_emociones(texto):
    """
    Analiza las emociones usando un modelo especializado en español.
    Se mapea la etiqueta del modelo a una categoría:
      "feliz", "triste", "furioso", "desilusionado", "sorprendido", "miedoso", "confundido", o "neutral".
    """
    if not isinstance(texto, str) or not texto.strip():
        return "neutral"
    pipeline_emociones = get_emotion_pipeline()
    if pipeline_emociones is None:
        return "neutral"
    try:
        resultado = pipeline_emociones(texto)[0]
    except Exception as e:
        return "neutral"
    # Ampliamos el diccionario de mapeo de emociones
    mapping = {
        "alegría": "feliz",
        "felicidad": "feliz",
        "contento": "feliz",
        "euforia": "feliz",
        "tristeza": "triste",
        "melancolía": "triste",
        "depresión": "triste",
        "ira": "furioso",
        "enfado": "furioso",
        "indignación": "furioso",
        "desilusión": "desilusionado",
        "desagrado": "desilusionado",
        "asco": "desilusionado",
        "sorpresa": "sorprendido",
        "miedo": "miedoso",
        "confusión": "confundido",
        "neutral": "neutral"
    }
    etiqueta = resultado['label'].lower()
    return mapping.get(etiqueta, etiqueta)

def analizar_emociones_nrc(texto):
    """Analiza emociones usando NRC Emotion Lexicon"""
    if not isinstance(texto, str) or not texto.strip():
        return "neutral"
    analysis = NRCLex(texto)
    emociones_detectadas = analysis.top_emotions
    if not emociones_detectadas:
        return "neutral"
    return max(emociones_detectadas, key=lambda x: x[1])[0]  # Retorna la emoción dominante

# =============================================================================
# HELPER PARA APLICAR FUNCIONES EN MULTIPROCESAMIENTO CON BARRA DE PROGRESO
# =============================================================================

def multiprocess_apply(series, func, desc):
    """
    Aplica la función 'func' a cada elemento de la 'series' utilizando ProcessPoolExecutor.
    Se muestra una barra de progreso con tqdm.
    """
    results = []
    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(func, series), total=len(series), desc=desc):
            results.append(res)
    return results

# =============================================================================
# EJEMPLO ADICIONAL: USO DE ThreadPoolExecutor PARA PROCESAR UN CONJUNTO DE FILAS
# =============================================================================

def procesar_comentario(row):
    """
    Procesa una fila (se espera que contenga 'URL Producto' y 'comentario_limpio')
    y retorna una tupla con los análisis: (URL Producto, sentimiento_vader, sentimiento_bert, emocion).
    """
    vader = analizar_sentimiento_vader(row['comentario_limpio'])
    bert = analizar_sentimiento_bert(row['comentario_limpio'])
    emo = analizar_emociones(row['comentario_limpio'])
    return (row['URL Producto'], vader, bert, emo)

# =============================================================================
# FUNCIÓN PARA REGISTRAR EL TIEMPO DE EJECUCIÓN
# =============================================================================

def registrar_tiempo_df(ejecucion_tiempo):
    nueva_entrada = pd.DataFrame({
        "Fecha de ejecución": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Tiempo de ejecución (segundos)": [ejecucion_tiempo]
    })
    try:
        df_tiempo = pd.read_csv("tiempo.txt", sep="\t")
        df_tiempo = pd.concat([df_tiempo, nueva_entrada], ignore_index=True)
    except FileNotFoundError:
        df_tiempo = nueva_entrada
    df_tiempo.to_csv("tiempo.txt", sep="\t", index=False)
    print("El tiempo de ejecución ha sido registrado en tiempo.txt")

# =============================================================================
# BLOQUE FINAL: AGRUPACIÓN DE COMENTARIOS Y ANÁLISIS DE TÓPICOS (LDA)
# =============================================================================

def agrupar_y_analizar_temas(df):
    # 1. Agrupar por 'URL Producto' para obtener un texto unificado por cada producto
    df_resultado_agrupado = df.groupby('URL Producto', as_index=False).agg({
        'Tienda': 'first',
        'URL Producto': 'first',
        'comentario_limpio': lambda x: ' '.join(x)
    })

    # 2. LDA sobre la columna 'comentario_limpio'
    stop_words_spanish = stopwords.words('spanish')
    vectorizer = CountVectorizer(stop_words=stop_words_spanish, max_df=0.9, min_df=0.1)
    dtm = vectorizer.fit_transform(df_resultado_agrupado['comentario_limpio'])

    lda_model = LatentDirichletAllocation(n_components=125, random_state=20)
    lda_model.fit(dtm)

    # Asignar a cada fila el tópico más probable
    df_resultado_agrupado['topico'] = lda_model.transform(dtm).argmax(axis=1)

    # Extraer las 10 palabras más relevantes por tópico
    palabras = vectorizer.get_feature_names_out()
    topicos_palabras = []
    for i, topic in enumerate(lda_model.components_):
        palabras_clave = [palabras[idx] for idx in topic.argsort()[-10:]]
        topicos_palabras.append(" ".join(palabras_clave))

    df_resultado_agrupado['palabras_topico'] = df_resultado_agrupado['topico'].apply(lambda x: topicos_palabras[x])

    # 3. Unir con el DataFrame original usando 'URL Producto' para añadir 'topico' y 'palabras_topico'
    df_merged = df.merge(
        df_resultado_agrupado[['URL Producto', 'topico', 'palabras_topico']],
        on='URL Producto',
        how='left'
    )

    return df_merged

# =============================================================================
# BLOQUE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("Inicio del análisis de comentarios...")
    start_time = time.time()
    
    # Cargar el CSV de comentarios. Se espera que contenga al menos las columnas:
    # "comentario", "URL Producto" y "Tienda"
    df = pd.read_csv("Consolidado_scraping_comentarios.csv", encoding="utf-8", delimiter=",")
    
    # Verificar que las columnas obligatorias existen
    if 'comentario' not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'comentario'")
    if 'URL Producto' not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'URL Producto'")
    
    # Aplicar la función de limpieza y guardarla en una nueva columna
    df['comentario_limpio'] = df['comentario'].apply(limpiar_texto).astype(str)
    
    # -------------------------------------------------------------------------
    # Análisis de Sentimiento y Emociones usando multiprocesamiento
    # -------------------------------------------------------------------------
    print("Analizando sentimiento con VADER...")
    df["sentimiento_vader"] = multiprocess_apply(df["comentario_limpio"], analizar_sentimiento_vader, "VADER")
        
    print("Analizando sentimiento con BERT...")
    df["sentimiento_bert"] = multiprocess_apply(df["comentario_limpio"], analizar_sentimiento_bert, "BERT")
        
    print("Analizando emociones...")
    df["emocion"] = multiprocess_apply(df["comentario_limpio"], analizar_emociones, "Emociones")
        
    print("Analizando emociones con NRC Lexicon...")
    df["emocion_nrc"] = multiprocess_apply(df["comentario_limpio"], analizar_emociones_nrc, "Emociones NRC")
        
    # Mostrar algunas filas de ejemplo con resultados
    print(df[['comentario', 'comentario_limpio', 'sentimiento_vader', 'sentimiento_bert', 'emocion']].head(10))
    
    # -------------------------------------------------------------------------
    # Ejemplo adicional: Procesamiento concurrente de filas usando ThreadPoolExecutor
    # (Procesar un subconjunto único basado en "URL Producto")
    df_urls = df[['URL Producto', 'comentario_limpio']].drop_duplicates()
    last_processed_index = 0  # Usar checkpoint si se requiere
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(procesar_comentario, row): idx 
                   for idx, row in df_urls.iterrows() if idx >= last_processed_index}
        for future in tqdm(futures, total=len(futures), desc="Procesando comentarios con ThreadPool"):
            results.append(future.result())
    
    # Convertir resultados a DataFrame y guardar
    df_results = pd.DataFrame(results, columns=["URL Producto", "sentimiento_vader", "sentimiento_bert", "emocion_nrc"])
        
    # -------------------------------------------------------------------------
    # Bloque Final: Agrupar Comentarios y Análisis de Tópicos (LDA)
    # -------------------------------------------------------------------------
    df_resultado = agrupar_y_analizar_temas(df)
    df.to_csv("consolidado_comentarios_vader_bert_analisis_emociones_nrc_topicos.csv", index=False, encoding="utf-8")
    # -------------------------------------------------------------------------
    # Finalización y Registro del Tiempo de Ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"El tiempo total de ejecución fue de {execution_time:.2f} segundos.")
    registrar_tiempo_df(execution_time)
