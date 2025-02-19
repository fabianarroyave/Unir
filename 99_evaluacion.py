import pandas as pd
import numpy as np
import re
from datetime import datetime
import time
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from rapidfuzz import fuzz
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

########################################
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
########################################

# 1.1. Cargar la base consolidada de productos
df_consolidated = pd.read_csv("Consolidado_sin_duplicados_agrupados.csv", encoding="utf-8")

# 1.2. Cargar la base de especificaciones
df_specs = pd.read_csv("Consolidado_scraping_especificaciones_agrupados.csv", encoding="utf-8")
# Crear un texto agregado de especificaciones
df_specs['specs_text'] = df_specs['propiedad_limpia'].astype(str) + ": " + df_specs['valor_limpia'].astype(str)
# Agrupar las especificaciones por URL Producto
df_specs_agg = df_specs.groupby("URL Producto")['specs_text'].apply(lambda x: " ".join(x)).reset_index()

# 1.3. Cargar la base de comentarios y análisis de sentimientos
df_comments = pd.read_csv("consolidado_comentarios_vader_bert_analisis_emociones_nrc_topicos.csv", encoding="utf-8")
# Mapear las etiquetas de sentimiento a valores numéricos: negativo=-1, neutral=0, positivo=1
sentiment_mapping = {'negativo': -1, 'neutral': 0, 'positivo': 1}
df_comments['sentiment_vader_num'] = df_comments['sentimiento_vader'].map(sentiment_mapping)
df_comments['sentiment_bert_num'] = df_comments['sentimiento_bert'].map(sentiment_mapping)
# Agrupar por URL Producto y calcular el promedio
df_comments_agg = df_comments.groupby("URL Producto").agg({
    'sentiment_vader_num': 'mean',
    'sentiment_bert_num': 'mean'
}).reset_index()
df_comments_agg['sentiment'] = df_comments_agg[['sentiment_vader_num', 'sentiment_bert_num']].mean(axis=1)

# 1.4. Unir toda la información en un único DataFrame (relacionados por "URL Producto")
df = df_consolidated.merge(df_specs_agg, on="URL Producto", how="left") \
                    .merge(df_comments_agg[['URL Producto','sentiment']], on="URL Producto", how="left")
df['specs_text'] = df['specs_text'].fillna("")
df['sentiment'] = df['sentiment'].fillna(0)  # Si no hay información, se asume neutral

# 1.5. Preprocesar precios y calcular ratios de descuento
for col in ["Precio sin descuento", "Precio descuento tienda", "Precio descuento aliados", "Precio otros"]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9]", "", regex=True), errors='coerce')
for col in ["Precio descuento tienda", "Precio descuento aliados", "Precio otros"]:
    df[col] = df[col].fillna(df["Precio sin descuento"])
df['ratio_tienda'] = df["Precio descuento tienda"] / df["Precio sin descuento"]
df['ratio_aliados'] = df["Precio descuento aliados"] / df["Precio sin descuento"]
df['ratio_otros'] = df["Precio otros"] / df["Precio sin descuento"]
price_vectors = df[['ratio_tienda', 'ratio_aliados', 'ratio_otros']].values

# 1.6. Crear una característica combinada de texto para recomendaciones
df['combined_features'] = (df['Nombre'].astype(str) + " " + 
                           df['Marca'].astype(str) + " " +
                           df['Descripcion'].astype(str) + " " +
                           df['specs_text'])


# Usamos las stopwords en español obtenidas de NLTK
spanish_stopwords = stopwords.words("spanish")

# -------------------------------
# Alternative 1 – Content-based usando TF-IDF
# -------------------------------
def recommendation_tfidf(url_producto, top_n=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    try:
        idx = df[df["URL Producto"] == url_producto].index[0]
    except IndexError:
        print("Producto no encontrado.")
        return None
    # Calcular la similitud solo para la fila de la consulta
    query_vector = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Excluir el producto consultado
    cosine_sim[idx] = -1
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    rec_indices = [i for i, score in sim_scores]
    return df.iloc[rec_indices]

# -------------------------------
# Alternative 2 – Sistema híbrido combinando TF-IDF, sentimiento y similitud de precios
# -------------------------------
def recommendation_hybrid(url_producto, top_n=5, weights=(0.4, 0.3, 0.3)):
    w_text, w_sent, w_price = weights
    tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    try:
        idx = df[df["URL Producto"] == url_producto].index[0]
    except IndexError:
        print("Producto no encontrado.")
        return None
    # Calcular similitud textual solo para la consulta
    query_vector = tfidf_matrix[idx]
    cosine_sim_text = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Similitud de sentimiento: 1/(1+abs(s1-s2)) para la consulta
    sentiment_scores = df['sentiment'].values
    query_sent = sentiment_scores[idx]
    sentiment_sim = 1/(1+np.abs(sentiment_scores - query_sent))
    
    # Similitud de precio: coseno entre el vector de precios del producto consultado y el resto
    query_price = price_vectors[idx]
    def cosine_sim(vec1, vec2):
        if norm(vec1) == 0 or norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    n = len(df)
    price_sim = np.array([cosine_sim(query_price, price_vectors[i]) for i in range(n)])
    
    final_sim = w_text * cosine_sim_text + w_sent * sentiment_sim + w_price * price_sim
    final_sim[idx] = -1  # Excluir el producto consultado
    sim_scores = list(enumerate(final_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    rec_indices = [i for i, score in sim_scores]
    return df.iloc[rec_indices]

# -------------------------------
# Alternative 3 – Basado en clustering: Reducción dimensional con SVD + KMeans
# -------------------------------
def recommendation_clustering(url_producto, top_n=5, n_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    svd = TruncatedSVD(n_components=50, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    scaler = MinMaxScaler()
    price_norm = scaler.fit_transform(price_vectors)
    features = np.hstack([tfidf_reduced, price_norm])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    df['cluster'] = clusters
    try:
        idx = df[df["URL Producto"] == url_producto].index[0]
    except IndexError:
        print("Producto no encontrado.")
        return None
    target_cluster = df.loc[idx, 'cluster']
    cluster_indices = df[df['cluster'] == target_cluster].index
    target_feature = features[idx]
    distances = {i: np.linalg.norm(features[i] - target_feature) for i in cluster_indices if i != idx}
    sorted_indices = sorted(distances, key=distances.get)[:top_n]
    return df.iloc[sorted_indices]

# Se asume que dos productos son relevantes si comparten el mismo "Nombre" (tras aplicar .strip().lower())
def is_relevant(query_idx, rec_idx):
    return df.iloc[query_idx]['Nombre'].strip().lower() == df.iloc[rec_idx]['Nombre'].strip().lower()

def evaluate_recommendation_system(recommend_func, top_n=5, sample_size=100):
    np.random.seed(42)
    indices = np.random.choice(range(len(df)), size=min(sample_size, len(df)), replace=False)
    precision_list, recall_list, mae_list = [], [], []
    for idx in indices:
        query_url = df.iloc[idx]["URL Producto"]
        # Consideramos relevantes los productos que tienen el mismo "Nombre"
        gt = set(df[df['Nombre'].str.strip().str.lower() == df.iloc[idx]['Nombre'].strip().lower()].index)
        gt.discard(idx)
        rec_df = recommend_func(query_url, top_n=top_n)
        if rec_df is None or rec_df.empty:
            continue
        rec_indices = rec_df.index.tolist()
        hits = sum([1 for i in rec_indices if i in gt])
        precision = hits / top_n
        recall = hits / len(gt) if len(gt) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        # MAE: diferencia absoluta entre "Puntaje" del producto consultado y el promedio de "Puntaje" de los recomendados
        query_rating = df.iloc[idx]["Puntaje"]
        rec_ratings = df.iloc[rec_indices]["Puntaje"].values
        mae = np.abs(query_rating - np.mean(rec_ratings)) if len(rec_ratings) > 0 else np.nan
        mae_list.append(mae)
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_mae = np.nanmean(mae_list) if mae_list else 0
    return {"Precision@K": avg_precision, "Recall@K": avg_recall, "MAE": avg_mae}

# Evaluar las tres alternativas
metrics_tfidf = evaluate_recommendation_system(recommendation_tfidf, top_n=5, sample_size=100)
metrics_hybrid = evaluate_recommendation_system(recommendation_hybrid, top_n=5, sample_size=100)
metrics_cluster = evaluate_recommendation_system(recommendation_clustering, top_n=5, sample_size=100)

print("Evaluation Metrics:")
print("Alternative 1 (TF-IDF):", metrics_tfidf)
print("Alternative 2 (Hybrid):", metrics_hybrid)
print("Alternative 3 (Clustering):", metrics_cluster)

# Visualización de las métricas
methods = ["TF-IDF", "Hybrid", "Clustering"]
precision_vals = [metrics_tfidf["Precision@K"], metrics_hybrid["Precision@K"], metrics_cluster["Precision@K"]]
recall_vals = [metrics_tfidf["Recall@K"], metrics_hybrid["Recall@K"], metrics_cluster["Recall@K"]]
mae_vals = [metrics_tfidf["MAE"], metrics_hybrid["MAE"], metrics_cluster["MAE"]]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.bar(methods, precision_vals, color='skyblue')
plt.title("Precision@K")
plt.subplot(1,3,2)
plt.bar(methods, recall_vals, color='salmon')
plt.title("Recall@K")
plt.subplot(1,3,3)
plt.bar(methods, mae_vals, color='lightgreen')
plt.title("Mean Absolute Error (MAE)")
plt.tight_layout()
plt.show()
