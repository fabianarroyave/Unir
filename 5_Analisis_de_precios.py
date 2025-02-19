import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Establecer estilo y paleta global de Seaborn
sns.set_style("whitegrid")
sns.set_palette("pastel")


### Funciones de procesamiento 
def ajustar_valores(valor):
    if valor > 10000000000000:
        return valor / 100000000
    elif valor > 1000000000000:
        return valor / 10000000
    elif valor > 10000000000:
        return valor / 10000
    elif valor > 1000000000:
        return valor / 1000
    elif valor > 100000000:
        return valor / 100
    elif valor > 30000000:
        return valor / 10
    else:
        return valor

# Eliminar outliers mediante el calculo de percentiles
def eliminar_outliers_percentil(df, columnas, lower=0.01, upper=0.99):
    for col in columnas:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Modificar funciones de visualización para guardar las gráficas
def guardar_grafico(fig, titulo):
    filename = f"000_{titulo.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(filename, dpi=300)

### Funciones de visualización y análisis
def normalizar_por_nombre(grupo):
        return (grupo - grupo.min()) / (grupo.max() - grupo.min())
    
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

### BLOQUE PRINCIPAL if name
if __name__ == "__main__":
    
    # Cargar los archivos CSV
    df = pd.read_csv("Consolidado_sin_duplicados.csv", encoding="utf-8", sep=",", engine="python", quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    df_verificacion = pd.read_csv("Z_verificacion_marcas.csv", encoding="utf-8", sep=";", engine="python", quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    df = df.replace('\n', ' ', regex=True)
    df.dropna(inplace=True)
    
    print(f'############ {df.shape}')
    # Reemplazar URLs en "Tienda" usando el diccionario definido
    replace_dict = {
        "https://www.alkosto.com/": "retail_1",
        "https://www.exito.com/": "retail_2",
        "https://www.falabella.com.co/falabella-co": "retail_3",
        "https://www.mercadolibre.com.co/": "retail_4",
        "https://www.tiendasmetro.co/": "retail_5"        
    }
    df["Tienda"] = df["Tienda"].replace(replace_dict)
    
### Definir diccionarios de colores fijo
    tienda_colores = {
        "retail_1": "#7EC8E3",  # azul
        "retail_2": "#77DD77",  # Verde
        "retail_3": "#FF6961",  # Rojo
        "retail_4": "#FFB347",  # Naranja
        "retail_5": "#DDA0DD"   # Purpura
    }  
   
    #Ajuste de formato de fecha y rango de fechas
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(by="Precio sin descuento", ascending=False).iloc[3:]
    df = df.sort_values(by='Fecha')
    df = df[df["Fecha"] >= pd.to_datetime("2024-10-06")]
    df["Marca"] = df["Marca"].str.replace('por ', '', regex=False)

    #Ajuste de valores de precios
    columnas_precios = ["Precio sin descuento", "Precio descuento tienda", "Precio descuento aliados", "Precio otros"]
    columna_puntaje = "Puntaje"
    columna_producto = "Producto"
    valores_dict = df.groupby(columna_producto)[columna_puntaje].mean().dropna().to_dict()

    # Reemplazar valores faltantes en 'Puntaje' de manera vectorizada
    #    Primero mapeamos el promedio de cada producto, luego rellenamos NaN
    if df[columna_puntaje].isna().any():
        df["Puntaje_promedio"] = df[columna_producto].map(valores_dict)
        df[columna_puntaje] = df[columna_puntaje].fillna(df["Puntaje_promedio"])
        df.drop(columns=["Puntaje_promedio"], inplace=True)
        print("Valores de 'Puntaje' vacíos han sido reemplazados con el promedio de su producto.")
    else:
        print("No existen valores NaN en 'Puntaje'. Se omite reemplazo.")

    # Reemplazar valores faltantes en las columnas de precios con el máximo de la fila
    n_miss_prices = df[columnas_precios].isna().sum().sum()
    if n_miss_prices > 0:
        # Calcula el máximo por fila (vectorizado)
        max_por_fila = df[columnas_precios].max(axis=1)
        # Rellenar cada columna de precios
        for col in columnas_precios:
            df[col] = df[col].fillna(max_por_fila)
        print("Valores de precios vacíos han sido reemplazados con el mayor precio de la fila.")
    else:
        print("No existen valores NaN en las columnas de precios. Se omite reemplazo.")
    df[columnas_precios] = df[columnas_precios].applymap(ajustar_valores)
    # Método de percentiles 
    df = eliminar_outliers_percentil(df, columnas_precios)
    print(f'############ {df.shape}')
    
    
    #Delimitación de tipo de productos equipos generales y accesorios
    df = pd.merge(df, df_verificacion[['Marca', 'Tipo']], left_on='Marca', right_on='Marca', how='left')
    df['Marca'] = df['Marca'].str.lower().str.strip().str.replace(r'^por\s+', '', regex=True).str.replace(r'\s+', ' ', regex=True)
    df["Tipo"].fillna("Errores", inplace=True)
    
    #Selección de precios minimos en cada producto
    columnas_precios2 = ["Precio sin descuento minimo", "Precio descuento tienda minimo", "Precio descuento aliados minimo", "Precio otros minimo"]
    for col, col2 in zip(columnas_precios, columnas_precios2):
        df[col2] = df.groupby("Nombre")[col].transform("min")
        
    # Normalización de columnas_precios y columnas_precios2 según el mismo Nombre
    #for col in columnas_precios + columnas_precios2:
    #    df[f'{col}_normalizado'] = df.groupby('Nombre')[col].transform(normalizar_por_nombre)  

    df.to_csv("Consolidado_sin_duplicados_agrupado.csv", encoding= "utf-8")
    print(f'############ {df.shape}')
            
    #Selección solo de producto equipos clasificados
    df_equipos = df[df["Tipo"] == "Equipos"]
    productos = df_equipos["Producto"].unique()
    print(f'############ {df_equipos.shape}')

################
### Gráficos ###
################

    ### 1. Gráfico de líneas: Variación de precios mínimos por producto
    for prod in productos:
        df_prod = df_equipos[df_equipos["Producto"] == prod].sort_values("Fecha")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_prod, x="Fecha", y="Precio sin descuento minimo", hue="Tienda",
                    marker="o", palette=tienda_colores)
        plt.title(f"Variación de precios mínimos por producto {prod} por Tiendas", pad=10)
        plt.xlabel("Fecha", labelpad=7)
        plt.ylabel("Precio sin descuento", labelpad=10)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig = plt.gcf()  # Obtener la figura actual
        titulo = "1. Variación de precios mínimos para " + prod
        guardar_grafico(fig, titulo)
        

    
    ### 2. Gráfico: Boxplots de Coeficiente de Variación (CV) mínimos por producto
    df_coeficiente = df_equipos.copy()
    df_coeficiente["CV_minimo"] = df_coeficiente[columnas_precios2].std(axis=1) / df_coeficiente[columnas_precios2].mean(axis=1)
    subtabla = df_coeficiente[['Tienda', 'Producto', 'Nombre', 'CV_minimo']].dropna().sort_values(by="Tienda")
    for prod in productos:
        df_temp = subtabla[subtabla["Producto"] == prod]
        plt.figure(figsize=(11, 7))
        sns.boxplot(x='Tienda', y='CV_minimo', data=df_temp, palette=tienda_colores)
        plt.title(f'Variación del Coeficiente de Variación Mínimo por Tienda - Producto: {prod}')
        plt.xlabel('Tienda')
        plt.ylabel('Coeficiente de Variación Mínimo')
        plt.xticks(rotation=0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig = plt.gcf()
        titulo = "2. Coeficiente de Variación mínimo en " + prod
        guardar_grafico(fig, titulo)
        
    
    
    ### 3. Gráfico: Densidad de diferencia para afiliados
    # Calcular diferencia porcentual
    df_diferencia = df_equipos.copy()
    df_diferencia['diferencia'] = ((df_diferencia['Precio sin descuento'] - df_diferencia['Precio descuento tienda minimo']) / 
                                    df_diferencia['Precio descuento tienda minimo'])
    diferencia_porcentaje = df_diferencia[['Tienda', 'Producto', 'Nombre', 'diferencia']].dropna().sort_values(by="Tienda")
    for prod in productos:
        df_temp = diferencia_porcentaje[diferencia_porcentaje['Producto'] == prod]
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")
        for tienda, data in df_temp.groupby('Tienda'):
            sns.kdeplot(data=data, x='diferencia', label=tienda, hue='Tienda',
                        fill=True, alpha=0.3, linewidth=2, palette=tienda_colores)
        plt.title(f'Densidad de Diferencia para {prod}')
        plt.xscale('log')  # Escala logarítmica en el eje X
        plt.xlabel('Porcentaje de Diferencia')
        plt.ylabel('Densidad')
        plt.legend(title='Tienda')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig = plt.gcf()
        titulo = "3. Densidad de la diferencia de costo y descuento en " + prod
        guardar_grafico(fig, titulo)
        ### 3.1 Gráfico: Densidad de diferencia para afiliados
        # Calcular diferencia porcentual
        df_diferencia = df_equipos.copy()
        df_diferencia['diferencia'] = ((df_diferencia['Precio sin descuento'] - df_diferencia['Precio descuento tienda minimo']) / 
                                        df_diferencia['Precio descuento tienda minimo'])
        diferencia_porcentaje = df_diferencia[['Tienda', 'Producto', 'Nombre', 'diferencia']].dropna().sort_values(by="Tienda")
        for prod in productos:
            df_temp = diferencia_porcentaje[diferencia_porcentaje['Producto'] == prod]
            plt.figure(figsize=(10, 5))
            sns.set_style("whitegrid")
            for tienda, data in df_temp.groupby('Tienda'):
                sns.kdeplot(data=data, x='diferencia', label=tienda, hue='Tienda',
                            fill=True, alpha=0.3, linewidth=2, palette=tienda_colores)
            plt.title(f'Densidad de Diferencia para {prod}')
            plt.xscale('log')  # Escala logarítmica en el eje X
            plt.xlabel('Porcentaje de Diferencia')
            plt.ylabel('Densidad')
            plt.legend(title='Tienda')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            fig = plt.gcf()
            titulo = "3. Densidad de la diferencia de costo y descuento en " + prod + " -X"
            guardar_grafico(fig, titulo)
      

                            
    ### 4. Gráfico: Línea de Variación de diferencia neta por producto
    df_diferencia_neta = df_equipos.copy()
    df_diferencia_neta['diferencia_neta'] = (df_diferencia_neta['Precio sin descuento'] - df_diferencia_neta['Precio descuento tienda minimo'])
    diferencia_neta = df_diferencia_neta[['Fecha', 'Tienda', 'Producto', 'Nombre', 'diferencia_neta']].dropna().sort_values(by="Tienda")
    for prod in productos:
        df_temp = diferencia_neta[diferencia_neta['Producto'] == prod]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_temp, x="Fecha", y="diferencia_neta", hue="Tienda", style="Tienda", markers=True, palette=tienda_colores)
        plt.title(f"Variación de diferencia neta por producto: {prod}", pad=10)
        plt.xlabel("Fecha")
        plt.ylabel("Diferencia Neta (COP)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.legend(title="Tienda")
        fig = plt.gcf()
        titulo = "4. Variación de diferencia neta en " + prod
        guardar_grafico(fig, titulo)
        
    
    
    ### 5. Gráficos de dispersión comparativo por producto
    df_precios = df_equipos.drop_duplicates(subset=['Producto', 'Tienda', 'Nombre', 'Precio sin descuento', 'Precio descuento tienda', 'Precio otros'])
    for prod in productos:
        df_temp = df_precios[df_precios['Producto'] == prod]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # Scatter plot: Precio sin descuento vs Precio descuento tienda
        sns.scatterplot(data=df_temp, x='Precio sin descuento', y='Precio descuento tienda',
                        hue='Tienda', palette=tienda_colores, alpha=0.7, legend='brief', ax=axes[0])
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_title(f'Precio sin descuento vs Precio descuento tienda para {prod}')
        axes[0].set_xlabel('Precio sin descuento (COP) [Escala log]')
        axes[0].set_ylabel('Precio descuento tienda (COP) [Escala log]')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        # Scatter plot: Precio sin descuento vs Precio otros
        sns.scatterplot(data=df_temp, x='Precio sin descuento', y='Precio otros',
                        hue='Tienda', palette=tienda_colores, alpha=0.5, legend=False, ax=axes[1])
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title(f'Precio sin descuento vs Precio otros para {prod}')
        axes[1].set_xlabel('Precio sin descuento (COP) [Escala log]')
        axes[1].set_ylabel('Precio otros (COP) [Escala log]')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig = plt.gcf()
        titulo = "5. Comparación por tipo de descuento en " + prod
        guardar_grafico(fig, titulo)
        
            
    
    ### 6. Gráficos: Línea de descuento por producto
    data_por = df_equipos.drop_duplicates(subset=['Fecha', 'Producto', 'Tienda', 'Nombre', 'Precio sin descuento', 'Precio descuento tienda', 'Precio otros'])
    data_por['descuento_tienda'] = (data_por['Precio sin descuento'] - data_por['Precio descuento tienda'])
    data_por['descuento_otros'] = (data_por['Precio sin descuento'] - data_por['Precio otros'])
    for prod in productos:
        df_temp = data_por[data_por['Producto'] == prod]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.lineplot(data=df_temp, x='Fecha', y='descuento_tienda', hue='Tienda', palette=tienda_colores, marker='o', ax=axes[0])
        axes[0].set_title(f'Valor de Descuento Tienda para {prod}')
        axes[0].set_xlabel('Fecha')
        axes[0].set_ylabel('Descuento (%)')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        sns.lineplot(data=df_temp, x='Fecha', y='descuento_otros', hue='Tienda', palette=tienda_colores, marker='x', ax=axes[1])
        axes[1].set_title(f'Porcentaje de Descuento Otros para {prod}')
        axes[1].set_xlabel('Fecha')
        axes[1].set_ylabel('Descuento (%)')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig = plt.gcf()
        titulo = "6. Diferencia de precio sin descuentos y por tipo de descuento en " + prod
        guardar_grafico(fig, titulo)
        
            
    
    ### 7. Gráfico: Cajas y Bigotes por día de la semana y por tienda
    # Crear columna con el precio mínimo entre "Precio descuento tienda" y "Precio otros"
    df_equipos = df_equipos.copy()
    df_equipos.loc[:, 'descuento_minimo'] = df_equipos[['Precio descuento tienda', 'Precio otros']].min(axis=1)

    # Crear la columna 'Dia_Semana' sin .sort()
    df_equipos.loc[:, 'Dia_Semana'] = df_equipos['Fecha'].dt.day_name()

    # (Opcional) Ordenar el DataFrame por día de la semana en orden natural (lunes a domingo)
    ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df_equipos['Dia_Semana'] = pd.Categorical(df_equipos['Dia_Semana'],categories=ordered_days,ordered=True)
    df_equipos = df_equipos.sort_values('Dia_Semana')
    for prod in productos:
        df_prod = df_equipos[df_equipos['Producto'] == prod]
        tiendas = df_prod['Tienda'].unique()
        fig, axes = plt.subplots(len(tiendas), 1, figsize=(12, 6 * len(tiendas)), sharex=True)
        fig.suptitle(f'Diagrama de Cajas y Bigotes por Día de la Semana y Tienda para {prod}', fontsize=16)
        if len(tiendas) == 1:
            axes = [axes]
        for ax, tienda in zip(axes, tiendas):
            sns.boxplot(data=df_prod[df_prod['Tienda'] == tienda], x='Dia_Semana', y='descuento_minimo', ax=ax )
            ax.set_title(f'Tienda: {tienda}')
            ax.set_ylabel('Precio Mínimo (COP)')
            ax.set_xlabel('Día de la Semana')
            ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        titulo = "7. Variación de precios con menores descuentos por día en " + prod
        guardar_grafico(fig, titulo)
        
    ### 8. Correlación de valores por Producto
    for prod in df_equipos["Producto"].unique():
        # Filtrar el DataFrame para el producto actual
        df_prod = df_equipos[df_equipos["Producto"] == prod]
        # Calcular la correlación para cada grupo definido por "Nombre"
        correlaciones = df_prod.groupby("Nombre").apply(lambda x: x["Precio sin descuento"].corr(x["Precio otros"]))
        # Convertir la Serie de correlaciones a DataFrame para facilitar el uso en seaborn
        df_corr = pd.DataFrame({"correlacion": correlaciones})
        # Crear la figura
        plt.figure(figsize=(10, 6))
        # Generar el boxplot horizontal
        sns.boxplot(
            x="correlacion",
            data=df_corr,
            orient="h",  # Orientación horizontal
            color="lightblue",
            linewidth=2,  # Bordes más visibles
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5})
        plt.title(f"Distribución de correlación entre\n'Precio sin descuento' y 'Precio otros'\npor Producto: {prod}", fontsize=13, pad=15)
        plt.xlabel("Coeficiente de correlación", fontsize=12, labelpad=10)
        plt.ylabel("")  # Se oculta la etiqueta del eje Y
        plt.yticks([])  # Se eliminan las marcas del eje Y
        plt.grid(axis="x", linestyle="--", alpha=0.7)  # Líneas de referencia en el eje X
        plt.tight_layout()
        # Guardar la figura con el título correspondiente
        fig = plt.gcf()  # Obtener la figura actual
        guardar_grafico(fig, f"8. Correlación por Producto {prod}")
        
    
    
    # 9. Descomposición STL (Seasonal-Trend using Loess)
    for tienda in df_equipos['Tienda'].unique():
        # Filtrar datos para la tienda actual
        df_ts = df_equipos[df_equipos['Tienda'] == tienda]
        df_ts = df_ts.set_index('Fecha').sort_index()
        # Calcular el promedio diario de "Precio sin descuento" para la tienda
        ts = df_ts['Precio sin descuento'].resample('D').mean().dropna()
        # Aplicar STL para descomponer la serie
        stl = STL(ts, period=7, robust=True)
        result = stl.fit()
        # Graficar los componentes: tendencia, estacionalidad y residuales
        result.plot()
        plt.suptitle(f'STL Decomposition - {tienda}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Guardar la gráfica
        titulo = "9. Descomposición STL (Seasonal-Trend using Loess) en " + tienda
        fig = plt.gcf()
        guardar_grafico(fig, titulo)
    
### Gráfico individualizado
    # 2. Filtrar donde Tipo == "Equipos"
    df_equipo = df[df["Tipo"] == "Equipos"].copy()
    df_nombre_stats = df_equipo.groupby("Nombre").agg({
        "Precio sin descuento": ["nunique", "mean"],
        "Precio descuento tienda": "mean",
        "Precio descuento aliados": "mean",
        "Precio otros": "mean"
    })
    df_nombre_stats.columns = ["distinct_count_psd", "mean_psd", "mean_pdt", "mean_pda", "mean_po"]
    df_nombre_stats.reset_index(inplace=True)

    # Definir la condición:
    def medias_no_iguales(row):
        valores = {row["mean_psd"], row["mean_pdt"], row["mean_pda"], row["mean_po"]}
        return len(valores) > 1

    df_nombre_stats = df_nombre_stats[
        (df_nombre_stats["distinct_count_psd"] > 3) &
        (df_nombre_stats.apply(medias_no_iguales, axis=1))
    ]

    # Filtrar df_equipos para quedarnos sólo con esos Nombres
    df_equipos2 = df_equipo[df_equipos["Nombre"].isin(df_nombre_stats["Nombre"])].copy()
    productos_unicos = df_equipos2["Producto"].unique()
    nombres_final = []
    for prod in productos_unicos:
        # Filtrar filas de ese Producto
        subdf = df_equipos2[df_equipos2["Producto"] == prod]
        # Filtrar Precios sin descuento entre 1,000,000 y 3,000,000
        subdf_filtrado = subdf[
            (subdf["Precio sin descuento"] >= 1_000_000) &
            (subdf["Precio sin descuento"] <= 3_000_000)
        ]

        # Tomar hasta 5 valores únicos de Nombre
        nombres_prod = subdf_filtrado["Nombre"].unique()[:3]

        # Agregarlos a la lista final
        nombres_final.extend(nombres_prod)

    # Quitar duplicados en la lista final
    nombres_final = list(set(nombres_final))
    # Filtrar el DataFrame final con esos Nombres
    df_filtrado = df_equipos2[df_equipos2["Nombre"].isin(nombres_final)].copy()

    # Graficar un diagrama de líneas por cada Nombre, mostrando las 4 columnas de precios
    # ----------------------------------------------------------------------------
    sns.set_palette("pastel")
    for nombre in df_filtrado["Nombre"].unique():
        df_name = df_filtrado[df_filtrado["Nombre"] == nombre].sort_values("Fecha")
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        # Precio sin descuento: marcador "o", línea discontinua
        sns.lineplot(data=df_name, x="Fecha", y="Precio sin descuento",label="Precio sin descuento", marker="X", linestyle="-")  # solid
        # Precio descuento tienda: marcador "X", línea continua
        sns.lineplot(data=df_name, x="Fecha", y="Precio descuento tienda",label="Precio descuento tienda", marker="o", linestyle="--")  # dashed
        # Precio descuento aliados: marcador "*", línea punteada
        sns.lineplot(data=df_name, x="Fecha", y="Precio descuento aliados",label="Precio descuento aliados", marker="*", linestyle=":")  # dotted
        # Precio otros: marcador "^", línea con puntos
        sns.lineplot(data=df_name, x="Fecha", y="Precio otros",label="Precio otros", marker="+", linestyle="-.")  # dashdot
        plt.title(f"Variación de Precios - Nombre: {nombre}")
        plt.xlabel("Fecha")
        plt.ylabel("Precio (COP)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()