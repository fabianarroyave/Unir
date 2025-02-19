import concurrent.futures
import os
import csv
import pandas as pd
import time
import random
import unicodedata
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from datetime import datetime
from bs4 import BeautifulSoup


# Función para iniciar el driver
def iniciar_driver():
    option = Options()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.2; rv:85.0) Gecko/20100101 Firefox/85.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; SM-A515F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36",
        # Navegadores de escritorio
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.1; rv:84.0) Gecko/20100101 Firefox/84.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/17.17134 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        # Navegadores móviles
        "Mozilla/5.0 (Linux; Android 10; SM-A515F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Opera/9.80 (Android; Opera Mini/36.2.2254/191.161; U; en) Presto/2.12.423 Version/12.16",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) FxiOS/9.0 Mobile/15E148 Safari/602.1",
        # Otros dispositivos y navegadores
        "Mozilla/5.0 (PlayStation 4 3.11) AppleWebKit/537.73 (KHTML, like Gecko)",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.215 Safari/535.1",
        "Mozilla/5.0 (Nintendo Switch; WifiWebAuth) AppleWebKit/601.6 (KHTML, like Gecko) NF/4.1.0.1.10 NintendoBrowser/5.1.0.13343",
    ]#***********
    user_agent = random.choice(user_agents)#***********
    option.add_argument(f"user-agent={user_agent}")
    option.add_argument("--headless") #sin una ventana gráfica visible
    option.add_argument("--disable-web-security") #Desactiva la política de seguridad web del navegador
    option.add_argument("--disable-extensions") #Desactiva todas las extensiones del navegador
    option.add_argument("--disable-notifications") #Desactiva las notificaciones emergentes del navegador
    option.add_argument("--disable-popup-blocking") #Popoup quita ventanas
    option.add_argument("--disable-infobars")  # Desactiva infobars como la barra de control de Chrome
    option.add_argument("--ignore-certificate-errors") #Ignora los errores de certificado (por ejemplo, certificados SSL no válidos o caducados)
    option.add_argument("--ignore-certificate-errors-spki-list") #ignora errores relacionados con la lista SPKI
    option.add_argument("--no-sandbox") #el scraping se ejecuta en contenedores como Docker o en servidores que no permiten el uso de sandbox 
    option.add_argument("--log-level=3") #Reduce el nivel de registro del navegador, solo mostrando errores críticos
    option.add_argument("--allow-running-insecure-content") #Permite la ejecución de contenido no seguro (HTTP) en sitios que usan HTTPS
    option.add_argument("--no-default-browser-check") #Evita que el navegador verifique si es el predeterminado
    option.add_argument("--no-first-run") #Para evitar que la página de bienvenida interrumpa el proceso de scraping
    option.add_argument("--no-proxy-server") #Si deseas usar proxies para evitar bloqueos o rotar IPs, no deberías utilizar esta opción.
    option.add_argument("--disable-blink-features=AutomationControlled") #Desactiva las características de "AutomationControlled", que son utilizadas por los sitios web para detectar que el navegador está automatizado por Selenium
    except_options = ["enable-automation", "ignore-certificate-errors"] 
    option.page_load_strategy = 'eager' #Reduce tiempo de carga inicial
    option.add_experimental_option("excludeSwitches", except_options)
    prefs = {"profile.default_content_setting_values.notifications": 2,"profile.default_content_setting_values.geolocation": 2, "intl.accept_languages": ["es-ES", "es"]}
    option.add_experimental_option("prefs", prefs)
    def ocultar_automatizacion(driver):#***********
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_script('navigator.permissions.query = (parameters) => Promise.resolve({ state: "denied" });')

    serv = Service()
    driver = webdriver.Chrome(service=serv, options=option)
    ocultar_automatizacion(driver)
    
    return driver


# Función para iniciar el driver
def iniciar__driver():
    option = Options()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
    ]#***********
    user_agent = random.choice(user_agents)#***********
    option.add_argument(f"user-agent={user_agent}")
    option.add_argument("--headless") #sin una ventana gráfica visible
    option.add_argument("--disable-web-security") #Desactiva la política de seguridad web del navegador
    option.add_argument("--disable-extensions") #Desactiva todas las extensiones del navegador
    option.add_argument("--disable-notifications") #Desactiva las notificaciones emergentes del navegador
    option.add_argument("--disable-popup-blocking") #Popoup quita ventanas
    option.add_argument("--disable-infobars")  # Desactiva infobars como la barra de control de Chrome
    option.add_argument("--ignore-certificate-errors") #Ignora los errores de certificado (por ejemplo, certificados SSL no válidos o caducados)
    option.add_argument("--ignore-certificate-errors-spki-list") #ignora errores relacionados con la lista SPKI
    option.add_argument("--no-sandbox") #el scraping se ejecuta en contenedores como Docker o en servidores que no permiten el uso de sandbox 
    option.add_argument("-- |log-level=3") #Reduce el nivel de registro del navegador, solo mostrando errores críticos
    option.add_argument("--allow-running-insecure-content") #Permite la ejecución de contenido no seguro (HTTP) en sitios que usan HTTPS
    option.add_argument("--no-default-browser-check") #Evita que el navegador verifique si es el predeterminado
    option.add_argument("--no-first-run") #Para evitar que la página de bienvenida interrumpa el proceso de scraping
    option.add_argument("--no-proxy-server") #Si deseas usar proxies para evitar bloqueos o rotar IPs, no deberías utilizar esta opción.
    option.add_argument("--disable-blink-features=AutomationControlled") #Desactiva las características de "AutomationControlled", que son utilizadas por los sitios web para detectar que el navegador está automatizado por Selenium
    except_options = ["enable-automation", "ignore-certificate-errors"] 
    option.add_experimental_option("excludeSwitches", except_options)
    prefs = {"profile.default_content_setting_values.notifications": 2,"profile.default_content_setting_values.geolocation": 2, "intl.accept_languages": ["es-ES", "es"]}
    option.add_experimental_option("prefs", prefs)
    def ocultar_automatizacion(driver):#***********
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_script('navigator.permissions.query = (parameters) => Promise.resolve({ state: "denied" });')

    serv = Service()
    driver = webdriver.Chrome(service=serv, options=option)
    ocultar_automatizacion(driver)
    
    return driver

# Función para cerrar pop-ups
def cerrar_popup(driver):
    try:
        # Buscar el pop-up que bloquea el clic (se puede ajustar el selector)
        popup = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div#wps_popup'))
        )
        cerrar_boton = driver.find_element(By.CSS_SELECTOR, 'div#wps_popup button')
        driver.execute_script("arguments[0].click();", cerrar_boton)
        print("Pop-up cerrado")
    except:
        print("No hay pop-up visible")

# Hacer scroll para cargar todos los elementos de la página
def scroll_hasta_cargar_todo(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Hacer scroll hasta el final de la página
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Esperar que carguen los elementos
        # Calcular nueva altura de la página
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # Romper el bucle si no hay más contenido por cargar
        last_height = new_height  


def lectura_comentarios_exito(html):
    soup = BeautifulSoup(html, "html.parser")
    bloques = soup.select('div[data-fs-reviews-container="true"]')
    datos = []
    try:
        for bloque in bloques:
            propiedad = bloque.select_one('div[data-fs-review-message="true"]').get_text(strip=True).replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
            datos.append({"comentario": propiedad})
    except:
        datos.append({"comentario": "sin comentarios"})
    df = pd.DataFrame(datos).replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
    return df

def comentarios_exito (producto_url):
     # Iniciar el driver
    driver = iniciar_driver()
    url = producto_url
    driver.get(url)
    scroll_hasta_cargar_todo(driver)
    time.sleep(random.uniform(1, 3))
    score= WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-fs-average-score="true"]'))).text
    boton_busqueda = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".list-reviews-and-ratings_fs-list-ratings-and-reviews__ujkLS .drawer_openDrawer__9Jx_r")))
    clic_expandir = driver.find_element(By.CSS_SELECTOR, '.list-reviews-and-ratings_fs-list-ratings-and-reviews__ujkLS .drawer_openDrawer__9Jx_r')
    driver.execute_script("arguments[0].click();", clic_expandir)
    time.sleep(random.uniform(1,3))   
    try:
        seccion = driver.find_element(By.CSS_SELECTOR, '[data-fs-reviews-drawer="true"]')
        contenido_html = seccion.get_attribute("outerHTML")
        # Extraer información y convertirla en DataFrame
        df = lectura_comentarios_exito(contenido_html).replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
        df["score"]=score
    except Exception as e:
        df["comentario"]="sin comentarios"
    driver.quit()
    return df

# Extraer comentarios de la página actual
def extraer_comentarios_alkosto(driver):
    try:
        reviews_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "yotpo-reviews-container"))
        )
        reviews = reviews_container.find_elements(By.CLASS_NAME, "yotpo-review")

        data = []
        for review in reviews:
            try:
                content = review.find_element(By.CLASS_NAME, "yotpo-read-more-text").text.replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
                data.append({'comentario': content.strip()})
            except NoSuchElementException:
                data.append({'comentario': "sin comentarios"})                
        return data
    except TimeoutException:
        data.append({'comentario': "sin comentarios"})
        return []
# Procesar la paginación y extraer datos
def procesar_paginacion_alkosto(driver):
    df_total = pd.DataFrame()
    while True:
        # Extraer datos de la página actual
        comentarios = extraer_comentarios_alkosto(driver)
        if comentarios:
            df_total = pd.concat([df_total, pd.DataFrame(comentarios)], ignore_index=True)
        # Verificar si existe el botón "Siguiente" y está habilitado
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.yotpo-reviews-pagination-item[aria-label='Goto next page']"))
            )
            if next_button.get_attribute("aria-disabled") == "true":
                print("No hay más páginas disponibles.")
                break
            next_button.click()
            time.sleep(random.uniform(1, 3))
        except TimeoutException:
            print("Error: No se encontró el botón 'Siguiente' o está deshabilitado.")
            break
    return df_total

def comentarios_alkosto(producto_url):
    driver = iniciar_driver()
    url = producto_url
    driver.get(url)
    scroll_hasta_cargar_todo(driver)
    score = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.yotpo-bottom-line-left-panel.yotpo-bottom-line-score'))).text.replace(",","")
    df_comentarios = procesar_paginacion_alkosto(driver)
    df_comentarios["score"]=score
    driver.quit()
    return df_comentarios
    

# Función para extraer información
def extraer_informacion_falabella(contenido_html):
    soup = BeautifulSoup(contenido_html, "html.parser")
    reseñas = soup.select("div._review-text_xpeer_69")  # Selector para cada reseña
    datos = []
    for reseña in reseñas:
        try:
            texto_reseña = reseña.select_one("div._review-text_16yc3_2").get_text(strip=True).replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
            datos.append({"comentario": texto_reseña})
        except Exception as e:
            datos.append({"comentario": "sin comentarios"})
    df = pd.DataFrame(datos).replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
    return df

def comentarios_falabella(producto_url):
    driver = iniciar_driver()
    url = producto_url
    driver.get(url)
    scroll_hasta_cargar_todo(driver)
    time.sleep(random.uniform(1, 3))
    score = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '._numerator_7vdzo_22'))).text
    while True:
        try:
            boton_busqueda = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "._more-comments-button_1lnif_33"))
            )
            boton_busqueda.click()
            time.sleep(random.uniform(1, 3))
        except:
            print("No se encontró más el botón, saliendo del bucle.")
            break
    
    time.sleep(random.uniform(1, 3))
    #_review-text_16yc3_2
    try:
        # Esperar la sección que contiene la tabla de especificaciones
        seccion = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='user-reviews-container']")))
        contenido_html = seccion.get_attribute("outerHTML")       
        # Extraer información
        df = extraer_informacion_falabella(contenido_html)
        df["score"]=score
    except Exception as e:
        print(f"Error al extraer información: {e}")
    finally:
        driver.quit()
    
    return df

def scroll_en_cuadro(driver, cuadro_selector):
    """Realiza scroll dentro de un cuadro específico hasta cargar todo el contenido."""
    cuadro = driver.find_element(By.CSS_SELECTOR, cuadro_selector)
    last_height = driver.execute_script("return arguments[0].scrollHeight", cuadro)
    
    while True:
        # Realizar scroll hacia el final del cuadro
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", cuadro)
        time.sleep(random.uniform(1, 3))  # Esperar para cargar contenido
        
        # Obtener nueva altura
        new_height = driver.execute_script("return arguments[0].scrollHeight", cuadro)
        if new_height == last_height:
            break
        last_height = new_height

def extraer_comentarios_mercado(html):
    """Extrae los comentarios desde el HTML."""
    soup = BeautifulSoup(html, "html.parser")
    data = []

    # Buscar todas las opiniones dentro de artículos
    opiniones = soup.find_all("article", {"data-testid": "comment-component"})
    for opinion in opiniones:
        try:
            comentario = opinion.find("p", {"data-testid": "comment-content-component"})
            if comentario:
                texto_comentario = comentario.text.strip()
                # Remover caracteres no deseados
                texto_comentario = texto_comentario.replace(",", "").replace(";", "").replace(".", "").replace("\n", "").replace("\r", "")
            else:
                texto_comentario = "sin comentarios"
        except AttributeError:
            texto_comentario = "sin comentarios" 
        data.append({"comentario": texto_comentario})
    df=pd.DataFrame(data).replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
    return df

def comentarios_mercado (producto_url):
     # Iniciar el driver
    driver = iniciar_driver()
    url = producto_url
    driver.get(url)
    time.sleep(3)  # Esperar a que cargue la página
    scroll_hasta_cargar_todo(driver)
    
    try:
        # Esperar a que el elemento de la calificación sea visible
        score_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-review-capability__rating__average.ui-review-capability__rating__average--desktop")))
        # Extraer el texto de la calificación
        score = score_element.text
    except Exception as e:
        print(f"Error al extraer la calificación: {e}")
    print(score)
        
    try:
        # Clic en el botón para abrir la ventana superpuesta
        boton_busqueda = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button.show-more-click')))
        driver.execute_script("arguments[0].click();", boton_busqueda)
        time.sleep(random.uniform(1,3))
    except Exception as e:
        print(f"Error: {e}")
        
    try:
        # Cambiar al iframe de los comentarios
        iframe = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ui-pdp-iframe-reviews"))
        )
        driver.switch_to.frame(iframe)
        # Extraer el HTML del cuadro después del scroll
        seccion = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.ui-review-capability-comments')))
        contenido_html = seccion.get_attribute("outerHTML")
        # Extraer comentarios y convertirlos en DataFrame
        df = extraer_comentarios_mercado(contenido_html)
        df["score"]=score

    except Exception as e:
        print(f"Error al extraer comentarios: {e}")
    finally:
        # Volver al contexto principal y cerrar el driver
        driver.switch_to.default_content()
        driver.quit()
    
    return df

def comentarios_metro (producto_url):
    # Iniciar el driver
    driver = iniciar_driver()
    url = producto_url
    driver.get(url)

    scroll_hasta_cargar_todo(driver)
    time.sleep(random.uniform(1, 3))
    print("0")
    
    score = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.vtex-reviews-and-ratings-3-x-reviewsRatingAverage.review__rating--average'))).text
    print(score)    
    
    try:
        # Esperar a que las reseñas carguen
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "vtex-reviews-and-ratings-3-x-reviewCommentMessage"))
        )

        # Extraer todos los comentarios
        comentarios = driver.find_elements(By.CLASS_NAME, "vtex-reviews-and-ratings-3-x-reviewCommentMessage")
        
        # Lista para almacenar comentarios
        lista_comentarios = []

        for comentario in comentarios:
            try:
                # Extraer el texto visible (evitar "Mostrar más")
                texto = comentario.find_element(By.CSS_SELECTOR, "div > span > span").text.replace(",","").replace(";","").replace(".","").replace("\n", "").replace("\r", "")
                lista_comentarios.append(texto)
            except Exception as e:
                lista_comentarios.append("sin comentarios")
    except:
        lista_comentarios.append("sin comentarios")
    
    driver.quit()
    df=pd.DataFrame()
    df["score"]=score
    df["comentario"]=lista_comentarios
    return df

# Función que procesa los productos
def procesar_producto(url_producto, tienda, funcion, nombre):
    try:
        # Ejecutar la función asociada a la tienda
        print(url_producto)
        df_producto = funcion(url_producto)
        # Agregar columna con el identificador del producto y la tienda
        df_producto["Tienda"] = tienda
        df_producto["URL Producto"] = url_producto
        df_producto["Nombre"] = nombre
        return df_producto
    except Exception as e:
        # Crear un DataFrame con error de producto no disponible
        df_producto = pd.DataFrame({
            "Tienda": [tienda],
            "URL Producto": [url_producto],
            "Nombre": [nombre],
            "score": ["sin score"],
            "comentario": ["sin comentarios"]
        })
        return df_producto

# Función para guardar checkpoints
def guardar_checkpoint(index, filename="checkpoint_comentarios.csv"):
    checkpoint_df = pd.DataFrame({"index": [index]})
    checkpoint_df.to_csv(filename, index=False)

# Función para cargar el checkpoint (si existe)
def cargar_checkpoint(filename="checkpoint_comentarios.csv"):
    if os.path.exists(filename):
        checkpoint_df = pd.read_csv(filename)
        print(f"Checkpoint cargado. Último índice procesado: {checkpoint_df['index'].iloc[0]}")
        return checkpoint_df["index"].iloc[0]
    else:
        return 0

# Función para cargar resultados parciales (si existen)
def cargar_resultados_parciales(filename="resultados_parciales_scraping_comentarios.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding="utf-8")
    else:
        return pd.DataFrame()

# Función para guardar el tiempo de procesamiento
def guardar_tiempo_worker(index, nombre, elapsed_time, filename="tiempo_workers_comentarios.csv"):
    write_header = not os.path.exists(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Index", "Nombre", "Tiempo Transcurrido (s)"])
        writer.writerow([index, nombre, elapsed_time])


if __name__ == "__main__":   
    funciones_por_tienda = {
        "https://www.alkosto.com/": comentarios_alkosto,
        "https://www.exito.com/": comentarios_exito,
        "https://www.falabella.com.co/falabella-co": comentarios_falabella,
        "https://www.mercadolibre.com.co/": comentarios_mercado,
        "https://www.tiendasmetro.co/": comentarios_metro
    }

    # Leer el archivo CSV con las URLs de productos
    df_urls = pd.read_csv("consolidado_url.csv", encoding="utf-8", delimiter=",")
    df_urls.columns = df_urls.columns.str.strip()
    df_urls = df_urls.drop_duplicates(["Nombre"])
    print(df_urls.head(7))
    print(df_urls.shape)
    
    #Cargar resultados parciales y checkpoint
    df_resultados = cargar_resultados_parciales()
    last_processed_index = cargar_checkpoint()

    # Filtrar las filas no procesadas
    if not df_resultados.empty:
        # Obtener los nombres de los productos ya procesados
        nombres_procesados = df_resultados["Nombre"].unique()
        # Filtrar df_urls para obtener solo las filas no procesadas
        df_urls = df_urls[~df_urls["Nombre"].isin(nombres_procesados)]
        print(f"Filas no procesadas: {df_urls.shape[0]}")
        
    # Crear un ThreadPoolExecutor para la paralelización
    tasks = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for index, row in df_urls.iterrows():
            if index < last_processed_index:
                continue  # Saltar índices ya procesados

            tienda = row["Tienda"]
            url_producto = row["URL Producto"]
            nombre = row["Nombre"]
            funcion = funciones_por_tienda.get(tienda)

            if funcion:
                start_time = time.time()
                future = executor.submit(procesar_producto, url_producto, tienda, funcion, nombre)
                tasks.append((index, nombre, start_time, future))
            else:
                print(f"No hay función definida para la tienda {tienda}.")

        # Procesar resultados a medida que se completan
        for index, nombre, start_time, future in tasks:
            try:
                df_producto = future.result()
                elapsed_time = time.time() - start_time
                guardar_tiempo_worker(index, nombre, elapsed_time)

                # Acumular los resultados
                df_resultados = pd.concat([df_resultados, df_producto], ignore_index=True)

                # Guardado incremental cada 100 registros procesados
                if index % 50 == 0:
                    guardar_checkpoint(index)
                    df_resultados.to_csv("resultados_parciales_scraping_comentarios.csv", index=False, encoding="utf-8")
                    print(f"Checkpoint guardado en el índice {index}. Resultados parciales actualizados.")
            except Exception as e:
                print(f"Error procesando el índice {index}: {e}")

    # Guardar el DataFrame final con todos los resultados
    df_resultados.to_csv("resultados_scraping_comentarios.csv", index=False, encoding="utf-8")
    print("Scraping finalizado. Resultados guardados en 'comentarios_scraping3.csv'.")


