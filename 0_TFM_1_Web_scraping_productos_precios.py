from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
from datetime import datetime
import time
import random
import unicodedata
from datetime import datetime


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

def eliminar_tildes(texto):
    # Normalizar el texto eliminando las tildes
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

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

# Función para cargar más productos
def cargar_mas_productos(driver):
    while True:
        try:
            time.sleep(random.uniform(3, 9))
            scroll_hasta_cargar_todo(driver)
            # Espera a que el botón esté presente
            boton_cargar_mas = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button.generic-load-more-button"))
            )
            # Verifica si el botón está habilitado
            if "disabled" in boton_cargar_mas.get_attribute("class"):
                print("El botón 'Mostrar más productos' está deshabilitado. Fin de la carga.")
                break  # Rompe el bucle si el botón está deshabilitado

            # Si el botón está habilitado, haz clic en él
            boton_cargar_mas.click()
            print("Cargando más productos...")
            time.sleep(random.uniform(3, 9))  # Espera un momento para que se carguen más productos

        except Exception as e:
            print(f"Ocurrió un error: {str(e)}")
            break  # Rompe el bucle si hay un error

# Función para extraer información de productos
def extraer_productos(driver, url, producto):
    lista_productos = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'product-list-container'))
    )
    elementos_productos = lista_productos.find_elements(By.CLASS_NAME, "product-item")
    time.sleep(random.uniform(3, 9))
    df_scraping = []
    print("Extrayendo información de productos...")
    for product in elementos_productos:
        try:
            df_scrap = []
            df_scrap.append(datetime.now().strftime("%Y-%m-%d"))
            df_scrap.append(url)
            df_scrap.append(producto)
            
            # URL del producto
            try:
                url_element = product.find_element(By.CSS_SELECTOR, 'a.product-link')
                url_producto = url_element.get_attribute("href")  
            except:
                url_producto = "Na"
            df_scrap.append(url_producto)
            
            # Imagen del producto
            try:
                image_element = "Na"
            except:
                image_element = "Na"
            df_scrap.append(image_element)
            
            # Nombre del producto
            try:
                nombre_element = product.find_element(By.CSS_SELECTOR, 'h3.product-title').text.strip()
            except:
                nombre_element = "Na"
            df_scrap.append(nombre_element)    
            
            # Marca del producto
            try:
                marca_element = product.find_element(By.CLASS_NAME, 'product-brand').text.strip()
            except:
                marca_element = "Na"
            df_scrap.append(marca_element)
            
            # Vendido por
            try:
                vendido_element = product.find_element(By.CSS_SELECTOR, 'span.seller-info').text.strip()
            except:
                vendido_element = "Na"
            df_scrap.append(vendido_element)
            
            # Puntaje del producto
            try:
                puntaje_element = product.find_element(By.CLASS_NAME, 'product-rating').text.strip()
            except:
                puntaje_element = "Na"
            df_scrap.append(puntaje_element)    
            
            # Precio sin descuento
            try:
                precio_sin_descuento = product.find_element(By.CSS_SELECTOR, 'span.old-price').text.strip()
            except:
                precio_sin_descuento = "Na"
            df_scrap.append(precio_sin_descuento)    
            
            # Precio descuento tienda
            try:
                precio_descuento_tienda = product.find_element(By.CSS_SELECTOR, 'span.discounted-price').text.strip()
            except:
                precio_descuento_tienda = "Na"
            df_scrap.append(precio_descuento_tienda)
            
            # Precio descuento aliados
            try:
                precio_descuento_aliados = product.find_element(By.CSS_SELECTOR, 'span.partner-price').text.strip()
            except:
                precio_descuento_aliados = "Na"
            df_scrap.append(precio_descuento_aliados)
            
            # Precio otros
            try:
                precio_otros = product.find_element(By.CSS_SELECTOR, 'span.other-price').text.strip()
            except:
                precio_otros = "Na"
            df_scrap.append(precio_otros)
            
            df_scraping.append(df_scrap)
        except Exception as e:
            print(f"Error al extraer información de un producto: {str(e)}")

    return df_scraping

# Función principal para realizar el scraping
def retail_1(item):
    # Lista de productos para realizar la búsqueda
    lista_productos = [item]
    # Iniciar el driver
    driver = iniciar_driver()
    url = "https://www.ejemplo.com/"
    driver.get(url)
    # DataFrame global para almacenar la información de todas las búsquedas
    df_total = pd.DataFrame()
    for producto in lista_productos:
        # Ingresar el producto en el campo de búsqueda
        campo_busqueda = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "search-input"))
        )
        campo_busqueda.clear()
        campo_busqueda.click()
        time.sleep(2)  # Esperar para asegurarse de que el campo esté vacío
        campo_busqueda.send_keys(producto)
        enter_busqueda = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "search-button"))
        )
        driver.execute_script("arguments[0].click();", enter_busqueda)
        
        cargar_mas_productos(driver) 
        df_scraping = extraer_productos(driver, url, producto)
        df_pagina = pd.DataFrame(df_scraping, columns=["Fecha", "Tienda", "Producto", "URL Producto", "Imagen", "Nombre", "Marca", "Vendido por", "Puntaje", "Precio sin descuento", "Precio descuento tienda", "Precio descuento aliados", "Precio otros"])
        df_total = pd.concat([df_total, df_pagina], ignore_index=True)
    # Cerrar el navegador
    driver.quit()    
    # Guardar el DataFrame en un archivo CSV
    nombre_archivo = f"{datetime.now().strftime('%Y%m%d')}_scraping_{producto}.csv"
    df_total.to_csv(nombre_archivo, index=False, encoding='utf-8')
    print(df_total.head(3))
    return
def retail_2(item):
    driver=()
    url=item
    producto=item
    extraer_productos(driver, url, producto)
    
def retail_3(item):
    driver=()
    url=item
    producto=item
    extraer_productos(driver, url, producto)
    
def retail_4(item):
    driver=()
    url=item
    producto=item
    extraer_productos(driver, url, producto)
    
def retail_5(item):
    driver=()
    url=item
    producto=item
    extraer_productos(driver, url, producto)
    
def realizar_scraping(producto):
    print(f"Lista de productos para realizar la búsqueda de {producto}")
    
    funciones = ["retail_1", "retail_2", "retail_3", "retail_4", "retail_5"]
    for funcion in funciones:
        try:
            if funcion == retail_1:
                funcion([producto])  # Pasar lista para metro
            else:
                funcion(producto)
        except Exception as e:
            print(f"Error en {funcion.__name__} para {producto}: {e}")
        
        time.sleep(random.uniform(3, 9))  # Pausa aleatoria entre ejecuciones

if __name__ == "__main__":
    print("Inicio de scraping")
    
    item = ["televisor", "tablet", "computador", "celular", "consola"]
    
    for producto in item:
        realizar_scraping(producto)