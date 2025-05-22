from io import BytesIO
import cv2
import numpy as np
from ultralytics import YOLO
import time
from dotenv import load_dotenv
import os
import discord
import asyncio
import re
from datetime import datetime, timedelta

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
client = discord.Client(intents=intents)

imagery = {
    "warn": cv2.imread("warn.png")
}

tasks_for_users = {}

@client.event
async def on_ready():
    print(f'OpenCCTV está listo como {client.user}')

async def notificar(user_id, text, image):
    user = await client.fetch_user(int(user_id))  # fetch_user es más robusto que get_user
    if user is None:
        print("Usuario no encontrado.")
        return

    try:
        dm_channel = await user.create_dm()
        await dm_channel.send(content=text, file=cv2discordfile(image))
        print(f"Notificamos a {user.name}.")
    except Exception as e:
        print(f"Error notificando al usuario: {e}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    ip_a_vigilar = extraer_ip(message.content)

    try:
        dm_channel = await message.author.create_dm()
        if message.content == 'finalizar' or message.content == 'terminé':
            try:
                tasks_for_users[str(message.author.id)].cancel()
                del tasks_for_users[str(message.author.id)]
                await dm_channel.send(
                    f"Finalicé la vigilancia para tu hogar."
                )
            except:
                await dm_channel.send(
                    f"No encontré ninguna sesión activa."
                )
            return
        elif str(message.author.id) in tasks_for_users:
            await dm_channel.send(
                f"Ya me encuentro monitoreando tu hogar y no puedo abrir otra sesión. Si deseas terminar la actual, escribe \"finalizar\" o \"terminé\"."
            )
            return
        elif ip_a_vigilar:
            await dm_channel.send(
                f"Hola {message.author.name}. Con gusto monitoreo tu hogar."
            )
            tasks_for_users[str(message.author.id)] = asyncio.create_task(vigilar(ip_a_vigilar, message.author.id))
            print(tasks_for_users)
        else:
            await dm_channel.send("No encontré una IP válida en tu mensaje.")
    except discord.Forbidden:
        print(f"❌ No se pueden enviar DMs al usuario {message.author}")
        await message.channel.send("No puedo enviarte mensajes directos. Por favor activa tus DMs.")
    except Exception as e:
        print(f"Error inesperado en on_message: {e}")

def extraer_ip(texto):
    patron = r"(?:http:\/\/)?(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?"
    coincidencia = re.search(patron, texto)
    if coincidencia:
        ip = coincidencia.group(1)
        puerto = coincidencia.group(2) or "8080"
        return f"http://{ip}:{puerto}/video"
    return None

# --- Funciones de Preprocesamiento ---
def filtroGaussiano(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)
    
def filtroBordes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,150,250)
    return cv2.bitwise_or(image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

# Mejora la iluminación de un frame si el brillo promedio del canal V está por debajo de un umbral.
# Aplica CLAHE y/o Corrección Gamma al canal V del espacio HSV.
def mejorar_iluminacion_hsv(frame, v_threshold, apply_clahe=True, apply_gamma=True, gamma_val=1.5):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    avg_v = np.mean(v)
    # print(f"Brillo promedio (V channel): {avg_v}") # Descomentar para depurar

    if avg_v < v_threshold:
        print(f"Detectada baja iluminación (V_avg={avg_v:.2f} < {v_threshold}). Aplicando mejoras...")
        v_mejorado = v.copy()
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v_mejorado = clahe.apply(v_mejorado)
            print("  CLAHE aplicado al canal V.")
        
        if apply_gamma:
            invGamma = 1.0 / gamma_val
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            v_mejorado = cv2.LUT(v_mejorado, table)
            print(f"  Corrección Gamma (valor {gamma_val}) aplicada al canal V.")
        
        final_hsv = cv2.merge([h, s, v_mejorado])
        frame_mejorado = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return frame_mejorado
    else:
        return frame # Devuelve el frame original si no se necesita mejora

# Aplica operaciones morfológicas (apertura o cierre) a una máscara binaria.
def aplicar_operaciones_morfologicas(mascara_binaria, operacion="apertura", tamano_kernel=3):
    kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)
    if operacion == "apertura":
        return cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel)
    elif operacion == "cierre":
        return cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel)
    else: # Por defecto o si el tipo no es reconocido, devuelve la máscara original
        return mascara_binaria
    
# Filtro de sharpening
def enfocar_imagen_sharpening(frame):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened_frame = cv2.filter2D(frame, -1, kernel_sharpening)
    return sharpened_frame

def cv2discordfile(img):
    img_encode = cv2.imencode('.png', img)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    byteImage = BytesIO(byte_encode)
    image=discord.File(byteImage, filename='image.png')
    return image

# Variables globales para control de notificaciones
ultima_notificacion_por_usuario = {}
intervalo_min_notificacion = timedelta(seconds=30)

async def vigilar(video_ip, user_id):
    # --- Parámetros Configurables ---
    VIDEO_SOURCE = video_ip
    RESIZE_WIDTH = 640
    GAUSSIAN_KSIZE = 5
    MOTION_THRESHOLD_DIFF = 15
    MOTION_MIN_AREA = 250
    YOLO_MODEL_PATH = "yolov8n-seg.pt"
    YOLO_CONFIDENCE_THRESHOLD = 0.65
    PERSON_CLASS_ID = 0
    MAX_HISTORY_FRAMES_DURATION_SEC = 1.0 # Duración basada en frames PROCESADOS

    # --- Parámetros para Filtrado de Siluetas ---
    FILTER_SILHOUETTE = True
    MIN_PERSON_ASPECT_RATIO = 1.0
    MAX_PERSON_ASPECT_RATIO = 4.5
    MIN_PERSON_HEIGHT_PERCENT = 0.15
    MIN_PERSON_WIDTH_PERCENT = 0.05

    # --- Parámetros para Detección de Rostros ---
    DETECT_FACES = True
    FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Asegúrate de que este archivo esté accesible
    ROI_FACE_HEIGHT_PERCENT = 0.5

    # --- Parámetros de Optimización ---
    PROCESS_EVERY_N_FRAMES = 5 # Procesar 1 de cada N frames cuando hay movimiento. Ajusta según sea necesario.
    DRAW_LAST_KNOWN_SILHOUETTE_ON_SKIP = True # Dibujar la última silueta en frames saltados

    # --- Parámetros para Mejoras de Imagen Clásicas ---
    # 1. Mejora en Baja Luminosidad
    APPLY_LOW_LIGHT_ENHANCEMENT = True # Poner en True para activar
    V_CHANNEL_LOW_THRESHOLD = 80      # Umbral para el canal V (brillo) para aplicar la mejora (0-255)
    APPLY_CLAHE_ON_V = True           # Aplicar CLAHE al canal V si está en baja luz
    APPLY_GAMMA_ON_V = True           # Aplicar Corrección Gamma al canal V si está en baja luz
    GAMMA_CORRECTION_VALUE = 1.5      # Valor de Gamma (ej: 1.5 para aclarar)

    # 2. Operaciones Morfológicas para Máscara de Movimiento
    APPLY_MORPHOLOGICAL_OPS = True   # Poner en True para activar
    MORPH_OPERATION_TYPE = "apertura" # "apertura" o "cierre"
    MORPH_KERNEL_SIZE = 3             # Tamaño del kernel (ej: 3, 5)

    # 3. Sharpening (Enfoque)
    APPLY_SHARPENING = False          # Poner en True para activar

    # --- Cargar Modelo YOLO ---
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo YOLO: {e}")
        exit()

    # --- Cargar Clasificador de Rostros ---
    if DETECT_FACES:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            print(f"Error: No se pudo cargar el clasificador de rostros de Haar desde {FACE_CASCADE_PATH}")
            DETECT_FACES = False

    # --- Inicialización de Captura de Video ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el stream de video desde {VIDEO_SOURCE}")
        await notificar(user_id=user_id, text="No se pudo abrir el stream de video. La cámara podría estar apagada o mal configurada.", image=imagery["warn"])
        del tasks_for_users[str(user_id)]
        return
            

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0 or fps_video > 100: # Algunas cámaras IP dan valores extraños
        print(f"Advertencia: FPS del video ({fps_video}) no confiable, usando 30 por defecto.")
        fps_video = 30
    max_frames_in_history = int(fps_video * MAX_HISTORY_FRAMES_DURATION_SEC) # Basado en frames procesados

    ret, frame_anterior = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame del video.")
        await notificar(user_id=user_id, text="No se pudo leer el primer frame del video. Verifica la conexión con la cámara.", image=imagery["warn"])
        cap.release()
        del tasks_for_users[str(user_id)]
        return
    
    frame_height_orig, frame_width_orig = frame_anterior.shape[:2]
    if RESIZE_WIDTH:
        scale = RESIZE_WIDTH / frame_width_orig
        new_height = int(frame_height_orig * scale)
        frame_anterior = cv2.resize(frame_anterior, (RESIZE_WIDTH, new_height))
        processed_frame_height, processed_frame_width = new_height, RESIZE_WIDTH
    else:
        processed_frame_height, processed_frame_width = frame_height_orig, frame_width_orig

    min_person_height_px = processed_frame_height * MIN_PERSON_HEIGHT_PERCENT
    min_person_width_px = processed_frame_width * MIN_PERSON_WIDTH_PERCENT

    frame_anterior_gray = filtroGaussiano(cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY), GAUSSIAN_KSIZE)
    historial_deteccion = []
    persona_detectada_previamente = False # Estado general si se detectó una persona recientemente
    last_detected_face_img = None
    motion_frame_processing_counter = 0 # Contador para el N-ésimo frame
    perform_full_processing_now = False

    # Variables para dibujar la última silueta conocida en frames saltados
    last_known_puntos_silueta_display = None
    last_known_centro_persona_display = None

    print("Iniciando detección...")
    print(f"Procesando 1 de cada {PROCESS_EVERY_N_FRAMES} frames con movimiento.")

    # --- Bucle Principal de Procesamiento ---
    while True:
        if motion_frame_processing_counter % 5 == 0:
            await asyncio.sleep(0)  # Cede el control al event loop
        ret, frame_actual_orig = cap.read()
        if not ret:
            print("Se terminó el stream de video o hubo un error.")
            await notificar(user_id=user_id, text="Se desconectó la cámara.", image=imagery["warn"])
            break

        if RESIZE_WIDTH:
            frame_actual = cv2.resize(frame_actual_orig, (RESIZE_WIDTH, new_height))
        else:
            frame_actual = frame_actual_orig.copy()

        # --- APLICAR MEJORAS GENERALES AL FRAME ---
        if APPLY_LOW_LIGHT_ENHANCEMENT:
            frame_actual = mejorar_iluminacion_hsv(frame_actual, 
                                                V_CHANNEL_LOW_THRESHOLD,
                                                APPLY_CLAHE_ON_V,
                                                APPLY_GAMMA_ON_V,
                                                GAMMA_CORRECTION_VALUE)
        
        if APPLY_SHARPENING:
            frame_actual = enfocar_imagen_sharpening(frame_actual)

        frame_display = frame_actual.copy() # Para dibujar todo
        frame_gray_motion = filtroGaussiano(cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY), GAUSSIAN_KSIZE)
        diferencia = cv2.absdiff(frame_anterior_gray, frame_gray_motion)
        _, umbral = cv2.threshold(diferencia, MOTION_THRESHOLD_DIFF, 255, cv2.THRESH_BINARY)

         # --- APLICAR OPERACIONES MORFOLÓGICAS A LA MÁSCARA DE UMBRAL ---
        if APPLY_MORPHOLOGICAL_OPS:
            umbral = aplicar_operaciones_morfologicas(umbral, 
                                                    MORPH_OPERATION_TYPE, 
                                                    MORPH_KERNEL_SIZE)
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hay_movimiento_general = any(cv2.contourArea(c) > MOTION_MIN_AREA for c in contornos)
        
        # Esta bandera indica si en ESTA iteración específica se confirmó una persona tras el PROCESAMIENTO COMPLETO
        person_confirmed_in_current_processing_iteration = False
        rostro_detectado_en_frame_actual = False # Específico para la iteración actual de procesamiento

        if hay_movimiento_general:
            motion_frame_processing_counter += 1
            perform_full_processing_now = (motion_frame_processing_counter % PROCESS_EVERY_N_FRAMES == 0)

            if perform_full_processing_now:
                results = model(frame_actual, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                # Resetear para esta iteración de procesamiento
                temp_puntos_silueta = None
                temp_centro_persona = None

                for result in results:
                    if result.masks is None or result.boxes is None:
                        continue
                    for i, mask_coords in enumerate(result.masks.xy):
                        class_id = int(result.boxes.cls[i])
                        if class_id == PERSON_CLASS_ID:
                            puntos_silueta = np.array(mask_coords, dtype=np.int32)
                            if len(puntos_silueta) < 3: continue

                            x_br_person, y_br_person, w_br_person, h_br_person = cv2.boundingRect(puntos_silueta)
                            es_silueta_valida = True
                            if FILTER_SILHOUETTE:
                                if w_br_person == 0 or h_br_person == 0: es_silueta_valida = False
                                if es_silueta_valida:
                                    aspect_ratio = h_br_person / w_br_person
                                    if not (MIN_PERSON_ASPECT_RATIO <= aspect_ratio <= MAX_PERSON_ASPECT_RATIO): es_silueta_valida = False
                                if es_silueta_valida:
                                    if h_br_person < min_person_height_px or w_br_person < min_person_width_px: es_silueta_valida = False
                            
                            if not es_silueta_valida: continue

                            person_confirmed_in_current_processing_iteration = True
                            temp_puntos_silueta = puntos_silueta.copy() # Guardar para este frame procesado
                            
                            # Actualizar para visualización en frames saltados
                            last_known_puntos_silueta_display = temp_puntos_silueta 
                            
                            cv2.polylines(frame_display, [temp_puntos_silueta], isClosed=True, color=(0, 255, 0), thickness=2)
                            # frame_actual_para_historial = frame_actual.copy() # Crear copia ANTES de dibujar el centro si no lo quieres en la superposición
                            # cv2.polylines(frame_actual_para_historial, [temp_puntos_silueta], isClosed=True, color=(0, 255, 0), thickness=2)

                            M = cv2.moments(temp_puntos_silueta)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                temp_centro_persona = (cx, cy)
                                last_known_centro_persona_display = temp_centro_persona # Actualizar para display
                                cv2.circle(frame_display, temp_centro_persona, 5, (255, 0, 0), -1)
                                
                                # Usar frame_actual original (con la silueta dibujada si se desea para la trayectoria)
                                frame_con_silueta_para_historial = frame_actual.copy()
                                cv2.polylines(frame_con_silueta_para_historial, [temp_puntos_silueta], isClosed=True, color=(0, 255, 0), thickness=1) # Dibujar silueta en frame para historial
                                historial_deteccion.append((frame_con_silueta_para_historial, temp_centro_persona))
                            else:
                                person_confirmed_in_current_processing_iteration = False # Centro no calculable
                                last_known_puntos_silueta_display = None # Invalidar si no hay centro
                                last_known_centro_persona_display = None
                                continue

                            if DETECT_FACES and person_confirmed_in_current_processing_iteration:
                                face_roi_y_start = y_br_person
                                face_roi_y_end = y_br_person + int(h_br_person * ROI_FACE_HEIGHT_PERCENT)
                                face_roi_x_start = x_br_person
                                face_roi_x_end = x_br_person + w_br_person
                                face_roi_y_start = max(0, face_roi_y_start); face_roi_y_end = min(frame_actual.shape[0], face_roi_y_end)
                                face_roi_x_start = max(0, face_roi_x_start); face_roi_x_end = min(frame_actual.shape[1], face_roi_x_end)

                                if face_roi_y_end > face_roi_y_start and face_roi_x_end > face_roi_x_start:
                                    person_head_roi_color = frame_actual[face_roi_y_start:face_roi_y_end, face_roi_x_start:face_roi_x_end]
                                    person_head_roi_gray = cv2.cvtColor(person_head_roi_color, cv2.COLOR_BGR2GRAY)
                                    faces_in_roi = face_cascade.detectMultiScale(person_head_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                                    if len(faces_in_roi) > 0:
                                        (fx, fy, fw, fh) = faces_in_roi[0]
                                        detected_face_img = person_head_roi_color[fy:fy+fh, fx:fx+fw]
                                        if detected_face_img.size > 0:
                                            last_detected_face_img = detected_face_img.copy()
                                            rostro_detectado_en_frame_actual = True # Para la ventana de rostro
                                            global_fx = face_roi_x_start + fx; global_fy = face_roi_y_start + fy
                                            cv2.rectangle(frame_display, (global_fx, global_fy), (global_fx + fw, global_fy + fh), (0,0,255), 2)
                                            cv2.putText(frame_display, "Rostro", (global_fx, global_fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                            break # Procesar solo la primera persona válida
                    if person_confirmed_in_current_processing_iteration: break # Salir del bucle de results
                
                # Si el procesamiento completo se ejecutó y NO se encontró persona, limpiar las "últimas conocidas"
                if not person_confirmed_in_current_processing_iteration:
                    last_known_puntos_silueta_display = None
                    last_known_centro_persona_display = None
            
            elif DRAW_LAST_KNOWN_SILHOUETTE_ON_SKIP and last_known_puntos_silueta_display is not None:
                # Es un frame saltado, pero había una persona detectada recientemente
                cv2.polylines(frame_display, [last_known_puntos_silueta_display], isClosed=True, color=(120, 255, 120), thickness=1) # Color más tenue
                if last_known_centro_persona_display:
                    cv2.circle(frame_display, last_known_centro_persona_display, 4, (255, 120, 120), -1) # Color más tenue

        # Lógica de estado general y texto
        if person_confirmed_in_current_processing_iteration:
            cv2.putText(frame_display, "Persona (Detectada)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ahora = datetime.now()
            ultima_not = ultima_notificacion_por_usuario.get(user_id)
            if ultima_not is None or ahora - ultima_not > intervalo_min_notificacion:
                await notificar(user_id=user_id, text="Se ha detectado a la siguiente persona en movimiento.", image=frame_con_silueta_para_historial)
                ultima_notificacion_por_usuario[user_id] = ahora
            persona_detectada_previamente = True
        elif not perform_full_processing_now and hay_movimiento_general and last_known_puntos_silueta_display is not None:
            # Estamos en un frame saltado pero siguiendo a alguien visualmente
            cv2.putText(frame_display, "Persona (Siguiendo)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
            # persona_detectada_previamente sigue True de la última detección real
        else: # No hay persona confirmada en esta iteración (o no hubo procesamiento completo que la confirmara)
            if persona_detectada_previamente: # Si antes había y ahora no (o se interrumpió movimiento)
                print("Detección de persona interrumpida / Sin movimiento.")
                historial_deteccion.clear()
                last_known_puntos_silueta_display = None # Limpiar para que no se dibuje más
                last_known_centro_persona_display = None
            persona_detectada_previamente = False # Actualizar estado general
            
            if hay_movimiento_general:
                cv2.putText(frame_display, "Movimiento (sin persona)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else: # No hay movimiento general
                cv2.putText(frame_display, "Sin Movimiento", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
                # Si no hay movimiento general, también se interrumpe la detección de persona
                if persona_detectada_previamente: historial_deteccion.clear() # Redundante, pero seguro
                last_known_puntos_silueta_display = None 
                last_known_centro_persona_display = None
                persona_detectada_previamente = False


        # Mostrar ventana de rostro si hay una imagen de rostro guardada
        if last_detected_face_img is not None:
            cv2.imshow("Rostro Detectado", last_detected_face_img)
            ahora = datetime.now()
            ultima_not = ultima_notificacion_por_usuario.get(user_id)
            if ultima_not is None or ahora - ultima_not > intervalo_min_notificacion:
                await notificar(user_id=user_id, text="¿Has visto a esta persona antes? Está parado enfrente de tu casa.", image=last_detected_face_img)
                ultima_notificacion_por_usuario[user_id] = ahora
                last_detected_face_img = None

        # Procesar historial para trayectoria
        if len(historial_deteccion) >= max_frames_in_history:
            print(f"⚠️ Movimiento sostenido de persona detectado ({len(historial_deteccion)} frames procesados)")
            (primer_frame_hist, primer_centro), (_, ultimo_centro) = historial_deteccion[0], historial_deteccion[-1]
            (fx, fy), (lx, ly) = primer_centro, ultimo_centro
            dx, dy = lx - fx, ly - fy
            min_desplazamiento_dir = 10 
            direccion = "estática"
            if dx > min_desplazamiento_dir: direccion = "derecha"
            elif dx < -min_desplazamiento_dir: direccion = "izquierda"
            elif dy > min_desplazamiento_dir: direccion = "abajo"
            elif dy < -min_desplazamiento_dir: direccion = "arriba"
            distancia_pixeles = np.sqrt(dx**2 + dy**2)
            
            # El tiempo ahora se basa en los frames procesados y el intervalo entre ellos
            # Si cada frame en historial_deteccion representa PROCESS_EVERY_N_FRAMES frames reales (aproximadamente)
            tiempo_total_seg_estimado = (len(historial_deteccion) * PROCESS_EVERY_N_FRAMES) / fps_video
            # O más simple, si max_frames_in_history define la duración en frames procesados:
            # tiempo_total_seg = len(historial_deteccion) / fps_video_efectivo_procesamiento
            # donde fps_video_efectivo_procesamiento = fps_video / PROCESS_EVERY_N_FRAMES
            # Por ahora, la duración original de 1 segundo (MAX_HISTORY_FRAMES_DURATION_SEC) se refiere a que
            # el historial contendrá `fps_video * 1.0` puntos de datos que fueron muestreados cada N frames.
            # El tiempo real que cubren esos puntos es MAX_HISTORY_FRAMES_DURATION_SEC * PROCESS_EVERY_N_FRAMES
            
            # Para la velocidad, usamos el número de puntos en el historial y asumimos que fueron capturados
            # a una tasa efectiva de fps_video / PROCESS_EVERY_N_FRAMES
            tiempo_entre_puntos_historial_seg = PROCESS_EVERY_N_FRAMES / fps_video
            tiempo_total_para_velocidad_seg = (len(historial_deteccion) -1) * tiempo_entre_puntos_historial_seg if len(historial_deteccion) > 1 else tiempo_entre_puntos_historial_seg

            velocidad_aparente_px_s = distancia_pixeles / tiempo_total_para_velocidad_seg if tiempo_total_para_velocidad_seg > 0 else 0
            
            print(f"  Dirección: {direccion}")
            print(f"  Velocidad aparente: {velocidad_aparente_px_s:.2f} px/s (sobre {tiempo_total_para_velocidad_seg:.2f}s)")
            
            superposicion = np.zeros_like(primer_frame_hist, dtype=np.float32)
            num_frames_historial = len(historial_deteccion)
            alpha = 1.0 / num_frames_historial if num_frames_historial > 0 else 1.0
            for frame_hist, _ in historial_deteccion:
                superposicion += frame_hist.astype(np.float32) * alpha
            superposicion = np.clip(superposicion, 0, 255).astype(np.uint8)
            for i_hist in range(len(historial_deteccion)):
                centro_actual = historial_deteccion[i_hist][1]
                cv2.circle(superposicion, centro_actual, 3, (0, 0, 255), -1)
                if i_hist > 0:
                    centro_anterior = historial_deteccion[i_hist-1][1]
                    cv2.line(superposicion, centro_anterior, centro_actual, (0,0,255),1)
            cv2.imshow("Trayectoria Detectada", superposicion)
            historial_deteccion.clear()

        frame_anterior_gray = frame_gray_motion
        cv2.imshow("Video en Tiempo Real - Deteccion de Siluetas", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saliendo...")
            del tasks_for_users[str(user_id)]
            break

    cap.release()
    cv2.destroyAllWindows()

client.run(os.environ["TOKEN"])