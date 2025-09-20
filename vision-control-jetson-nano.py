# Versión del control pensada para ser ejecutada en una Jetson Nano (4GB).
# Para configurar todo lo necesario seguir la guia en: https://jetson-docs.com/libraries/mediapipe/overview
# No recomiendo utilizar contenedores de Docker por la latencia que introduce al procesamiento

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

# Fuente desde la cual se va a obtener el feed de video
# Con 0 simplemente abrimos la camara web default de la PC
# Puede abrirse también una URl, por ejemplo: "http://192.168.0.147:4747/video"
FUENTE_VIDEO = "http://192.168.0.147:4747/video"
USAR_GPU = True
CONFIDENCE = 0.5        # Valor para configuraciones del modelo "min_hand_detection_confidence", "min_hand_presence_confidence" y "min_tracking_confidence"
EVITAR_COLA = True      # Forzamos el que no se genere cola de frames permitiendo solo enviar un frame a procesar cuando ya se terminó de procesar el anterior

PUNTO_INICIAL = 0           # 0 = Muñeca
PUNTO_FINAL = 12            # 12 = Dedo mayor
UMBRAL_MANO_CERRADA = -0.02 # Factor para determinar si la mano está cerrada o no
LIMITE_PENDIENTE = 0.5      # Valor máximo que puede tener la pendiente (tanto positiva como negativa)
DEADZONE_STICK = 6000       # Deadzone del stick, si el calculo con la pendiente da un valor menor que este se reemplaza por 0

# Colores para OpenCV (Azul, Verde, Rojo)
ROJO = (0, 0, 255)
VERDE = (0, 255, 0)
AZUL = (255, 0, 0)
CYAN = (255, 255, 0)
AMARILLO = (0, 255, 255)

# Lista donde se colocan los puntos para los cuales se obtienen las coordenadas
# Si bien no se puede evitar el que modelo prediga los 21 landmarks
# Evitamos calcular las coordenadas de todos estos para después no usarlas
puntos_utilizados = [PUNTO_INICIAL, PUNTO_FINAL]

# Variable global donde se guarda el último frame procesado
ultimo_resultado = None

# Variable que se usa para evitar que se forme una cola de frames y se retrase el video
# Solo se usa si tenemos el flag 'EVITAR_COLA' activado
procesando = False

# Lock que se usa para no acceder a la variable 'procesando' en simultaneo
# Se puede terminar de procesar un frame y llamar a la función de callback cuando se está modificando la variable
# Solo se usa si tenemos el flag 'EVITAR_COLA' activado
lock = threading.Lock()

# Medimos la distancia entre dos puntos
def calcular_distancia(punto_1, punto_2):
    return math.hypot(punto_2[0] - punto_1[0], punto_2[1] - punto_1[1])

# Calculamos la pendiente de la recta que se forma al unir dos puntos
def calcular_pendiente(punto_1, punto_2):
    # Verificamos no estar dividiendo por cero si por alguna razón las muñecas tienen la misma coordenada en X
    return (punto_2[1] - punto_1[1])/(punto_2[0] - punto_1[0]) if((punto_2[0] - punto_1[0]) != 0) else 0

# Limitamos el valor de la pendiente
def limitar_pendiente(pendiente):
    return max(pendiente, -LIMITE_PENDIENTE) if (pendiente < 0) else min(pendiente, LIMITE_PENDIENTE)

# Dibuja en un frame todos los elementos relacionados a una mano
def dibujar_mano(frame, mano_etiqueta, estado, punto_inicial, punto_final, distancia):
    # Escribimos el estado de la mano
    cv2.putText(frame, f"{mano_etiqueta}: {estado}", (punto_inicial[0], punto_inicial[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

    # Dibujamos una línea entre la muñeca y el dedo mayor
    cv2.line(frame, punto_inicial, punto_final, AMARILLO, 2)
    
    # Escribimos la distancia del dedo mayor al centro de la mano estimada por el modelo
    cv2.putText(frame, f"{abs(round(distancia * 100, 2))}cm",
        ((punto_inicial[0] + punto_final[0]) // 2 + 10, (punto_inicial[1] + punto_final[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMARILLO, 2)

    # Marcamos con un circulo tanto el punto inicial como el final
    cv2.circle(frame, punto_inicial, 8, AZUL, -1)
    cv2.circle(frame, punto_final, 8, ROJO, -1)

    return frame

# Actualiza los valores del joystick dependiendo del estado de las manos
def control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada):
    
    # Calculamos y limitamos la pendiente que se forma entre las dos muñecas
    pendiente_limitada = limitar_pendiente(calcular_pendiente(punto_izquierda, punto_derecha))

    # Traducimos el valor de la pendiente en un valor del stick entre -32767 y 32767
    valor_stick_x = int(32767 * (pendiente_limitada / LIMITE_PENDIENTE))

    # Aplicamos la deadzone si es necesario
    if -DEADZONE_STICK < valor_stick_x < DEADZONE_STICK:
        valor_stick_x = 0

    return valor_stick_x

# Callback que recibe los resultados del procesamiento de las imágenes
# Devuelve el resultado, la imagen original y el timestamp de la imagen original
# Guardamos el resultado en la variable 'ultimo_resultado' para usarlo en el loop principal
def callback(result, output_image, timestamp_ms):
    global ultimo_resultado, procesando
    ultimo_resultado = result
    with lock:
        # Habilitamos a procesar un nuevo frame
        procesando = False

# Configuración de Mediapipe
# Elegimos si queremos usar o no la GPU para acelerar el procesamiento
if(not USAR_GPU): base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

# Forzamos a que el modelo de Mediapipe corra en la GPU (Solo soportado en Linux por ahora
else: base_options = python.BaseOptions(model_asset_path="hand_landmarker.task", delegate=python.BaseOptions.Delegate.GPU)

options = vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # Configuramos modo de video en vivo
    num_hands = 2,                                          # Cantidad máxima de manos a trackear
    min_hand_detection_confidence = CONFIDENCE,
    min_hand_presence_confidence = CONFIDENCE,
    min_tracking_confidence = CONFIDENCE,
    result_callback = callback
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(FUENTE_VIDEO)
frame_id = 0

# Último estado registrado de cada mano
# Primer Elemento   -> Par de coordenadas X e Y
# Segundo Elemento  -> Cerrada o Abierta (0 cerrada, 1 abierta, -1 sin registrar todavia)
ultimo_estado = {'Derecha': [(0, 0), -1], 'Izquierda': [(0, 0), -1]}

# Últimas coordenadas que tuvimos cuando habia dos manos reconocidas
ultimo_estado_ambas_manos = {'Derecha': [(0, 0), -1], 'Izquierda': [(0, 0), -1]}

# Bucle principal del programa
while True:
    # Leemos un frame del feed de video
    ret, frame = cap.read()

    # La función devuelve false si no hay un frame
    if not ret:
        print("No se pudo inicializar el feed de video")
        break

    # Invertimos la imagen verticalmente
    frame = cv2.flip(frame, 1)

    # Obtenemos las dimensiones de la imagen (alto-ancho-canales)
    imagen_alto, imagen_ancho, _ = frame.shape

    # Creamos una imagen que pueda ser utilizada por mediapipe, la guardamos en color
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)

    # Dependiendo del método de procesamiento que hayamos elegido pueden pasar dos cosas:
    # Se procesan todos los frames, donde tecnicamente se agregan a una cola y se pueden acumular
    # Se procesan frames solo cuando sabemos que no estamos procesando otro frame, evitando la acumulación
    if EVITAR_COLA:
        with lock:
            if not procesando:
                procesando = True

                # Enviamos el frame al detector, cuando el procesamiento esté listo va a llamar a la función 'callback'
                detector.detect_async(mp_image, frame_id)
    else:
        detector.detect_async(mp_image, frame_id)
    
    frame_id += 1   # Esta variable no se usa pero se debe llevar la cuenta de los frames
    
    # Creamos un diccionario para usar a la hora de controlar el joystick, cada ciclo se reinicia para que refleje los resultados de la detección
    manos_detectadas = {}

    # Calculamos todo en función del último frame procesado
    # Verificamos que se hayan detectado las dos manos para entrar en el ciclo inicialmente, una vez que tenemos un historial permitimos detectar solo una
    # Podemos elegir entre 'hand_landmarks' o 'hand_world_landmarks', el primero nos da las coordenadas en la imagen y el otro la distancia en metros hacia el centro de la mano
    # Más información en: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
    if (ultimo_resultado and (
        (len(ultimo_resultado.hand_landmarks) == 2 and len(ultimo_resultado.hand_world_landmarks) == 2) or
        (len(ultimo_resultado.hand_landmarks) == 1 and len(ultimo_resultado.hand_world_landmarks) == 1 and ultimo_estado['Derecha'][1] != -1 and ultimo_estado['Izquierda'][1] != -1))):

        # Ejecutamos una vez por cada mano
        for nro_mano, (mano_derecha, mano_3d) in enumerate(zip(ultimo_resultado.hand_landmarks, ultimo_resultado.hand_world_landmarks)):
            # Guardamos las coordenadas de los puntos de referencia que definimos al inicio del programa
            puntos_referencia = {
                i: (int(mano_derecha[i].x * imagen_ancho), int(mano_derecha[i].y * imagen_alto))
                for i in puntos_utilizados
            }

            # Determinamos si la mano está cerrada en función de la distancia del dedo mayor en referencia al centro de la mano
            # El modelo devuelve la distancia en metros, por lo tanto funciona sin importar a que distancia estamos de la cámara
            mano_cerrada = mano_3d[PUNTO_FINAL].y > UMBRAL_MANO_CERRADA

            # El modelo también puede estimar de que mano se trata con el atributo "handedness"
            # Invertimos la selección porque el feed de video fue invertido antes
            mano_etiqueta = 'Izquierda' if ultimo_resultado.handedness[nro_mano][0].category_name == 'Right' else 'Derecha'
            estado = 'CERRADA' if mano_cerrada else 'ABIERTA'
            
            # Guardamos las coordenadas de la muñeca y si la mano está cerrada o no
            manos_detectadas[mano_etiqueta] = {'pos': puntos_referencia[PUNTO_INICIAL], 'cerrada': mano_cerrada}

            # Actualizamos el valor en el diccionario
            ultimo_estado[mano_etiqueta][0] = puntos_referencia[PUNTO_INICIAL]
            ultimo_estado[mano_etiqueta][1] = mano_cerrada

            # Solo se ejecuta cuando estamos detectando ambas manos
            if(len(ultimo_resultado.hand_landmarks) == 2):
                # Actualizamos el valor en el diccionario
                ultimo_estado_ambas_manos[mano_etiqueta][0] = puntos_referencia[PUNTO_INICIAL]
                ultimo_estado_ambas_manos[mano_etiqueta][1] = mano_cerrada

            # Dibujamos las referencias en el frame para esta mano
            frame = dibujar_mano(frame, mano_etiqueta, estado, puntos_referencia[PUNTO_INICIAL], puntos_referencia[PUNTO_FINAL], mano_3d[PUNTO_FINAL].y)
    

    # Si no detectamos ninguna mano todos los valores del control serán cero
    else:
        valor_stick_x = 0

    # Control de joystick
    # Si detectamos ambas manos calculamos la pendiente normalmente
    if 'Izquierda' in manos_detectadas and 'Derecha' in manos_detectadas:
        # Datos de la manos detectadas
        punto_izquierda = manos_detectadas['Izquierda']['pos']
        punto_derecha = manos_detectadas["Derecha"]['pos']
        mano_izquierda_cerrada = manos_detectadas['Izquierda']['cerrada']
        mano_derecha_cerrada = manos_detectadas["Derecha"]['cerrada']

        # Actualizamos los valores del control emulado
        valor_stick_x = control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada)
    
    # Detectamos solamente a la mano izquierda, tenemos que intentar predecir el valor de la otra mano
    elif 'Izquierda' in manos_detectadas:
        # Datos de la mano detectada
        punto_izquierda = manos_detectadas['Izquierda']['pos']
        mano_izquierda_cerrada = manos_detectadas['Izquierda']['cerrada']
        
        # Calculamos la diferencia por cada coordenada y se la aplicamos en sentido contrario al punto de la derecha
        diferencia_punto_izquierda = tuple(p1 - p2 for p1, p2 in zip(punto_izquierda, ultimo_estado_ambas_manos['Izquierda'][0]))
        punto_derecha = tuple(p1 - p2 for p1, p2 in zip(ultimo_estado_ambas_manos['Derecha'][0], diferencia_punto_izquierda))
        ultimo_estado['Derecha'][0] = punto_derecha
        
        # El estado de apertura de la mano derecha es el último registrado
        mano_derecha_cerrada = ultimo_estado_ambas_manos['Derecha'][1]

        # Dibujamos un circulo sobre las coordenadas predecidas
        cv2.circle(frame, punto_derecha, 8, AZUL, -1)

        # Actualizamos los valores del control emulado
        valor_stick_x = control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada)

    # Detectamos solamente a la mano derecha, tenemos que intentar predecir el valor de la otra mano
    elif 'Derecha' in manos_detectadas:
        # Datos de la mano detectada
        punto_derecha = manos_detectadas['Derecha']['pos']
        mano_derecha_cerrada = manos_detectadas['Derecha']['cerrada']
        
        # Calculamos la diferencia por cada coordenada y se la aplicamos en sentido contrario al punto de la izquierda
        diferencia_punto_derecha = tuple(p1 - p2 for p1, p2 in zip(punto_derecha, ultimo_estado_ambas_manos['Derecha'][0]))
        punto_izquierda = tuple(p1 - p2 for p1, p2 in zip(ultimo_estado_ambas_manos['Izquierda'][0], diferencia_punto_derecha))
        ultimo_estado['Izquierda'][0] = punto_izquierda
        
        # El estado de apertura de la mano izquierda es el último registrado
        mano_izquierda_cerrada = ultimo_estado_ambas_manos['Izquierda'][1]

        # Dibujamos un circulo sobre las coordenadas predecidas
        cv2.circle(frame, punto_izquierda, 8, AZUL, -1)

        # Actualizamos los valores del control emulado
        valor_stick_x = control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada)

    
    # Estamos detectando alguna mano
    if 'Izquierda' in manos_detectadas or 'Derecha' in manos_detectadas:
        # Dibujamos una línea entre ambas muñecas
        cv2.line(frame, ultimo_estado['Derecha'][0], ultimo_estado['Izquierda'][0], VERDE, 2)

    # Escribimos en el frame el valor que le corresponde al analógico
    cv2.putText(frame, f"Stick: {valor_stick_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, VERDE, 2)

    # Mostramos el frame
    cv2.imshow("Volante Vision Artificial", frame)

    # Salimos con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
