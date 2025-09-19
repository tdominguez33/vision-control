# Versión inicial del control, todas las funciones están contenidas en la misma PC
# Puede ser dificil de correr al mismo tiempo que se juega por la carga extra que el procesamiento pone en el CPU
# Pensado principalmente para Windows porque la libreria de vgamepad es más dificil de hacer correr en Linux

import cv2
import math
import vgamepad as vg
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PUNTO_INICIAL = 0           # 0 = Muñeca
PUNTO_FINAL = 12            # 12 = Dedo mayor
UMBRAL_MANO_CERRADA = -0.02 # Factor para determinar si la mano está cerrada o no
LIMITE_PENDIENTE = 0.5      # Valor máximo que puede tener la pendiente (tanto positiva como negativa)
DEADZONE_STICK = 6000       # Deadzone del stick, si el calculo con la pendiente da un valor menor que este se reemplaza por 0

# Lista donde se colocan los puntos para los cuales se obtienen las coordenadas
# Si bien no se puede evitar el que modelo prediga los 21 landmarks
# Evitamos calcular las coordenadas de estos para después no usarlas
puntos_utilizados = [PUNTO_INICIAL, PUNTO_FINAL]

# Inicializar gamepad virtual
gamepad = vg.VX360Gamepad()

# Variable global donde se guarda el último frame procesado
ultimo_resultado = None

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

# Callback que recibe los resultados del procesamiento de las imágenes
# Devuelve el resultado, la imagen original y el timestamp de la imagen original
# Guardamos el resultado en la variable 'ultimo_resultado' para usarlo en el loop principal
def callback(result, output_image, timestamp_ms):
    global ultimo_resultado
    ultimo_resultado = result

def dibujar_mano(frame, mano_etiqueta, estado, punto_inicial, punto_final):
    # Escribimos el texto en el frame
    cv2.putText(frame, f"{mano_etiqueta}: {estado}", (punto_inicial[0], punto_inicial[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Dibujamos una línea entre la muñeca y el dedo mayor
    cv2.line(frame, punto_inicial, punto_final, (0, 255, 255), 2)

    # Calculamos la distancia entre ambos puntos
    distancia = int(calcular_distancia(punto_inicial, punto_final))
    
    # Escribimos la distancia calculada
    cv2.putText(frame, f"{distancia}px",
        ((punto_inicial[0] + punto_final[0]) // 2,
        (punto_inicial[1] + punto_final[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Marcamos con un circulo tanto el punto inicial como el final
    cv2.circle(frame, punto_inicial, 8, (255, 0, 0), -1)
    cv2.circle(frame, punto_final, 8, (0, 0, 255), -1)

    return frame

def control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada):
    
    # Calculamos y limitamos la pendiente que se forma entre las dos muñecas
    pendiente_limitada = limitar_pendiente(calcular_pendiente(punto_izquierda, punto_derecha))

    # Traducimos el valor de la pendiente en un valor del stick entre -32767 y 32767
    valor_stick_x = int(32767 * (pendiente_limitada / LIMITE_PENDIENTE))

    # Aplicamos la deadzone si es necesario
    if -DEADZONE_STICK < valor_stick_x < DEADZONE_STICK:
        valor_stick_x = 0

    # Seteamos el valor del stick con el nuevo valor calculado
    gamepad.left_joystick(x_value = valor_stick_x, y_value = 0)

    # Acelerar
    if mano_izquierda_cerrada and mano_derecha_cerrada:
        gamepad.left_trigger(value = 0)
        gamepad.right_trigger(value = 255)
    # Nada
    elif mano_izquierda_cerrada or mano_derecha_cerrada:
        gamepad.left_trigger(value = 0)
        gamepad.right_trigger(value = 0)
    # Frenar
    else:
        gamepad.left_trigger(value = 255)
        gamepad.right_trigger(value = 0)

    # Actualizamos el control con los nuevos valores
    gamepad.update()

    return valor_stick_x

# Configuración de Mediapipe
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
    #delegate=python.BaseOptions.Delegate.GPU   # Forzamos a que el modelo de Mediapipe corra en la GPU (Solo soportado en Linux por ahora)
)

options = vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # Configuramos modo de video en vivo
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    result_callback = callback
)

detector = vision.HandLandmarker.create_from_options(options)

# Abrimos la camara web default de la PC
cap = cv2.VideoCapture(0)
frame_id = 0

# Último estado registrado de cada mano
# Primer Elemento   -> Par de coordenadas X e Y
# Segundo Elemento  -> Cerrada o Abierta (0 cerrada, 1 abierta, -1 sin registrar todavia)
ultimo_estado = {'derecha': [(0, 0), -1], 'izquierda': [(0, 0), -1]}

# Últimas coordenadas que tuvimos cuando habia dos manos reconocidas
ultimo_estado_ambas_manos = {'derecha': [(0, 0), -1], 'izquierda': [(0, 0), -1]}

# Bucle principal del programa
while True:
    # Leemos un frame del feed de video
    ret, frame = cap.read()

    # La función devuelve false si no hay un frame
    if not ret:
        break

    # Invertimos la imagen verticalmente
    frame = cv2.flip(frame, 1)

    # Obtenemos las dimensiones de la imagen (alto-ancho-canales)
    imagen_alto, imagen_ancho, _ = frame.shape

    # Creamos una imagen que pueda ser utilizada por mediapipe, la guardamos en color
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)

    # Enviamos el frame al detector, cuando el procesamiento esté listo va a llamar a la función 'callback'
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
        (len(ultimo_resultado.hand_landmarks) == 1 and len(ultimo_resultado.hand_world_landmarks) == 1 and ultimo_estado['derecha'][1] != -1 and ultimo_estado['izquierda'][1] != -1))):

        # Ejecutamos una vez por cada mano
        for nro_mano, (mano_derechad, mano_3d) in enumerate(zip(ultimo_resultado.hand_landmarks, ultimo_resultado.hand_world_landmarks)):
            # Guardamos las coordenadas de los puntos de referencia que definimos al inicio del programa
            puntos_referencia = {
                i: (int(mano_derechad[i].x * imagen_ancho), int(mano_derechad[i].y * imagen_alto))
                for i in puntos_utilizados
            }

            # Determinamos si la mano está cerrada en función de la distancia del dedo mayor en referencia al centro de la mano
            # El modelo devuelve la distancia en metros, por lo tanto funciona sin importar a que distancia estamos de la cámara
            mano_cerrada = mano_3d[PUNTO_FINAL].y > UMBRAL_MANO_CERRADA

            # Invertimos la selección porque el feed de video fue invertido antes
            mano_etiqueta = "izquierda" if ultimo_resultado.handedness[nro_mano][0].category_name == "Right" else "derecha"
            estado = "CERRADA" if mano_cerrada else "ABIERTA"
            
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
            frame = dibujar_mano(frame, mano_etiqueta, estado, puntos_referencia[PUNTO_INICIAL], puntos_referencia[PUNTO_FINAL])
    

    # Si no detectamos ninguna mano todos los valores del control serán cero
    else:
        valor_stick_x = 0
        gamepad.left_trigger(value = 0)
        gamepad.right_trigger(value = 0)
        gamepad.left_joystick(x_value = 0, y_value = 0)

    # Control de joystick
    # Si detectamos ambas manos calculamos la pendiente normalmente
    if "izquierda" in manos_detectadas and "derecha" in manos_detectadas:
        # Datos de la manos detectadas
        punto_izquierda = manos_detectadas['izquierda']['pos']
        punto_derecha = manos_detectadas["derecha"]['pos']
        mano_izquierda_cerrada = manos_detectadas['izquierda']['cerrada']
        mano_derecha_cerrada = manos_detectadas["derecha"]['cerrada']

        # Actualizamos los valores del control emulado
        valor_stick_x = control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada)
    
    # Detectamos solamente a la mano izquierda, tenemos que intentar predecir el valor de la otra mano
    elif "izquierda" in manos_detectadas:
        # Datos de la mano detectada
        punto_izquierda = manos_detectadas['izquierda']['pos']
        mano_izquierda_cerrada = manos_detectadas['izquierda']['cerrada']
        
        # Calculamos la diferencia por cada coordenada y se la aplicamos en sentido contrario al punto de la derecha
        diferencia_punto_izquierda = tuple(p1 - p2 for p1, p2 in zip(punto_izquierda, ultimo_estado_ambas_manos['izquierda'][0]))
        punto_derecha = tuple(p1 - p2 for p1, p2 in zip(ultimo_estado_ambas_manos['derecha'][0], diferencia_punto_izquierda))
        ultimo_estado['derecha'][0] = punto_derecha
        
        # El estado de apertura de la mano derecha es el último registrado
        mano_derecha_cerrada = ultimo_estado_ambas_manos['derecha'][1]

        # Dibujamos un circulo sobre las coordenadas predecidas
        cv2.circle(frame, punto_derecha, 8, (255, 0, 0), -1)

        # Actualizamos los valores del control emulado
        valor_stick_x = control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada)

    # Detectamos solamente a la mano derecha, tenemos que intentar predecir el valor de la otra mano
    elif "derecha" in manos_detectadas:
        # Datos de la mano detectada
        punto_derecha = manos_detectadas['derecha']['pos']
        mano_derecha_cerrada = manos_detectadas['derecha']['cerrada']
        
        # Calculamos la diferencia por cada coordenada y se la aplicamos en sentido contrario al punto de la izquierda
        diferencia_punto_derecha = tuple(p1 - p2 for p1, p2 in zip(punto_derecha, ultimo_estado_ambas_manos['derecha'][0]))
        punto_izquierda = tuple(p1 - p2 for p1, p2 in zip(ultimo_estado_ambas_manos['izquierda'][0], diferencia_punto_derecha))
        ultimo_estado['izquierda'][0] = punto_izquierda
        
        # El estado de apertura de la mano izquierda es el último registrado
        mano_izquierda_cerrada = ultimo_estado_ambas_manos['izquierda'][1]

        # Dibujamos un circulo sobre las coordenadas predecidas
        cv2.circle(frame, punto_izquierda, 8, (255, 0, 0), -1)

        # Actualizamos los valores del control emulado
        valor_stick_x = control_joystick(punto_izquierda, punto_derecha, mano_izquierda_cerrada, mano_derecha_cerrada)

    
    # Estamos detectando alguna mano
    if "izquierda" in manos_detectadas or "derecha" in manos_detectadas:
        # Dibujamos una línea entre ambas muñecas
        cv2.line(frame, ultimo_estado['derecha'][0], ultimo_estado['izquierda'][0], (0, 255, 0), 2)

    # Escribimos en el frame el valor que le corresponde al analógico
    cv2.putText(frame, f"LX: {valor_stick_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostramos el frame
    cv2.imshow("Volante Vision Artificial", frame)

    # Salimos con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
