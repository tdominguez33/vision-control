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
# Evitamos calcular las coordenadas de los 21 posibles landmarks
puntos_utilizados = [PUNTO_INICIAL, PUNTO_FINAL]

maximaDistancia = -1
minimaDistancia = -1

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

# === Configuración de Mediapipe ===
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

    # Calculamos todo en función del último frame procesado
    # Verificamos que se hayan detectado las dos manos para entrar en el ciclo
    # Podemos elegir entre 'hand_landmarks' o 'hand_world_landmarks', el primero nos da las coordenadas en la imagen y el otro la distancia en metros hacia el centro de la mano
    if ultimo_resultado and len(ultimo_resultado.hand_landmarks) == 2 and len(ultimo_resultado.hand_world_landmarks) == 2:
        # Creamos un diccionario para usar a la hora de controlar el joystick, cada ciclo se reinicia para poder actuar solo cuando tenemos las dos manos
        manos_detectadas = {}

        # Ejecutamos una vez por cada mano
        for nro_mano, (mano_2d, mano_3d) in enumerate(zip(ultimo_resultado.hand_landmarks, ultimo_resultado.hand_world_landmarks)):
            # Guardamos las coordenadas de los puntos de referencia que definimos al inicio del programa
            puntos_referencia = {
                i: (int(mano_2d[i].x * imagen_ancho), int(mano_2d[i].y * imagen_alto))
                for i in puntos_utilizados
            }

            # Determinamos si la mano está cerrada en función de la distancia del dedo mayor en referencia al centro de la mano
            # El modelo devuelve la distancia en metros, por lo tanto funciona sin importar a que distancia estamos de la cámara
            mano_cerrada = mano_3d[PUNTO_FINAL].y > UMBRAL_MANO_CERRADA

            mano_etiqueta = "Izquierda" if nro_mano == 1 else "Derecha"
            estado = "CERRADA" if mano_cerrada else "ABIERTA"
            
            # Guardamos las coordenadas de la muñeca y si la mano está cerrada o no
            manos_detectadas[mano_etiqueta] = {'pos': puntos_referencia[PUNTO_INICIAL], 'cerrada': mano_cerrada}

            # Escribimos el texto en el frame
            cv2.putText(frame, f"{mano_etiqueta}: {estado}", (puntos_referencia[PUNTO_INICIAL][0], puntos_referencia[PUNTO_INICIAL][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # A partir de este punto todo es completamente estéticos, no se utilizan para calcular nada
            # Dibujar línea entre la muñeca y el pulgar
            cv2.line(frame, puntos_referencia[PUNTO_INICIAL], puntos_referencia[PUNTO_FINAL], (0, 255, 255), 2)

            # Calculamos la distancia entre ambos puntos
            distancia = int(calcular_distancia(puntos_referencia[PUNTO_INICIAL], puntos_referencia[PUNTO_FINAL]))

            cv2.putText(frame, f"{distancia}px",
                        ((puntos_referencia[PUNTO_INICIAL][0] + puntos_referencia[PUNTO_FINAL][0]) // 2,
                         (puntos_referencia[PUNTO_INICIAL][1] + puntos_referencia[PUNTO_FINAL][1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.circle(frame, puntos_referencia[PUNTO_INICIAL], 8, (255, 0, 0), -1)
            cv2.circle(frame, puntos_referencia[PUNTO_FINAL], 8, (0, 255, 0), -1)
                
            
        # Control de joystick
        if "Izquierda" in manos_detectadas and "Derecha" in manos_detectadas:
            punto_1 = manos_detectadas["Izquierda"]["pos"]
            punto_2 = manos_detectadas["Derecha"]["pos"]

            # Calculamos y limitamos la pendiente que se forma entre las dos muñecas
            pendiente_limitada = limitar_pendiente(calcular_pendiente(punto_1, punto_2))

            # Traducimos el valor de la pendiente en un valor del stick entre -32767 y 32767
            valor_stick_x = int(32767 * (pendiente_limitada / LIMITE_PENDIENTE))

            # Aplicamos la deadzone si es necesario
            if -DEADZONE_STICK < valor_stick_x < DEADZONE_STICK:
                valor_stick_x = 0

            # Seteamos el valor del stick con el nuevo valor calculado
            gamepad.left_joystick(x_value = valor_stick_x, y_value = 0)

            # Acelerar
            if manos_detectadas["Izquierda"]["cerrada"] and manos_detectadas["Derecha"]["cerrada"]:
                gamepad.left_trigger(value = 0)
                gamepad.right_trigger(value = 255)
            # Nada
            elif manos_detectadas["Izquierda"]["cerrada"] or manos_detectadas["Derecha"]["cerrada"]:
                gamepad.left_trigger(value = 0)
                gamepad.right_trigger(value = 0)
            # Frenar
            else:
                gamepad.left_trigger(value = 255)
                gamepad.right_trigger(value = 0)

        else:
            valor_stick_x = 0

        cv2.putText(frame, f"LX: {valor_stick_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # No hay manos detectadas
        gamepad.left_trigger(value = 0)
        gamepad.right_trigger(value = 0)
        gamepad.left_joystick(x_value = 0, y_value = 0)

    # Actualizamos el control con los nuevos valores
    gamepad.update()

    # Mostramos el frame
    cv2.imshow("Volante Vision Artificial", frame)
    
    # Salimos con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
