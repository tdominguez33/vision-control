import cv2
import mediapipe as mp
import math
import vgamepad as vg

PUNTO_INICIAL = 0           # 0 = Muñeca
PUNTO_FINAL = 12            # 12 = Dedo mayor
MULTIPLICADOR_RANGO = 0.5   # A partir de que punto del rango total de movimiento consideramos que la mano está cerrada

maximaDistancia = 0
minimaDistancia = 1000

# Inicializar gamepad virtual - Emulamos un joystick de Xbox 360
gamepad = vg.VX360Gamepad()

# Inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands=2, min_detection_confidence = 0.7, min_tracking_confidence = 0.5)

cap = cv2.VideoCapture(0)

# Función que se utiliza para limitar el ángulo que se forma con las manos
def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))

# Medimos la distancia entre dos puntos
def distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Ciclo principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    altura, ancho, _ = frame.shape
    manos_detectadas = {}

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label  # "Left" o "Right"

            puntos = {
                i: (int(hand_landmarks.landmark[i].x * ancho),
                    int(hand_landmarks.landmark[i].y * altura))
                for i in range(21)
            }

            # Dibujar línea entre la muñeca y el pulgar
            cv2.line(frame, puntos[PUNTO_INICIAL], puntos[PUNTO_FINAL], (0, 255, 255), 2)

            # Calcular e imprimir la distancia
            distancia_medio_muneca = int(distancia(puntos[PUNTO_INICIAL], puntos[PUNTO_FINAL]))
            # Si la distancia es mayor que el máximo registrado lo guardamos en la variable
            if(distancia_medio_muneca > maximaDistancia):
                maximaDistancia = distancia_medio_muneca
            # Si la distancia es menor que el mínimo registrado lo guardamos en la variable
            elif(distancia_medio_muneca < minimaDistancia):
                minimaDistancia = distancia_medio_muneca

            cv2.putText(frame, f"{distancia_medio_muneca}px",
                        ((puntos[PUNTO_INICIAL][0] + puntos[PUNTO_FINAL][0]) // 2, (puntos[PUNTO_INICIAL][1] + puntos[PUNTO_FINAL][1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.circle(frame, puntos[PUNTO_INICIAL], 8, (255, 0, 0), -1)  # puntos[0] = Muñeca
            cv2.circle(frame, puntos[PUNTO_FINAL], 8, (0, 255, 0), -1) # puntos[12] = Punta del dedo medio

            # Determinamos si la mano está cerrada en función de las distancias máximas y mínimas registradas
            # Mano cerrada si el índice está cerca de la muñeca
            dist = distancia(puntos[PUNTO_INICIAL], puntos[PUNTO_FINAL])
            rangoMovimiento = maximaDistancia - minimaDistancia
            mano_cerrada = dist < minimaDistancia + (MULTIPLICADOR_RANGO * rangoMovimiento)

            manos_detectadas[hand_label] = {
                'pos': puntos[PUNTO_INICIAL],
                'cerrada': mano_cerrada
            }

            # Dibujar estado
            estado = "CERRADA" if mano_cerrada else "ABIERTA"
            cv2.putText(frame, f"{hand_label}: {estado}", (puntos[PUNTO_INICIAL][0], puntos[PUNTO_INICIAL][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Si ambas manos están presentes, calcular dirección del stick
        if "Left" in manos_detectadas and "Right" in manos_detectadas:
            p1 = manos_detectadas["Left"]["pos"]
            p2 = manos_detectadas["Right"]["pos"]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            angulo = math.atan2(dy, dx) * (180 / math.pi)
            angulo_limitado = limitar(angulo, -40, 40)

            valor_stick_x = int((angulo_limitado / 40) * 32767)

            # Deadzone
            if (-6000 < valor_stick_x < 6000):
                valor_stick_x = 0

            # Presionar botones si las manos están cerradas
            # Acelerar
            if(manos_detectadas["Left"]["cerrada"]) and ((manos_detectadas["Right"]["cerrada"])):
                gamepad.left_trigger(value=0)
                gamepad.right_trigger(value=255)
            
            # No hacer nada
            elif (manos_detectadas["Left"]["cerrada"]) or ((manos_detectadas["Right"]["cerrada"])):
                gamepad.left_trigger(value=0)
                gamepad.right_trigger(value=0)

            # Frenar
            elif (not manos_detectadas["Left"]["cerrada"]) or ((not manos_detectadas["Right"]["cerrada"])):
                gamepad.left_trigger(value=255)
                gamepad.right_trigger(value=0)

        else:
            valor_stick_x = 0

    else:
        # No hay dos manos detectadas, ponemos todos los botones en cero
        valor_stick_x = 0
        gamepad.left_trigger(value=0)
        gamepad.right_trigger(value=0)

    # Enviar valores al gamepad
    gamepad.left_joystick(x_value=valor_stick_x, y_value=0)
    gamepad.update()

    # Mostrar valor del stick
    cv2.putText(frame, f"LX: {valor_stick_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Volante + Botones", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
