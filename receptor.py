# Programa utilizado para recibir los valores del joystick enviados por la Jetson Nano
# Emula un control de Xbox-360 en la PC con los valores recibidos

import socket
import vgamepad as vg

# Configuramos el socket UDP (SOCK_DGRAM)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Escuchamos en el puerto 5555 (Debe coincidir con el puerto en el programa de la Jetson Nano)
sock.bind("0.0.0.0", 5555)

# Creamos el gamepad virtual
gamepad = vg.VX360Gamepad()

print("Esperando datos desde la Jetson Nano...")

while True:
    data, _ = sock.recvfrom(1024)
    print("Datos recibidos:", data.decode().strip())

    try:
        stick_str, izquierda_str, derecha_str = data.decode().strip().split(',')
        valor_stick_x = int(stick_str)
        mano_izquierda_cerrada = izquierda_str == 'True'
        mano_derecha_cerrada = derecha_str == 'True'
    except Exception:
        continue

    # Aplicamos el valor del stick a nuestro control virtual
    gamepad.left_joystick(valor_stick_x, 0)

    # Si ambas manos están cerradas aceleramos
    if mano_izquierda_cerrada and mano_derecha_cerrada:
        gamepad.left_trigger(value=0)
        gamepad.right_trigger(value=255)
    
    # Si una sola de ambas manos está cerrada no hacemos nada
    elif mano_izquierda_cerrada or mano_derecha_cerrada:
        gamepad.left_trigger(value=0)
        gamepad.right_trigger(value=0)
    
    # Si abrimos ambas manos frenamos
    else:
        gamepad.left_trigger(value=255)
        gamepad.right_trigger(value=0)

    gamepad.update()
