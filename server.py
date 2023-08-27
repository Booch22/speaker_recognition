import controller1
#controller1.main_controller()
import socket


def control_door(action):
    controller1.freez_output()
    if action == 0 :
        controller1.main_controller( action )
    elif action == 1 :
        controller1.main_controller( action )
    elif action == 2 :
        n = None


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))
    server_socket.listen(1)

    print('Server listening on port 12345...')

    while True:
        client_socket, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        predict = client_socket.recv(1024).decode('utf-8')
        
        predict = int(predict)
        print(f'Received data: {predict}')
        print(type(predict))
        
        if predict == 0 :
            client_socket.sendall("Open the door".encode('utf-8'))
        elif predict == 1 :
            client_socket.sendall("Close the door".encode('utf-8'))
        elif predict == 2 :
            client_socket.sendall("Do nothing".encode('utf-8'))
        
        #control_door(predict)
        
        #client_socket.sendall('process finish'.encode('utf-8'))
        #client_socket.sendall("OK AM DOING -__-".encode('utf-8'))
        client_socket.close()

if __name__ == '__main__':
    
    start_server()
