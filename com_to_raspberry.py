import socket

def send_command( action ):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.1.10', 12346))  # ใส่ IP Address ของ Raspberry Pi ที่เชื่อมต่ออยู่
    
    client_socket.sendall( action.encode('utf-8'))#2-

    response = client_socket.recv(1024).decode('utf-8')#3
    print(f'Response from server: {response}')

    client_socket.sendall( '0'.encode('utf-8'))#4- ถ้าใส่'1'จะเป็นการปิด client_socket.close
    response1 = client_socket.recv(1024).decode('utf-8')#5
    #print(f'Response from server: {response1}')
    if (response1 == 1):
        print("client_socket.close")
        client_socket.close()

#if _name_ == '_main_':

    #action = input("0, 1, 2 -> ") 
    #print(type(action))
    #send_command( action )