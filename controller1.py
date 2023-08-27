import RPi.GPIO as GPIO
import time

# กำหนดโหมดของขา GPIO
GPIO.setmode(GPIO.BCM)

# กำหนดขา GPIO ที่ต้องการอ่านสถานะลอจิก
input_pin1 = 8
input_pin2 = 7

OUTPUT_GPIO_pin1 = 23
OUTPUT_GPIO_pin2 = 24
OUTPUT_GPIO_pin3 = 25

# กำหนดโหมดของขา GPIO
#GPIO.setmode(GPIO.BCM)
GPIO.setup(OUTPUT_GPIO_pin1, GPIO.OUT) #ขาเปิด
GPIO.setup(OUTPUT_GPIO_pin2, GPIO.OUT) #ขาปิด
GPIO.setup(OUTPUT_GPIO_pin3, GPIO.OUT) #แจ้งเตือนระบบมีปัญหา
GPIO.setup(input_pin1, GPIO.IN) # ลิมิตสวิต 1
GPIO.setup(input_pin2, GPIO.IN) # ลิมิตสวิต 2


def input_check():

    # อ่านสถานะลอจิกจากขา GPIO
    input_state1 = GPIO.input(input_pin1)
    input_state2 = GPIO.input(input_pin2)
    print("Input state1:", input_state1)
    print("Input state2:", input_state2)
    #if( input_state1 == input_state2 ): # ถ้าอ่านแล้วค่าของ sw เท่ากันหรือประตูค้าง จะมีแจ้งเตือน
        #GPIO.output(OUTPUT_GPIO_pin3, GPIO.HIGH)  

    # ปิดการใช้งาน GPIO ยังมีการเรียกใช้ทำให้ต้องนำไปใส่ใน main
    #GPIO.cleanup(input_pin1)
    #GPIO.cleanup(input_pin2)
    return input_state1,input_state2  # ส่งค่า input_state


def close():
    
    ip1 , ip2 = input_check()
    #print("close")
    #if( ip1 == 0 and ip2 == 0 ): # ถ้าอ่านแล้วค่าของ sw เท่ากันหรือประตูค้าง จะมีแจ้งเตือน
        #GPIO.output(OUTPUT_GPIO_pin3, GPIO.HIGH)  

    while  ip1 == 0 and ip2 == 1 :
        print("in loop")
        #ip1, ip2 = input_check()
        GPIO.output(OUTPUT_GPIO_pin1, GPIO.LOW)
        GPIO.output(OUTPUT_GPIO_pin2, GPIO.HIGH)
        #ip1 , ip2 = input_check()
        if input_check()[0] == 1 and input_check()[1] == 0: #เมื่อประตูอยู่ในตำแหน่งเปิดจะสั่งให้รีเลยทั้งสองตัวหยุดการทำงานและออกจาก loop while
            GPIO.output(OUTPUT_GPIO_pin1, GPIO.LOW) # สั้งให้ทั้้งสองขาลอจิกเป็น 0 เพื่อให้อยู่ในสถานะหยุดนิ่งไม่ทำอะไร แต่ จะต้องดูเงื่อนไขให้ดีไม่งั้นวนลูปแล้วจะทำให้เพี้นได้
            GPIO.output(OUTPUT_GPIO_pin2, GPIO.HIGH)
           # ip1 == 1 and ip2 == 0 
            break  # ออกจาก while loop
            #print("ปิดเรียบร้อย")
    freez_output()
    print("ปิดเรียบร้อย1")

    
def open():
    
    ip1 , ip2 = input_check()
    #print("open")
    #if( ip1 == 0 and ip2 == 0 ): # ถ้าอ่านแล้วค่าของ sw เท่ากันหรือประตูค้าง จะมีแจ้งเตือน
        #GPIO.output(OUTPUT_GPIO_pin3, GPIO.HIGH)  

    while  ip1 == 1 and ip2 == 0 :
        print("in loop")
        GPIO.output(OUTPUT_GPIO_pin1, GPIO.HIGH)
        GPIO.output(OUTPUT_GPIO_pin2, GPIO.LOW)
        #ip1 , ip2 = input_check()
        if input_check()[0] == 0 and input_check()[1] == 1: #เมื่อประตูอยู่ในตำแหน่งปิดจะสั่งให้รีเลยทั้งสองตัวหยุดการทำงานและออกจาก loop while
            #GPIO.output(OUTPUT_GPIO_pin1, GPIO.LOW)# สั้งให้ทั้้งสองขาลอจิกเป็น 0 เพื่อให้อยู่ในสถานะหยุดนิ่งไม่ทำอะไร แต่ จะต้องดูเงื่อนไขให้ดีไม่งั้นวนลูปแล้วจะทำให้เพี้นได้
            #GPIO.output(OUTPUT_GPIO_pin2, GPIO.LOW)
            break  # ออกจาก while loop
            #print("เปิดเรียบร้อย") 
    freez_output()
    print("เปิดเรียบร้อย1")


def freez_output():
    GPIO.output(OUTPUT_GPIO_pin1, GPIO.HIGH)#relay active logic 1
    GPIO.output(OUTPUT_GPIO_pin2, GPIO.HIGH)#relay active logic 1
    GPIO.output(OUTPUT_GPIO_pin3, GPIO.LOW)

def main_controller(n):
    freez_output()
    print(" เปิด = 0  ปิด = 1 ")
    #n = int(input())
    ip1 , ip2 = input_check()
    GPIO.output(OUTPUT_GPIO_pin3, GPIO.LOW)
    freez_output()

    if( n == 0 and ip1 == 1 and ip2 == 0 ):
        open(  )
    elif ( n == 1 and ip1 == 0 and ip2 == 1):
        close(  )
    elif ( n == 2 ):
        freez_output()
    elif( ip1 == 0 and ip2 == 0 ): # ถ้าอ่านแล้วค่าของ sw เท่ากันหรือประตูค้าง จะมีแจ้งเตือน
        GPIO.output(OUTPUT_GPIO_pin3, GPIO.HIGH)
        time.sleep(1)  # พักเวลา 1 วินาที
    elif( ip1 == 1 and ip2 == 1 ): # ถ้าอ่านแล้วค่าของ sw เท่ากันหรือประตูค้าง จะมีแจ้งเตือน
        GPIO.output(OUTPUT_GPIO_pin3, GPIO.HIGH)
        time.sleep(1)  # พักเวลา 1 วินาที

    # ปิดการใช้งาน GPIO
    GPIO.cleanup(input_pin1)
    GPIO.cleanup(input_pin2)
    GPIO.cleanup(OUTPUT_GPIO_pin3)
    
#n = int(input())    
#main_controller(n)
    