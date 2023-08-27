import numpy as np
from python_speech_features import mfcc
import speech_recognition as sr
import noisereduce as nr
import time
import tensorflow as tf
import matplotlib.pyplot as plt
#import com_to_raspberry as C2R

def plot( Data2Plt ) :
    second = len( Data2Plt ) / Rate
    Time = np.linspace( 0, second, len(Data2Plt) ) #กำหนดขอบเขตของกราฟ เริ่มที่ 0, สิ้นสุดที่ 96256/44100 = 2, เแบ่งออกเป็น 96256 ส่วน (แกน x)
    Amplitude = Data2Plt 
    #พล็อตกราฟจากไฟล์ wav
    plt.title( "Waveform of audio" ) 
    plt.xlabel( "Time" ) 
    plt.ylabel( "Amplitude" ) 
    plt.plot( Time, Amplitude )
    plt.show()

############################### Test auto microphone ###############################
def Record() :
    r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
    r.energy_threshold = 10000 # ไมโครโฟนจะทำงานเมื่อมีเสียงพูด //7000
    try :
        with sr.Microphone() as source : # ใช้คำสั่ง with เปิดใช้ไมค์โดยตั้งชื่อว่า source
            print( "Listening...\n" )
            AudioData = r.listen( source ,timeout = 60 ) # phrase_time_limit = 3
            print( "Recording...\n" )
            RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
            RecData = np.frombuffer( RawData, dtype = "int16" ) # แปลง bytes เป็น int
    except :
        print( "Listening time out" )
        print( "Try again" )
        RecData = ReRecord( 3 )
    return RecData

def ReRecord( second ) :
    r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
    with sr.Microphone() as source :
        print( "Recording more...\n" )
        AudioData = r.record( source, duration = second ) # เก็บข้อมูลเสียง 3 วิ 
        RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
        ReRecData = np.frombuffer( RawData, dtype = "int16" ) # แปลง bytes เป็น int
    return ReRecData

def check_data( data ) : # เช็คข้อมูลหากข้อมูลมี len ไม่เท่ากับ 132096
    constant_of_arr_len = 132096 # ค่าคงที่
    data_len = len( data ) # เก็บ len ของข้อมูลเสียง
    
    if data_len < constant_of_arr_len : # ถ้า len ของข้อมูลน้อยกว่าที่กำหนดให้อัดเสียงพื้้นหลังเพิ่ม
        ReRecData = ReRecord( 2 )
        append_data = np.append( data, ReRecData )
        new_data = check_data( append_data ) # หาจำนวนที่ต้องเพิ่มให้กับ array

    elif data_len > constant_of_arr_len : # ถ้า len ของข้อมูลมากกว่าที่กำหนดให้ลดจากกว่าจะเหลือ 132096
        data_to_delete = data_len - constant_of_arr_len # หาจำนวนที่ต้องลดใน array
        for delete in range( data_to_delete ) :
            data = np.delete( data, -1 ) # ลดค่าใน array จนกว่าจะเท่ากับค่าที่กำหนด
        new_data = data

    elif data_len == constant_of_arr_len : # ถ้าข้อมูลเสียงมีความยาว 3 วินาที
        new_data = data
    return new_data
###################################################################################

# features extraction by mfcc
def audio_to_mfcc( feature ) :
    rate = Rate
    answer = mfcc( feature, rate ) #นำไปเข้าฟังก์ชั่น MFCC 
    return answer

######################################## Normalization set of function ########################################
def reshape_to_1d( feature ) :
    reshape_size = feature.shape[0] * feature.shape[1] # ขนาดที่ต้องการเปลี่ยนจาก 2 มิติเป็น 1 มิติ
    reshape_feature = np.reshape( feature, reshape_size )
    return reshape_feature

def scaling_for_test( data_for_scale, min, max ) :
    min = np.float64( min )
    max = np.float64( max )
    rescale_data = ( data_for_scale - min ) / ( max - min )
    return rescale_data

def normalize_for_test( mfcc_feature, min, max ) :
    re_feature = reshape_to_1d( mfcc_feature ) # reshape
    normalized_data = [] # เก็บข้อมูลที่ scale ข้อมูลแล้ว
    num_of_column = len( re_feature ) # จำนวน feature ในแต่ละข้อมูล
    
    for column in range( num_of_column ) : # normalize โดยทำทีละ column
        data_to_normalize = re_feature[column]
        # scaling
        scale_data = scaling_for_test( data_to_normalize, min[column], max[column] ) # นำค่าที่ได้ไป scaling ให้อยู่ระหว่าง 0 - 1
        # append new scaling data
        normalized_data.append( scale_data )
    return normalized_data
######################################## Normalization set of function ########################################

# Show prediction
def model_predicted( output ) :
    output = np.reshape( output, 3 )
    if np.max( output ) == output[0] :
        print( f'เปิดประตู --> %.2fเปอร์เซ็น' %(output[0]*100) )
        #C2R.send_command("0") # ส่งค่าให้ Raspberry pi ผ่านสายแลน
    elif np.max( output ) == output[1] :
        print( f'ปิดประตู --> %.2fเปอร์เซ็น' %(output[1]*100) )
        #C2R.send_command("1") # ส่งค่าให้ Raspberry pi ผ่านสายแลน
    if np.max( output ) == output[2] :
        print( f'ไม่ทำอะไร --> %.2fเปอร์เซ็น' %(output[2]*100) )
        #C2R.send_command("2") # ส่งค่าให้ Raspberry pi ผ่านสายแลน

# Load the TFLite model in TFLite Interpreter
filename = "RaspberryTest.tflite"
NpzName = "RaspberryTest.npz" 
interpreter = tf.lite.Interpreter( model_path = filename )

# เปิดไฟล์ที่เก็บข้อมูล min และ max มาใช้งาน
npzfile = np.load( NpzName )

# เก็บค่า min และ max ใส่ dictionary
Min = npzfile['Min']
Max = npzfile['Max']

# กำหนดให้แปลงข้อมูลเป็นรูปแบบที่เหมาะสมกับโมเดล
interpreter.allocate_tensors()

# โหลดข้อมูลนำเข้า
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ทดสอบโมเดล
loop = 0
Rate = 44100
while loop == 0 :
    # Record speaker
    data = Record()
    audio_data = check_data( data )
    clean_data = nr.reduce_noise( y = audio_data, sr = Rate )
    mfcc_data = mfcc( clean_data, 44100 )
    normalize_data = normalize_for_test( mfcc_data, Min, Max )
    input_data = np.float32( np.reshape( normalize_data, (1, 3887, 1) ) ) # เปลี่ยนขนาดของข้อมูลโดยเพิ่ม batch_size ไปข้างหน้า

    # กำหนดข้อมูลนำเข้าให้กับโมเดล
    interpreter.set_tensor( input_details[0]['index'], input_data )
    # ประมวลผลโมเดล
    interpreter.invoke()
    # รับผลลัพธ์
    output_data = interpreter.get_tensor( output_details[0]['index'] )
    # แสดงผลลัพธ์ที่ได้
    print( 'result..' )
    model_predicted( output_data )
    print( output_data )
    print( '\n\n' )
        