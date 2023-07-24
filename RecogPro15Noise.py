import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from python_speech_features import mfcc
from sklearn.model_selection import KFold
import speech_recognition as sr
#import noisereduce as nr
import tkinter as tk
import time
from scipy.io.wavfile import write

# Record speaker data
def Record( sequence ) :
    Rate = 44100
    r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
    with sr.Microphone() as source :
        if sequence == 1 : # แสดงการ Record ช่อง 1
            record_label1.config( text = 'Recording' ) 
            record_label1.update() # อัพเดทสถานะ
        elif sequence == 2 : # แสดงการ Record ช่อง 2
            record_label2.config( text = 'Recording' ) 
            record_label2.update() # อัพเดทสถานะ
        elif sequence == 3 : # แสดงการ Record ช่อง 3
            bg_noise.config( text = 'Recording' ) 
            bg_noise.update() # อัพเดทสถานะ

        AudioData = r.record( source, duration = 3 ) # เก็บข้อมูลเสียง 3 วิ
        RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
        RecData = np.frombuffer( RawData, dtype = 'int16' ) # แปลง bytes เป็น int
        record_label1.config( text = ' ' )
        record_label1.update()
        record_label2.config( text = ' ' )
        record_label2.update()
        bg_noise.config( text = ' ' )
        bg_noise.update()
    return RecData

# Sorting data as [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2], 5 folds
def sort_data( feature, label ) :
    n_folds = 5
    x_data = np.zeros( feature.shape ) # สร้างข้อมูลสำหรับแทนที่
    y_data = np.zeros( label.shape ) # สร้างข้อมูลสำหรับแทนที่
    j1 = 0 # ใช้แทนลำดับใน x_data, y_data เท่านั้น
    j2 = 0 # ใช้แทนลำดับใน x_data, y_data เท่านั้น
    for i in range( n_folds ) : # 5 folds
        x_data[j1], x_data[j1 + 1], x_data[j1 + 2] = feature[j2], feature[j2 + 1], feature[j2 + 2] # 0 คนที่ 1
        x_data[j1 + 3], x_data[j1 + 4], x_data[j1 + 5] = feature[j2 + 15], feature[j2 + 16], feature[j2 + 17]  #1 คนที่ 1
        x_data[j1 + 6], x_data[j1 + 7], x_data[j1 + 8] = feature[j2 + 30], feature[j2 + 31], feature[j2 + 32]  # 0 คนที่ 2
        x_data[j1 + 9], x_data[j1 + 10], x_data[j1 + 11] = feature[j2 + 45], feature[j2 + 46], feature[j2 + 47]  #1 คนที่ 2
        x_data[j1 + 12] = feature[i + 60]
        y_data[j1], y_data[j1 + 1], y_data[j1 + 2] = label[j2], label[j2 + 1], label[j2 + 2] # 0 คนที่ 1
        y_data[j1 + 3], y_data[j1 + 4], y_data[j1 + 5] = label[j2 + 15], label[j2 + 16], label[j2 + 17]  #1 คนที่ 1
        y_data[j1 + 6], y_data[j1 + 7], y_data[j1 + 8] = label[j2 + 30], label[j2 + 31], label[j2 + 32]  # 0 คนที่ 2
        y_data[j1 + 9], y_data[j1 + 10], y_data[j1 + 11] = label[j2 + 45], label[j2 + 46], label[j2 + 47]  #1 คนที่ 2
        y_data[j1 + 12] = label[i + 60]
        j1 = j1 + 13 # validate 13 batch
        j2 = j2 + 3 # เลือกทีละ 3 ก้อน (เช่น 0,0,0 หรือ 1,1,1)
    return x_data, y_data

# generate folds
def gen_folds( index, x, y ) : # ข้อมูลใน index คือแกนที่จะใช้แบ่งข้อมูลแต่ละ folds ให้แตกต่างกันออกไป
    feature = np.zeros( [len(index), x.shape[1]] ) # สร้างข้อมูลสำหรับแทนที่
    label = np.zeros( [len(index)] ) # สร้างข้อมูลสำหรับแทนที่
    for i in range( 0, len(index) ) :
        feature[i] = x[ index[i] ]
        label[i] = y[ index[i] ]
    return feature, label 

# Reduse noise from data
def ReNoise( AudioData ) :
    Rate = 44100
    reduced_noise = np.zeros( AudioData.shape )
    for i in range( len(AudioData) ) :
        reduced_noise[i] = nr.reduce_noise( y = AudioData[i], sr = Rate )
    return reduced_noise

# pre-procesing data for model
def process_data( x_data, y_data ) :
    num_classes = 3
    #reduced_noise = ReNoise( x_data )
    mfcc_data = audio_to_mfcc( x_data ) # use mfcc function
    normaliz_data = min_max_normaliz( mfcc_data ) # normalization data 0-1
    X_data = np.expand_dims( normaliz_data, -1 ) # reshape (299, 13, 1)
    Y_data = keras.utils.to_categorical( y_data, num_classes ) # แปลง labels เป็น one-hot encoding
    return X_data, Y_data

# features extraction by mfcc
def audio_to_mfcc( features ) :
    rate = 44100
    samples = features.shape[0] #เช็คจำนวนของข้อมูล
    answer = np.zeros( (samples, 299, 13) ) #สร้างเพื่อนำมาใส่ข้อมูลหลังเข้า mfcc function
    for i in range( 0, samples ) :
        answer[ i : i+1 ] = mfcc( features[i], rate ) #นำไปเข้าฟังก์ชั่น MFCC 
    return answer

# MinMax normalization function
def min_max_normaliz( input_data ) :
    output_data = np.zeros( input_data.shape ) #create new_data 
    for i in range( 0, input_data.shape[0] ) : #loop ตามจำนวน sample
        output_data[i] = ( input_data[i:i+1] - np.min(input_data[i:i+1]) ) / ( np.max(input_data[i:i+1]) - np.min(input_data[i:i+1]) )
    return output_data 

# Train the model and plot graph
def train_model( model, X_train, Y_train, X_val, Y_val, epochs, batch_size ) :
    logs = model.fit( X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val) ).history # train
    train_accuracy = logs["accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ train_accuracy
    val_accuracy = logs["val_accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ val_accuracy
    train_loss = logs["loss"][epochs - 1] # เลิอกค่าสุดท้ายของ train_loss
    val_loss = logs["val_loss"][epochs - 1] # เลิอกค่าสุดท้ายของ val_loss
    return model, train_accuracy, val_accuracy, train_loss, val_loss

# Return the model predicted answer
def TurnPredicted( predicted ) :
    model_predicted = 0
    if predicted[1] < predicted[0] > predicted[2] :
        model_predicted = 0
    elif predicted[0] < predicted[1] > predicted[2] :
        model_predicted = 1
    elif predicted[0] < predicted[2] > predicted[1] :  
        model_predicted = 2
    return model_predicted

def ProgramStart() :
    # Model / data parameters
    x_train = np.zeros( [65, 132096] ) # 65 sample train
    y_train = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2] ) # 65 sample train
    input_shape = ( 299, 13, 1 ) # ปรับตามขนาดของข้อมูล
    num_classes = 3
    n_folds = 5

    # Recording speaker
    for rec in range( len(x_train) ) :
        if rec <= 59 : # Class 0, 1
            if rec <= 14 : # บันทึกเปิดประตูคนที่ 1 สิบครั้ง
                if rec == 0 :
                    result_No1.config( text = 'Say Open' )
                    result_No1.update()
                    root.after( 1000 ) # delay 1 sec
                root.after( 1000 ) # delay 1 sec
                record_status1.config(text = f'({rec + 1})' )
                x_train[rec] = Record(1) # บันทึกเสียงคนที่ 1
                
            elif rec <= 29 : # บันทึกปิดประตูคนที่ 1 สิบครั้ง
                if rec == 15 :
                    result_No1.config( text = 'Say Close' )
                    result_No1.update()
                    root.after( 1000 ) # delay 1 sec
                root.after( 1000 ) # delay 1 sec
                record_status1.config(text = f'({rec - 14})' )
                x_train[rec] = Record(1) # บันทึกเสียงคนที่ 1
                if rec == 29 :
                    record_label1.config( text = 'Finish...' )
                    record_status1.config(text = ' ' )
                    record_label1.update()  # อัพเดทสถานะ
                    record_status1.update()  # อัพเดทสถานะ
                    root.after( 2000 ) # delay 1 sec

            elif rec <= 44 : # บันทึกเปิดประตูคนที่ 2 สิบครั้ง
                if rec == 30 :
                    result_No2.config( text = 'Say Open' )
                    result_No2.update() 
                    root.after( 1000 ) # delay 1 sec
                root.after( 1000 ) # delay 1 sec
                record_status2.config(text = f'({rec - 29})' )
                x_train[rec] = Record(2) # บันทึกเสียงคนที่ 2

            elif rec <= 59 : # บันทึกปิดประตูคนที่ 2 สิบครั้ง
                if rec == 45 :
                    result_No2.config( text = 'Say Close' )
                    result_No2.update()
                    root.after( 1000 ) # delay 1 sec
                root.after( 1000 ) # delay 1 sec
                record_status2.config(text = f'({rec - 44})' )
                x_train[rec] = Record(2) # บันทึกเสียงคนที่ 2
                if rec == 59 :
                    record_label2.config( text = 'Finish...' )
                    record_status2.config(text = ' ' )
                    record_label2.update()  # อัพเดทสถานะ
                    record_status2.update()  # อัพเดทสถานะ
                    root.after( 2000 ) # delay 1 sec

        else : # บันทึก class2 สิบครั้ง
            if rec == 60 :
                bg_noise.config( text = 'Background noise...' )
                bg_noise.update()
            root.after( 2000 ) # delay 1 sec
            x_train[rec] = Record(3)
            if rec == 64 :
                bg_noise.config( text = 'Finish...' )
                bg_noise.config( text = ' ' )
                bg_noise.update()  # อัพเดทสถานะ
                root.after( 1000 )

    # Save dataset
    for i in range( len(x_train) ) :
        name = f'{i}.wav'
        write( name, 44100, x_train[i].astype(np.int16) )

    train_status.config( text = 'Training the model...' ) 
    train_status.update()  # อัพเดทสถานะ

    # sorting train set
    sort_x, sort_y = sort_data( x_train, y_train ) # train set ไปสลับข้อมูล

    # K-folds cross validation
    kf = KFold( n_splits = n_folds ) # ตั้งค่าจำนวน folds ที่จะแบ่งข้อมูล, และ shuffle คือการสับข้อมูลก่อนจะแบ่ง
    for i, ( train_index, val_index ) in enumerate( kf.split(X = sort_x) ) : # ทำการวนซ้ำแบบแจกแจง แบ่งข้อมูลเป็น fold จาก kf และแจกแจงให้ train_index, test_index
        print(f"\nFold {i}...")     
        globals()[ f"feature_train{i}" ], globals()[ f"label_train{i}" ] =  gen_folds( train_index, sort_x, sort_y ) # train fold
        globals()[ f"feature_val{i}" ], globals()[ f"label_val{i}" ] =  gen_folds( val_index, sort_x, sort_y ) # validation fold

    # Prepare data before train the model
    X_train0, Y_train0 = process_data( feature_train0, label_train0 )
    X_train1, Y_train1 = process_data( feature_train1, label_train1 )
    X_train2, Y_train2 = process_data( feature_train2, label_train2 )
    X_train3, Y_train3 = process_data( feature_train3, label_train3 )
    X_train4, Y_train4 = process_data( feature_train4, label_train4 )

    X_val0, Y_val0 = process_data( feature_val0, label_val0 )
    X_val1, Y_val1 = process_data( feature_val1, label_val1 )
    X_val2, Y_val2 = process_data( feature_val2, label_val2 )
    X_val3, Y_val3 = process_data( feature_val3, label_val3 )
    X_val4, Y_val4 = process_data( feature_val4, label_val4 )

    # Build the model
    for i in range( n_folds ) :
        globals()[ f"model{i}" ] = keras.Sequential(
            [
                keras.Input( shape = input_shape ),
                layers.Flatten(), #เป็นการปรับข้อมูลในอินพุตให้เรียบ (ทำให้เหลือมิติเดียว)
                layers.Dense( units = 250, activation = "selu" ),
                layers.Dense( units = 1000, activation = "selu" ),
                layers.Dense( units = 500, activation = "selu" ),
                layers.Dense( units = num_classes, activation = "softmax" )  
            ]
            )

    # Compile the model
    model0.compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )
    model1.compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )
    model2.compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )
    model3.compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )
    model4.compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )

    # Train the model
    batch_size = 24
    epochs = 75 #55
    result = { "acc" : np.zeros(n_folds), "val_acc" : np.zeros(n_folds),
                "loss" : np.zeros(n_folds), "val_loss" : np.zeros(n_folds) } # สร้าง dictionary ใช้เก็บค่าที่เทรนด์
    Model0, result["acc"][0], result["val_acc"][0], result["loss"][0], result["val_loss"][0] = train_model( model0, X_train0, Y_train0, X_val0, Y_val0, epochs, batch_size )
    Model1, result["acc"][1], result["val_acc"][1], result["loss"][1], result["val_loss"][1] = train_model( model1, X_train1, Y_train1, X_val1, Y_val1, epochs, batch_size )
    Model2, result["acc"][2], result["val_acc"][2], result["loss"][2], result["val_loss"][2] = train_model( model2, X_train2, Y_train2, X_val2, Y_val2, epochs, batch_size )
    Model3, result["acc"][3], result["val_acc"][3], result["loss"][3], result["val_loss"][3] = train_model( model3, X_train3, Y_train3, X_val3, Y_val3, epochs, batch_size )
    Model4, result["acc"][4], result["val_acc"][4], result["loss"][4], result["val_loss"][4] = train_model( model4, X_train4, Y_train4, X_val4, Y_val4, epochs, batch_size )

    # Get the best score from Val_accuracy
    acc_score = np.zeros( n_folds ) # สร้างตัวเก็บข้อมูล
    loss_score = np.zeros( n_folds ) # สร้างตัวเก็บข้อมูล
    for i in range( n_folds ) :
        acc_score[i] = result[ "acc" ][i] + result[ "val_acc" ][i] # นำค่าความแม่นยำมารวมกัน
        loss_score[i] = result[ "loss" ][i] + result[ "val_loss" ][i] # นำค่าความสูญเสียมารวมกัน
        if( i == (n_folds - 1) ) : # หลังจากเก็บข้อมูลครบ ให้แสดงค่า
            print( f"accuracy score --> {acc_score}" )
            print( f"loss_score ------> {loss_score}" )
            print( f"higher accuracy score -> {np.max(acc_score)}" )
            print( f"lower loss score {np.min(loss_score)}"  )

    # Choose the best model
    index = np.zeros( n_folds ) + 99 # สร้างตัวแปรเก็บค่า loss เพื่อนำไปประมวลผลต่อ (ใช้เปรียบเทียบเมื่อค่าความแม่นยำเท่ากัน)
    for fold in range( n_folds ) :
        if( acc_score[fold] == np.max(acc_score) ) : # นำค่า loss จากโมเดลที่มี accuracy สูงไปคิดต่อ (ใช้เปรียบเทียบเมื่อค่าความแม่นยำเท่ากัน)
            if( fold == 0 ) :
                index[0] = loss_score[0]
            elif( fold == 1 ) :
                index[1] = loss_score[1]
            elif( fold == 2 ) :
                index[2] = loss_score[2]
            elif( fold == 3 ) :
                index[3] = loss_score[3]
            elif( fold == 4 ) :
                index[4] = loss_score[4]
    print(index)
    for i in range( n_folds ) : #หาโมเดลที่ loss น้อยที่สุด
        if( index[i] == np.min(index) ) :
            if( i == 0 ) :
                print( "\nChoose -> model0" )
                print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" %(result["loss"][0], result["acc"][0], result["val_loss"][0], result["val_acc"][0]) )
                Best_model = Model0 
                break
            elif( i == 1 ) :
                print( "\nChoose -> model1" )
                print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" %(result["loss"][1], result["acc"][1], result["val_loss"][1], result["val_acc"][1]) )
                Best_model = Model1 
                break
            elif( i == 2 ) :
                print( "\nChoose -> model2" )
                print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" %(result["loss"][2], result["acc"][2], result["val_loss"][2], result["val_acc"][2]) )
                Best_model = Model2 
                break
            elif( i == 3 ) :
                print( "\nChoose -> model3" )
                print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" %(result["loss"][3], result["acc"][3], result["val_loss"][3], result["val_acc"][3]) )
                Best_model = Model3 
                break
            elif( i == 4 ) :
                print( "\nChoose -> model4" )
                print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" %(result["loss"][4], result["acc"][4], result["val_loss"][4], result["val_acc"][4]) )
                Best_model = Model4
                break
    
    # Convert the model to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model( Best_model )
    tflite_model = converter.convert()

    # Save the model as a tflite file
    filename = file_name_entry.get()
    with open( filename, "wb" ) as f :
        f.write( tflite_model )
    
    train_status.config( text = 'The model is already finish...' ) 
    train_status.update()  # อัพเดทสถานะ

# MAIN #
# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title( 'Speaker Recognition model' )
canvas = tk.Canvas( root, width = 400, height = 500 ) # สร้างพื้นหลัง
canvas.pack() # หลังจากตั้งค่าแล้วใช้ pack เพื่อให้แสดงค่าที่ตั้งไว้ก่อนที่จะรัน

# สร้างป้ายข้อความ
label1 = tk.Label( text = 'Train the Recognition model', fg = 'black', font = ('helvetica', 15, 'bold') )
canvas.create_window( 200, 50, window = label1 ) # จัดตำแหน่งปุ่มกด
label2 = tk.Label( text = 'Name the file then click start', fg = 'black', font = ('helvetica', 10, 'bold') )
canvas.create_window( 200, 100, window = label2 ) # จัดตำแหน่งปุ่มกด
label3 = tk.Label( text = 'Speaker No. 1', fg = 'black', font = ('helvetica', 10, 'bold') )
canvas.create_window( 160, 180, window = label3 ) # จัดตำแหน่งปุ่มกด
label4 = tk.Label( text = 'Speaker No. 2', fg = 'black', font = ('helvetica', 10, 'bold') )
canvas.create_window( 160, 250, window = label4 ) # จัดตำแหน่งปุ่มกด

# สร้างปุ่มกด
start_button = tk.Button( root, text = 'Start', command = ProgramStart, bg = 'red', fg = 'white', font = ('helvetica', 10, 'bold')  )
canvas.create_window( 280, 130, window = start_button )

# สร้างป้ายกำกับผลลัพธ์
record_label1 = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 185, 215, window = record_label1 )
record_label2 = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 185, 280, window = record_label2 )
record_status1 = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 255, 215, window = record_status1 )
record_status2 = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 255, 280, window = record_status2 )
result_No1 = tk.Label( root, fg = 'red', font = ('helvetica', 10, 'bold') )
canvas.create_window( 250, 180, window = result_No1 )
result_No2 = tk.Label( root, fg = 'red', font = ('helvetica', 10, 'bold') )
canvas.create_window( 250, 250, window = result_No2 )
bg_noise = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 200, 340, window = bg_noise )
train_status = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 200, 400, window = train_status )

# สร้างกล่องข้อความสำหรับตั้งชื่อไฟล์
file_name_entry = tk.Entry(root)
canvas.create_window( 185, 130, window = file_name_entry )

# เริ่มต้นการรันโปรแกรม
root.mainloop()