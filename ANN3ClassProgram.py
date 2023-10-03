'''
โปรแกรมนี้เป็นโปรแกรมสำหรับเทรนด์โมเดลกับข้อมูลเสียงที่ได้ทำการบันทึกจากผู้พูดทั้ง 2 คน
1. ตั้งชื่อไฟล์แล้วกดปุ่ม Start
2. ผู้พูดคนที่ 1 จะพูดเปิดประตู 10 ครั้ง และปิดประตู 10 ครั้ง
3. ผู้พูดคนที่ 2 จะพูดเปิดประตู 10 ครั้ง และปิดประตู 10 ครั้ง
4. หลังจากนั้นระบบจะบันทึกเสียงสภาพแวดล้อมพื้นหลังอีก 5 ครั้ง
5. ระบบจะนำข้อมูลเสียงที่ได้ทำการบันทึกมาเทรนด์กับโมเดล ANN
6. เมื่อเทรนด์เสร็จโมเดลจะถูกบันทึกเป็นไฟล์ .tflite รวมกับไฟล์ .npz ที่เก็บค่า Min-Max ของชุดข้อมูลเทรนด์(ข้อมูล Min-Max จะถูกนำมาใช้เพื่อทำการทดสอบโมเดล)
'''

import time # หน่วงเวลา
import numpy as np # ใช้จัดการข้อมูล เช่น array เป็นต้น
import tkinter as tk # สร้างหน้าต่างโปรแกรมจากโค้ด
import tensorflow as tf # machine learning framework
import noisereduce as nr # ลดเสียงรบกวน
from tensorflow import keras
import speech_recognition as sr # ใช้จัดการไมโครโฟน
from tensorflow.keras import layers
from python_speech_features import mfcc # ใช้สำหรับประมวลผลเสียงให้อยู่บนสเกล mel
from sklearn.model_selection import KFold # ใช้แบ่งข้อมูลแบบ K-fold cross validation

# Record speaker data
def RecordBackGround( RecordRound ) :
    Rate = 44100
    r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
    with sr.Microphone() as source :
        bg_noise.config( text = f'Recording ({RecordRound})' ) 
        bg_noise.update() # อัพเดทสถานะ
        AudioData = r.record( source, duration = 3 ) # เก็บข้อมูลเสียง 3 วิ
        RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
        RecData = np.frombuffer( RawData, dtype = 'int16' ) # แปลง bytes เป็น int
        bg_noise.config( text = ' ' )
        bg_noise.update()
    return RecData

# Sorting data as [0, 0, 1, 1, 0, 0, 1, 1, 2], 5 folds
def sort_data( feature, label ) :
    n_folds = 5
    x_data = np.zeros( feature.shape ) # สร้างข้อมูลสำหรับแทนที่
    y_data = np.zeros( label.shape ) # สร้างข้อมูลสำหรับแทนที่
    j1 = 0 # ใช้แทนลำดับใน x_data, y_data เท่านั้น
    j2 = 0 # ใช้แทนลำดับใน x_data, y_data เท่านั้น
    for i in range( n_folds ) : # 5 folds
        x_data[j1], x_data[j1 + 1] = feature[j2], feature[j2 + 1]        #0
        x_data[j1 + 2], x_data[j1 + 3] = feature[j2 + 10], feature[j2 + 11]  #1
        x_data[j1 + 4], x_data[j1 + 5] = feature[j2 + 20], feature[j2 + 21]  #0
        x_data[j1 + 6], x_data[j1 + 7] = feature[j2 + 30], feature[j2 + 31]  #1
        x_data[j1 + 8] = feature[i + 40]

        y_data[j1], y_data[j1 + 1] = label[j2], label[j2 + 1]        #0
        y_data[j1 + 2], y_data[j1 + 3] = label[j2 + 10], label[j2 + 11]  #1
        y_data[j1 + 4], y_data[j1 + 5] = label[j2 + 20], label[j2 + 21]  #0
        y_data[j1 + 6], y_data[j1 + 7] = label[j2 + 30], label[j2 + 31]  #1
        y_data[j1 + 8] = label[i + 40]
        j1 = j1 + 9 # ใช้เลือกข้อมูลมาใส่แบบล็อคค่า
        j2 = j2 + 2 # ใช้เลือกข้อมูลมาใส่แบบล็อคค่า
    return x_data, y_data

# generate folds
def gen_folds( index, x, y ) : # ข้อมูลใน index คือแกนที่จะใช้แบ่งข้อมูลแต่ละ folds โดยจะเลือกข้อมูลจาก x และ y
    feature = np.zeros( [len(index), x.shape[1], 1] ) # สร้างรูปแบบข้อมูลสำหรับแทนที่พร้อมนำไปเข้าโมเดล ( sample, 3887, 1 )
    label = np.zeros( [len(index), 3] ) # สร้างรูปแบบข้อมูลสำหรับแทนที่พร้อมนำไปเข้าโมเดล ( sample, num_classes )
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


######################################## Normalization set of function ########################################
def reshape_to_1d( feature ) :
    num_of_sample = feature.shape[0] # เช็คว่าข้อมูลทั้งหมดกี่ตัวอย่าง
    reshape_size = feature.shape[1] * feature.shape[2] # ขนาดที่ต้องการเปลี่ยนจาก 2 มิติเป็น 1 มิติ
    reshape_feature = np.zeros( [num_of_sample, reshape_size] ) # สร้างตัวแปรมาเก็บข้อมูลที่เรียงเสร็จ
    # reshape 2D to 1D array
    for re in range( num_of_sample ) :
        reshape_feature[re] = np.reshape( feature[re], reshape_size )
    print( f"New shape is {reshape_feature.shape}" )
    print( reshape_feature )
    return reshape_feature

def scaling( data_for_scale ) :
    min = np.min( data_for_scale )
    max = np.max( data_for_scale )
    rescale_data = ( data_for_scale - min ) / ( max - min )
    return rescale_data, min, max

def normalization( mfcc_feature ) :
    min_max_column = { "min" : [], "max" : [] }
    re_feature = reshape_to_1d( mfcc_feature ) # reshape
    normalized_data = np.zeros( re_feature.shape )
    num_of_sample = re_feature.shape[0] # จำนวนตัวอย่างข้อมูล
    num_of_column = re_feature.shape[1] # จำนวน feature ในแต่ละข้อมูล
    
    for column in range( num_of_column ) : # normalize โดยทำทีละ column
        data_to_normalize = []  # ใช้เก็บค่าที่จะนำไป scaling
        # append data for scaling
        for sample in range( num_of_sample ) : # ทำทุก sample แต่ column เดียวกัน
            data_to_normalize.append( re_feature[sample][column] ) # นำค่า column เดียวกันจากทุก sample ไปเก็บ
        # scaling 
        normalize_data, min_value, max_value = scaling( data_to_normalize ) # นำค่าที่ได้ไป scaling ให้อยู่ระหว่าง 0 - 1
        min_max_column["min"].append( min_value ) # เก็บค่า min เพื่อนำไปใช้ตอนทดสอบโมเดล
        min_max_column["max"].append( max_value ) # เก็บค่า max เพื่อนำไปใช้ตอนทดสอบโมเดล
        # append new scaling data
        for value in range( len(normalize_data) ) :
            normalized_data[value][column] = normalize_data[value]
    return normalized_data, min_max_column 
######################################## Normalization set of function ########################################

# pre-procesing data for model
def processing_data( x_data, y_data ) :
    num_classes = 3
    reduced_noise = ReNoise( x_data ) # renoise from audio data
    mfcc_data = audio_to_mfcc( reduced_noise ) # use mfcc function
    normaliz_data, min_max_value = normalization( mfcc_data ) # normalization data 0-1
    X_data = np.expand_dims( normaliz_data, -1 ) # reshape (sample, 3887, 1)
    Y_data = keras.utils.to_categorical( y_data, num_classes ) # แปลง labels เป็น one-hot encoding
    return X_data, Y_data, min_max_value

# features extraction by mfcc
def audio_to_mfcc( features ) :
    rate = 44100
    samples = features.shape[0] #เช็คจำนวนของข้อมูล
    answer = np.zeros( (samples, 299, 13) ) #สร้างเพื่อนำมาใส่ข้อมูลหลังเข้า mfcc function
    for i in range( 0, samples ) :
        answer[ i : i+1 ] = mfcc( features[i], rate ) #นำไปเข้าฟังก์ชั่น MFCC 
    return answer

# Train the model and plot graph
def train_model( model, X_train, Y_train, X_val, Y_val, epochs, batch_size ) :
    logs = model.fit( X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val) ).history # train
    train_accuracy = logs["accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ train_accuracy
    val_accuracy = logs["val_accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ val_accuracy
    train_loss = logs["loss"][epochs - 1] # เลิอกค่าสุดท้ายของ train_loss
    val_loss = logs["val_loss"][epochs - 1] # เลิอกค่าสุดท้ายของ val_loss
    return model, train_accuracy, val_accuracy, train_loss, val_loss

############################### Test auto microphone ###############################
def RecordSpeaker( NoSpeker, RecordRound ) : # ลำดับผู้พูด, รอบที่พูด
    r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
    r.energy_threshold = 10000 # ไมโครโฟนจะทำงานเมื่อ มีเสียงพูด > 10000 หากค่าเสียง < 10000 จะถือว่าเป็นเสียงเงียบ
    try :
        with sr.Microphone() as source : # ใช้คำสั่ง with เปิดใช้ไมค์โดยตั้งชื่อว่า source
            if NoSpeker == 1 :
                record_label1.config( text = 'Listening' ) 
                record_label1.update() # อัพเดทสถานะ
                AudioData = r.listen( source ,timeout = 30 )
                record_label1.config( text = f'Recording ({RecordRound})' ) 
                record_label1.update() # อัพเดทสถานะ

            elif NoSpeker == 2 :
                record_label2.config( text = 'Listening' ) 
                record_label2.update() # อัพเดทสถานะ
                AudioData = r.listen( source ,timeout = 30 )
                record_label2.config( text = f'Recording ({RecordRound})' ) 
                record_label2.update() # อัพเดทสถานะ

            RawData = AudioData.get_raw_data( convert_rate = 44100 ) # บันทึกเสียงในเรท 44100
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
        RawData = AudioData.get_raw_data( convert_rate = 44100 ) # บันทึกเสียงในเรท 44100
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

def ProgramStart() :
    # Model / data parameters
    x_train = [] # เก็บข้อมูลเสียงที่บันทึก
    y_train = [] # label
    input_shape = ( 3887, 1 ) # ปรับตามขนาดของข้อมูล
    num_classes = 3
    n_folds = 5

    # Recording speaker 40 sample class 0 = 20, class 1 = 20
    for rec in range( 40 ) :
        if rec <= 9 : # บันทึกเปิดประตูคนที่ 1 สิบครั้ง
            if rec == 0 :
                result_No1.config( text = 'Say Open' )
                result_No1.update()
                root.after( 1000 ) # delay 1 sec
            root.after( 1000 ) # delay 1 sec
            # บันทึกเสียงคนที่ 1 เปิด
            data = RecordSpeaker( 1, rec + 1 )  # บันทึกคนที่ 1, พร้อมนับรอบ
            speaker_data = check_data( data ) # เช็คว่าข้อมูลมีความยาว 3 วินาทีหรือไม่
            x_train.append( speaker_data ) # เพิ่มข้อมูลเสียงใน x_train
            y_train.append( 0 )
               
        elif rec <= 19 : # บันทึกปิดประตูคนที่ 1 สิบครั้ง
            if rec == 10 :
                result_No1.config( text = 'Say Close' )
                result_No1.update()
                root.after( 1000 ) # delay 1 sec
            root.after( 1000 ) # delay 1 sec
            # บันทึกเสียงคนที่ 1 ปิด
            data = RecordSpeaker( 1, rec - 9 ) # บันทึกคนที่ 1, พร้อมนับรอบ
            speaker_data = check_data( data ) # เช็คว่าข้อมูลมีความยาว 3 วินาทีหรือไม่
            x_train.append( speaker_data ) # เพิ่มข้อมูลเสียงใน x_train
            y_train.append( 1 )
            if rec == 19 :
                record_label1.config( text = 'Finish...' )
                record_label1.update()  # อัพเดทสถานะ
                root.after( 2000 ) # delay 1 sec

        elif rec <= 29 : # บันทึกเปิดประตูคนที่ 2 สิบครั้ง
            if rec == 20 :
                result_No2.config( text = 'Say Open' )
                result_No2.update() 
                root.after( 1000 ) # delay 1 sec
            root.after( 1000 ) # delay 1 sec
            # บันทึกเสียงคนที่ 2 เปิด
            data = RecordSpeaker( 2, rec - 19 ) # บันทึกคนที่ 2, พร้อมนับรอบ
            speaker_data = check_data( data ) # เช็คว่าข้อมูลมีความยาว 3 วินาทีหรือไม่
            x_train.append( speaker_data ) # เพิ่มข้อมูลเสียงใน x_train
            y_train.append( 0 )

        elif rec <= 39 : # บันทึกปิดประตูคนที่ 2 สิบครั้ง
            if rec == 30 :
                result_No2.config( text = 'Say Close' )
                result_No2.update()
                root.after( 1000 ) # delay 1 sec
            root.after( 1000 ) # delay 1 sec
            # บันทึกเสียงคนที่ 2 ปิด
            data = RecordSpeaker( 2, rec - 29 ) # บันทึกคนที่ 2, พร้อมนับรอบ
            speaker_data = check_data( data ) # เช็คว่าข้อมูลมีความยาว 3 วินาทีหรือไม่
            x_train.append( speaker_data ) # เพิ่มข้อมูลเสียงใน x_train
            y_train.append( 1 )
            if rec == 39 :
                record_label2.config( text = 'Finish...' )
                record_label2.update()  # อัพเดทสถานะ
                root.after( 2000 ) # delay 1 sec
    
    # Recording Background noise
    for rec in range( 5 ) : # บันทึกเสียงพื้นหลัง 5 ครั้ง
        if rec == 0 :
            bg_noise.config( text = 'Background noise...' )
            bg_noise.update()

        root.after( 2000 ) # delay 1 sec
        record_data = RecordBackGround( rec + 1 ) # บันทึกเสียงพร้อมบอกรอบในการบันทึก
        x_train.append( record_data ) # เพิ่มข้อมูลเสียงใน x_train
        y_train.append( 2 )
        if rec == 4 :
            bg_noise.config( text = ' ' )
            bg_noise.update()  # อัพเดทสถานะ
            bg_noise.config( text = 'Finish...' )
            bg_noise.update()  # อัพเดทสถานะ
            root.after( 1000 )

    train_status.config( text = 'Training the model...' ) 
    train_status.update()  # อัพเดทสถานะ

    # Convert list to numpy array
    x_train = np.array( x_train )
    y_train = np.array( y_train )

    # Sort train set
    sort_x, sort_y = sort_data( x_train, y_train ) # เรียงข้อมูล train set ให้เหมาะสมสำหรับการแบ่ง K-folds

    # Pre-processing data
    X_train, Y_train, min_max_value = processing_data( sort_x, sort_y )

    # K-folds cross validation
    TrainSet = { "feature_train" : [], "label_train" : [], "feature_val" : [], "label_val" : [] } # สร้าง dict เก็บชุดข้อมูลเทรนด์
    kf = KFold( n_splits = n_folds ) # ตั้งค่าจำนวน folds ที่จะแบ่งข้อมูล 5 folds
    for i, ( train_index, val_index ) in enumerate( kf.split(X = X_train) ) : # ทำการวนซ้ำแบบแจกแจง แบ่งข้อมูลเป็น fold จาก kf และแจกแจงให้ train_index, val_index
        print( f"\nFold {i}..." )     
        feature_train, label_train =  gen_folds( train_index, X_train, Y_train ) # train fold
        feature_val, label_val =  gen_folds( val_index, X_train, Y_train ) # validation fold
        TrainSet["feature_train"].append( feature_train )
        TrainSet["label_train"].append( label_train )
        TrainSet["feature_val"].append( feature_val )
        TrainSet["label_val"].append( label_val )

    # Build the model
    model_set = { "model" : [] } # สร้าง dict เก็บโมเดลทั้งหมด

    for build in range( n_folds ) :
        model = keras.Sequential(
              [
                keras.Input( shape = input_shape ),
                layers.Flatten(), #เป็นการปรับข้อมูลในอินพุตให้เรียบ (ทำให้เหลือมิติเดียว)
                layers.Dense( units = 250, activation = "selu" ),
                layers.Dense( units = 1000, activation = "selu" ),
                layers.Dense( units = 500, activation = "selu" ),
                layers.Dense( units = num_classes, activation = "softmax" ) 
              ]
        )
        model_set["model"].append( model ) # เก็บโมเดลที่สร้างไว้ใน dict

    # Compile the model
    for do in range( n_folds ) :
        model_set["model"][do].compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )

    # Train the model
    batch_size = 18
    epochs = 25
    train_result = { "model" : [], "acc" : [], "val_acc" : [], "loss" : [], "val_loss" : [] } # สร้าง dictionary ใช้เก็บค่าที่เทรนด์

    for train in range( n_folds ) :
        # เทรนด์ทีละโมเดล
        trained_model, train_acc, val_acc, train_loss, val_loss = train_model( 
            model_set["model"][train], 
            TrainSet["feature_train"][train], 
            TrainSet["label_train"][train], 
            TrainSet["feature_val"][train], 
            TrainSet["label_val"][train], 
            epochs, 
            batch_size 
        )
        # เก็บค่าผลลัพธ์ที่เทรนด์ เพื่อนำไปเปรียบเทียบต่อไป
        train_result["model"].append( trained_model )
        train_result["acc"].append( train_acc )
        train_result["val_acc"].append( val_acc )
        train_result["loss"].append( train_loss )
        train_result["val_loss"].append( val_loss )

    # Get the best score from Val_accuracy
    acc_score = [] # สร้างตัวเก็บข้อมูล
    loss_score = [] # สร้างตัวเก็บข้อมูล
    for get_score in range( n_folds ) :
        acc_score.append( train_result["val_acc"][get_score] )
        loss_score.append( train_result["val_loss"][get_score] )

        if( get_score == (n_folds - 1) ) : # หลังจากเก็บข้อมูลครบ ให้แสดงค่า
            print( f"accuracy score --> {acc_score}" )
            print( f"loss_score ------> {loss_score}" )
            print( f"higher accuracy score -> {np.max(acc_score)}" )
            print( f"lower loss score {np.min(loss_score)}"  )

    # Choose the best model
    index = np.ones( n_folds )# สร้างตัวแปรเก็บค่า loss เพื่อนำไปประมวลผลต่อ (ใช้เปรียบเทียบเมื่อค่าความแม่นยำเท่ากัน)
    for check_score in range( n_folds ) :
        if( train_result["val_acc"][check_score] == np.max(acc_score) ) : # นำค่า loss จากโมเดลที่มี accuracy สูงไปคิดต่อ (ใช้เปรียบเทียบเมื่อค่าความแม่นยำเท่ากัน)
            if( check_score == 0 ) :
                index[0] = loss_score[0]
            elif( check_score == 1 ) :
                index[1] = loss_score[1]
            elif( check_score == 2 ) :
                index[2] = loss_score[2]
            elif( check_score == 3 ) :
                index[3] = loss_score[3]
            elif( check_score == 4 ) :
                index[4] = loss_score[4]

    for search in range( n_folds ) : #หาโมเดลที่ loss น้อยที่สุด
        if( index[search] == np.min(index) ) :
            print( f"\nChoose -> model{search}" )
            print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" 
                  %(train_result["loss"][search], train_result["acc"][search], train_result["val_loss"][search], train_result["val_acc"][search]) )
            Best_model = train_result["model"][search]
            break
    
    
    # Convert the model to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model( Best_model )
    tflite_model = converter.convert()

    # Save the model as a tflite file
    filename = file_name_entry.get() # ตั้งชื่อไฟล์
    tflite_name = filename + '.tflite' 
    with open( tflite_name, "wb" ) as f :
        f.write( tflite_model )
    
    # Save MinMax for normalize test data
    npz_name = filename + '.npz'
    np.savez( npz_name, Min = min_max_value['min'], Max = min_max_value['max'] )
    
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

# สร้างปุ่มกด เมื่อกดปุ่มโปรแกรมจะเริ่มทำงาน
start_button = tk.Button( root, text = 'Start', command = ProgramStart, bg = 'red', fg = 'white', font = ('helvetica', 10, 'bold')  )
canvas.create_window( 280, 130, window = start_button )

# สร้างป้ายกำกับผลลัพธ์
record_label1 = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 200, 215, window = record_label1 )
record_label2 = tk.Label( root, font = ('helvetica', 15, 'bold') )
canvas.create_window( 200, 280, window = record_label2 )
result_No1 = tk.Label( root, fg = 'red', font = ('helvetica', 12, 'bold') )
canvas.create_window( 250, 180, window = result_No1 )
result_No2 = tk.Label( root, fg = 'red', font = ('helvetica', 12, 'bold') )
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
