'''
โปรแกรมนี้เป็นโปรแกรมสำหรับเทรนด์โมเดลกับข้อมูลเสียงที่ได้ทำการบันทึกจากผู้พูดทั้ง 2 คน
1. ตั้งชื่อไฟล์แล้วกดปุ่ม Start
2. ผู้พูดคนที่ 1 จะพูดเปิดประตู 10 ครั้ง และปิดประตู 10 ครั้ง
3. ผู้พูดคนที่ 2 จะพูดเปิดประตู 10 ครั้ง และปิดประตู 10 ครั้ง
4. หลังจากนั้นระบบจะบันทึกเสียงสภาพแวดล้อมพื้นหลังอีก 5 ครั้ง
5. ระบบจะนำข้อมูลเสียงที่ได้ทำการบันทึกมาเทรนด์กับโมเดล ANN
6. เมื่อเทรนด์เสร็จโมเดลจะถูกบันทึกเป็นไฟล์ .tflite รวมกับไฟล์ .npz ที่เก็บค่า Min-Max ของชุดข้อมูลเทรนด์(ข้อมูล Min-Max จะถูกนำมาใช้เพื่อทำการทดสอบโมเดล)
'''

import numpy as np 
import tkinter as tk
import tensorflow as tf
import noisereduce as nr # ลดเสียงรบกวน
from tensorflow import keras
from tkinter import messagebox
import speech_recognition as sr # เรียกใช้ไมโครโฟน
from tkinter import simpledialog 
from tensorflow.keras import layers # ใช้สร้างและเทรนด์โมเดล
from python_speech_features import mfcc # สกัดข้อมูลเสียง
from sklearn.model_selection import StratifiedShuffleSplit # ใช้แบ่งข้อมูล

Rate = 44100
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

# Reduse noise from data
def ReNoise( AudioData ) :
    Rate = 44100
    reduced_noise = np.zeros( AudioData.shape )
    print( "Reduce noise from audio data" )
    for i in range( len(AudioData) ) :
        reduced_noise[i] = nr.reduce_noise( y = AudioData[i], sr = Rate )
    return reduced_noise

# features extraction by mfcc
def audio_to_mfcc( features ) :
    samples = features.shape[0] #เช็คจำนวนของข้อมูล 
    answer = np.zeros( (samples, 299, 13) ) #สร้างเพื่อนำมาใส่ข้อมูลหลังเข้า mfcc function
    for i in range( 0, samples ) :
        answer[ i : i+1 ] = mfcc( features[i], Rate ) #นำไปเข้าฟังก์ชั่น MFCC 
    return answer

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
        min_max_column["min"].append( min_value )
        min_max_column["max"].append( max_value )
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

# generate folds
def gen_folds( index, x, y ) : # ข้อมูลใน index คือแกนที่จะใช้แบ่งข้อมูลแต่ละ folds โดยจะเลือกข้อมูลจาก x และ y
    feature = np.zeros( [len(index), x.shape[1], 1] ) # สร้างรูปแบบข้อมูลสำหรับแทนที่พร้อมนำไปเข้าโมเดล ( sample, 3887, 1 )
    label = np.zeros( [len(index), 3] ) # สร้างรูปแบบข้อมูลสำหรับแทนที่พร้อมนำไปเข้าโมเดล ( sample, num_classes )
    for i in range( 0, len(index) ) :
        feature[i] = x[ index[i] ]
        label[i] = y[ index[i] ]
    return feature, label  

# Train the model and plot graph
def train_model( model, X_train, Y_train, X_val, Y_val, epochs, batch_size ) :
    logs = model.fit( X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val) ).history # train
    train_accuracy = logs["accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ train_accuracy
    val_accuracy = logs["val_accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ val_accuracy
    train_loss = logs["loss"][epochs - 1] # เลิอกค่าสุดท้ายของ train_loss
    val_loss = logs["val_loss"][epochs - 1] # เลิอกค่าสุดท้ายของ val_loss
    return model, train_accuracy, val_accuracy, train_loss, val_loss

# คลาสสำหรับผู้พูด
class speaker_for_train :
    def __init__( self, master, name ) :
        self.master = master # master = frame
        self.name = name # เก็บลำดับผู้พูด
        self.Feature = [] # เก็บข้อมูลเสียงสำหรับเทรนด์
        self.Label = [] # เก็บป้ายเฉลยสำหรับเทรนด์
        self.create_widgets() # สร้างหน้าต่างแสดงผล
        
    def create_widgets( self ) :
        # สร้าง Frame เพื่อจัดวาง Label และ Button ในแต่ละ speaker
        self.frame = tk.Frame( self.master )
        self.frame.pack( fill = tk.X )

        # แสดงชื่อผู้พูดทางซ้าย
        self.label = tk.Label( self.frame, text = self.name )
        self.label.grid( row = 1, column = 0, padx = 10, pady = 10 )  # จัดวาง Label ให้อยู่ตรงกลางทางซ้าย

        # แสดงปุ่ม Record ทางขวา
        self.record_button = tk.Button( self.frame, text = "Record", command = self.record_speaker ) # ปุ่มบันทึกเสียง
        self.record_button.grid( row = 1, column = 1, padx = 10, pady = 10 )  # จัดวาง Button ให้อยู่ตรงกลางทางขวา

    # บันทึกเสียงผู้พูด
    def record_speaker( self ) :
        r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
        r.energy_threshold = 10000 # ไมโครโฟนจะทำงานเมื่อมีเสียงพูด //7000

        recording_label = tk.Label( self.frame, text = "Recording speaker" ) # แสดงข้อความ "Recording.." ใน Frame
        recording_label.grid( row = 2, column = 0, columnspan = 2 )  # จัดวาง Label ด้านล่างทั้งคอลัมน์
        recording_label.update() # อัพเดทข้อความ
        root.after( 2000 ) # ให้แสดงข้อความ 2 วินาที

        for rec in range(20) : # พูดเปิดประตู 10 ครั้ง ปิดประตู 10 ครั้ง
            if rec <= 9 : # 9
                word_to_say = f"Say open ({rec + 1})" # +1
            else :
                word_to_say = f"Say close ({rec - 9})" # -9
            # บันทึกเสียงผู้พูด
            try : # พูดทัน
                with sr.Microphone() as source : # ใช้คำสั่ง with เปิดใช้ไมค์โดยตั้งชื่อว่า source
                    print( "Listening...\n" )
                    recording_label.config( text = word_to_say ) # เปลี่ยนข้อความ
                    recording_label.update() # อัพเดทข้อความ
                    AudioData = r.listen( source ,timeout = 15 ) 
                    print( "Recording...\n" )
                    recording_label.config( text = "Recording..." )
                    recording_label.update() # อัพเดทข้อความ
                    RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
                    speaker_data = np.frombuffer( RawData, dtype = "int16" ) # แปลง bytes เป็น int
                    RecData = check_data( speaker_data )
            except : # พูดไม่ทัน
                with sr.Microphone() as source :
                    recording_label.config( text = "Listening time out" ) # เปลี่ยนข้อความ
                    recording_label.update() # อัพเดทข้อความ
                    root.after( 1500 ) # ให้แสดงข้อความ 1.5 วินาที
                    recording_label.config( text = "Try again" ) # เปลี่ยนข้อความ
                    recording_label.update() # อัพเดทข้อความ
                    root.after( 1500 ) # ให้แสดงข้อความ 1.5 วินาที

                    recording_label.config( text = "Listening" ) # เปลี่ยนข้อความ
                    recording_label.update() # อัพเดทข้อความ
                    AudioData = r.listen( source ,timeout = 15 )
                    print( "Recording...\n" )
                    recording_label.config( text = "Recording..." )
                    recording_label.update() # อัพเดทข้อความ
                    RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
                    speaker_data = np.frombuffer( RawData, dtype = "int16" ) # แปลง bytes เป็น int
                    RecData = check_data( speaker_data )

            self.Feature.append( RecData )
            if rec <= 9 : # 9
                self.Label.append(0)
            else :
                self.Label.append(1)
        recording_label.config( text = "Finished" )
        recording_label.update() # อัพเดทข้อความ

# ตัวแอพพลิเคชั่น
class TrainAndRecApp:
    def __init__(self, master):
        self.master = master # master คือ root
        self.speaker_list = [] # เก็บข้อมูลผู้พูดแต่ละคน มีทั้ง ชื่อ, ปุ่ม, feature, label
        self.bg_noise = [] # เก็บเสียงพื้นหลัง
        self.create_widgets() # สร้างวิทเจ็ท (หน้าแสดงผล)
        
    # สร้างปุ่มและหน้าต่าง
    def create_widgets( self ) :
        # แสดงปุ่มเพิ่มผู้พูด
        self.add_button = tk.Button( self.master, text = "Add speaker", command = self.add_speaker ) # ปุ่มเพิ่มผู้พูด
        self.add_button.pack( pady = 5 )
        
        # สำหรับบันทึกพื้นหลัง
        self.rec_bg_button = tk.Button( self.master, text = "Record background noise", command = self.rec_bg_noise ) # ปุ่มบันทึกเสียงพื้นหลัง
        self.rec_bg_button.pack( pady = 5 )

        # แสดงปุ่มเทรนด์โมเดล
        self.train_button = tk.Button( self.master, text = "Train", command = self.train_model ) # ปุ่มเทรนด์
        self.train_button.pack( pady = 5 )

        # แสดง frame ของผู้พูดแต่ละคนจาก Class speaker_for_train
        self.speaker_list_frame = tk.Frame( self.master )
        self.speaker_list_frame.pack( pady = 10 )

    def rec_bg_noise( self ) :
        recording_bg_label = tk.Label( self.master, text = "Recording background noise" ) # สร้างป้ายข้อความ
        recording_bg_label.pack()
        recording_bg_label.update() # อัพเดทข้อความ
        root.after( 1500 ) # ให้แสดงข้อความ 0.5 วินาที
        r = sr.Recognizer() #เปิดใช้งานการ record เก็บไว้ที่ตัวแปร r
        with sr.Microphone() as source : # เปิดใช้ไมโครโฟน
            for rec in range( 5 ) :
                recording_bg_label.config( text = f"Recording ({rec + 1})" ) # เปลี่ยนข้อความ
                recording_bg_label.update() # อัพเดทข้อความ

                AudioData = r.record( source, duration = 3 ) # เก็บข้อมูลเสียง 3 วิ 
                RawData = AudioData.get_raw_data( convert_rate = Rate ) # บันทึกเสียงในเรท 44100
                RecData = np.frombuffer( RawData, dtype = 'int16' ) # แปลง bytes เป็น int

                recording_bg_label.config( text = " " ) # เปลี่ยนข้อความ
                recording_bg_label.update() # อัพเดทข้อความ
                root.after( 500 ) # ให้แสดงข้อความ 0.5 วินาที
                self.bg_noise.append( RecData )
        recording_bg_label.config( text = "Finished" ) # เปลี่ยนข้อความ
        recording_bg_label.update() # อัพเดทข้อความ

    # เพิ่มผู้พูด
    def add_speaker( self ) :
        speaker_name = f"Speaker {len(self.speaker_list) + 1}" # แสดงลำดับผู้พูด เมื่อทำการกดเพิ่มผู้พูด
        speaker = speaker_for_train( self.speaker_list_frame, speaker_name ) # สร้างอ็อบเจ็ค speaker ที่ไม่ซ้ำกันส่ง frame และ ชื่อผู้พูดไปที่ Class
        self.speaker_list.append( speaker ) # เก็บอ็อปเจ็คไว้ใน speaker_list

    # เทรนด์โมเดลจากข้อมูลผู้พูด
    def train_model( self ) :
        input_shape = ( 3887, 1 ) # ปรับตามขนาดของข้อมูล
        num_classes = 3
        n_folds = 5
        x_data = []
        y_data = []
        result = messagebox.askokcancel( "Training Model", "Training in progress. Do you want to proceed?" ) # หน้าต่างถามตอบ
        if result :
            print( "\nTrain..." )  # แสดงสถานะว่ากำลังเทรนด์
            for read in self.speaker_list : # เข้าไปดึงข้อมูล feature, label ผู้พูดแต่ละคนเพื่อนำมาเทรนด์
                x_data.append( read.Feature )
                y_data.append( read.Label )
            # เปลี่ยนให้เป็น numpy array
            feature_data = np.array( x_data )
            label_data = np.array( y_data )
            # กำหนดรูปทรงข้อมูลที่จะเปลี่ยน
            new_shape_x = ( feature_data.shape[0] * feature_data.shape[1], 132096 ) # (speaker * sample per speaker, feature)
            new_shape_y = label_data.shape[0] * label_data.shape[1] # (speaker * sample per speaker)
            # เปลี่ยนรูปทรงของข้อมูล
            reshape_x = np.reshape( feature_data, new_shape_x )
            reshape_y = np.reshape( label_data, new_shape_y )
            # รวมข้อมูลเสียงผู้พูดกับเสียงพื้นหลัง
            x_train = np.append( reshape_x, self.bg_noise, axis = 0 )
            y_train = np.append( reshape_y, [2 for k in range(5)] )

            # นำข้อมูลไปประมวลผล
            X_train, Y_train, min_max_value = processing_data( x_train, y_train )
            # แบ่งข้อมูลด้วยวิธี StratifiedShuffleSplit
            TrainSet = { "feature_train" : [], "label_train" : [], "feature_val" : [], "label_val" : [] } # สร้าง dict เก็บชุดข้อมูลเทรนด์
            sss = StratifiedShuffleSplit( n_splits = n_folds, test_size = 0.2, random_state = 0 )
            for i, ( train_index, val_index ) in enumerate( sss.split(X_train, Y_train) ):
                print(f"Fold {i}:")
                print(f"  Train: index={train_index}")
                print(f"  Test:  index={val_index}")
                feature_train, label_train = gen_folds( train_index, X_train, Y_train )
                feature_val, label_val = gen_folds( val_index, X_train, Y_train )
                TrainSet["feature_train"].append( feature_train )
                TrainSet["label_train"].append( label_train )
                TrainSet["feature_val"].append( feature_val )
                TrainSet["label_val"].append( label_val )

            # เทรนด์โมเดล
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
            model_set["model"][0].summary() #สรุปโครงสร้างของโมเดล แสดงเป็นตาราง

            # Compile the model
            for do in range( n_folds ) :
                model_set["model"][do].compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )

            # Train the model
            batch_size = int( x_train.shape[0] / 5 ) # จำนวน sample หารด้วย 5 (ทำให้มีการวน 5 รอบ ต่อ 1 epoch)
            epochs = 15
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
            print(index)

            for search in range( n_folds ) : #หาโมเดลที่ loss น้อยที่สุด
                if( index[search] == np.min(index) ) :
                    print( f"\nChoose -> model{search}" )
                    print( "trian_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f" 
                        %(train_result["loss"][search], train_result["acc"][search], train_result["val_loss"][search], train_result["val_acc"][search]) )
                    Best_model = train_result["model"][search]
                    break
        
            messagebox.showinfo( "Training progress", "Completed Training" ) # หน้าต่างแจ้งเตือนหลังจากเทรนด์เสร็จ
            root.after( 1000 ) # ให้แสดงข้อความ 1 วินาที

            # บันทึกโมเดล
            file_name = tk.simpledialog.askstring( "Save model", "Name the model file" )

            # Convert the model to tflite
            converter = tf.lite.TFLiteConverter.from_keras_model( Best_model )
            tflite_model = converter.convert()

            # Save the model as a tflite file
            tflite_name = file_name + '.tflite' 
            with open( tflite_name, "wb" ) as f :
                f.write( tflite_model )
            
            # Save MinMax for normalize test data
            npz_name = file_name + '.npz'
            np.savez( npz_name, Min = min_max_value['min'], Max = min_max_value['max'] )

            messagebox.showinfo( "Training progress", "The model is already saved" ) # หน้าต่างแจ้งเตือนหลังจากบันทึกไฟล์

        else:
            print( "Training cancelled" )  # แสดงสถานะว่าการเทรนด์ถูกยกเลิก

        # นำข้อความออกหลังจาก 3 วินาที
        #self.master.after( 3000, lambda: info_label.grid_forget() )

if __name__ == "__main__":
    root = tk.Tk()
    root.title( "Train The Speaker Recognition" ) # 
    root.geometry( "250x300" ) # กำหนดขนาดหน้าต่าง
    app = TrainAndRecApp( root )

    root.mainloop()
