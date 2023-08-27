import os
import numpy as np
import tensorflow as tf
import noisereduce as nr
from scipy.io import wavfile
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from python_speech_features import mfcc
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def OpenTrainData() : # เปิดไฟล์สำหรับ Train data
    train_set = { "feature" : [], "label" : [] }
    for i in range(1, 46) : # 45 sample data
        if i <= 20 : # เปิดไฟล์ class 0
            file_0 = "0_audio_" + str(i) + ".wav"
            data_0_dir = os.path.join( 'DataSetLab\TrainSet\class_0', file_0 )
            samplerate, data_0 = wavfile.read( data_0_dir )
            train_set["feature"].append( data_0 )
            train_set["label"].append(0)
        elif i <= 40 : # เปิดไฟล์ class 1
            file_1 = "1_audio_" + str(i - 20) + ".wav" # ทำให้ค่า i ย้อนไปเป็นเลข 1 เพื่อเปิดไฟล์
            data_1_dir = os.path.join( 'DataSetLab\TrainSet\class_1', file_1 )
            samplerate, data_1 = wavfile.read( data_1_dir )
            train_set["feature"].append( data_1 )
            train_set["label"].append(1)
        else : # เปิดไฟล์ class 2
            file_2 = "2_audio_" + str(i - 40) + ".wav" # ทำให้ค่า i ย้อนไปเป็นเลข 1 เพื่อเปิดไฟล์
            data_2_dir = os.path.join( 'DataSetLab\TrainSet\class_2', file_2 )
            samplerate, data_2 = wavfile.read( data_2_dir )
            train_set["feature"].append( data_2 )
            train_set["label"].append(2)
    return train_set["feature"], train_set["label"]

def OpenTestData() : # เปิดไฟล์สำหรับ Test data
    test_set = { "feature" : [], "label" : [] }
    for i in range(1, 26) : # 25 sample data
        if i <= 10 : # เปิดไฟล์ class 0
            file_0 = "0_audio_" + str(i) + ".wav"
            data_0_dir = os.path.join( 'DataSetLab\TestSet\class_0', file_0 )
            samplerate, data_0 = wavfile.read( data_0_dir )
            test_set["feature"].append( data_0 )
            test_set["label"].append(0)
        elif i <= 20 : # เปิดไฟล์ class 1
            file_1 = "1_audio_" + str(i - 10) + ".wav" # ทำให้ค่า i ย้อนไปเป็นเลข 1 เพื่อเปิดไฟล์
            data_1_dir = os.path.join( 'DataSetLab\TestSet\class_1', file_1 )
            samplerate, data_1 = wavfile.read( data_1_dir )
            test_set["feature"].append( data_1 )
            test_set["label"].append(1)
        else : # เปิดไฟล์ class 2
            file_2 = "2_audio_" + str(i - 20) + ".wav" # ทำให้ค่า i ย้อนไปเป็นเลข 1 เพื่อเปิดไฟล์
            data_2_dir = os.path.join( 'DataSetLab\TestSet\class_2', file_2 )
            samplerate, data_2 = wavfile.read( data_2_dir )
            test_set["feature"].append( data_2 )
            test_set["label"].append(2)
    return test_set["feature"], test_set["label"]

# Sorting data as [0, 0, 0, 0, 1, 1, 1, 1, 2], 5 folds
def sort_data( feature, label ) :
    n_folds = 5
    x_data = np.zeros( feature.shape ) # สร้างข้อมูลสำหรับแทนที่
    y_data = np.zeros( label.shape ) # สร้างข้อมูลสำหรับแทนที่
    j1 = 0 # ใช้แทนลำดับใน x_data, y_data เท่านั้น
    j2 = 0 # ใช้แทนลำดับใน x_data, y_data เท่านั้น
    for i in range( n_folds ) : # 5 folds
        x_data[j1], x_data[j1 + 1] = feature[j2], feature[j2 + 1]        #0
        x_data[j1 + 2], x_data[j1 + 3] = feature[j2 + 10], feature[j2 + 11]  #0
        x_data[j1 + 4], x_data[j1 + 5] = feature[j2 + 20], feature[j2 + 21]  #1
        x_data[j1 + 6], x_data[j1 + 7] = feature[j2 + 30], feature[j2 + 31]  #1
        x_data[j1 + 8] = feature[i + 40]

        y_data[j1], y_data[j1 + 1] = label[j2], label[j2 + 1]        #0
        y_data[j1 + 2], y_data[j1 + 3] = label[j2 + 10], label[j2 + 11]  #0
        y_data[j1 + 4], y_data[j1 + 5] = label[j2 + 20], label[j2 + 21]  #1
        y_data[j1 + 6], y_data[j1 + 7] = label[j2 + 30], label[j2 + 31]  #1
        y_data[j1 + 8] = label[i + 40]
        j1 += 9 # ใช้เลือกข้อมูลมาใส่แบบล็อคค่า
        j2 += 2 # ใช้เลือกข้อมูลมาใส่แบบล็อคค่า
    return x_data, y_data

# generate folds
def gen_folds( index, x, y ) : # ข้อมูลใน index คือแกนที่จะใช้แบ่งข้อมูลแต่ละ folds โดยจะเลือกข้อมูลจาก x และ y
    feature = np.zeros( [len(index), x.shape[1], 1] ) # สร้างรูปแบบข้อมูลสำหรับแทนที่พร้อมนำไปเข้าโมเดล ( sample, 3887, 1 )
    label = np.zeros( [len(index), 3] ) # สร้างรูปแบบข้อมูลสำหรับแทนที่พร้อมนำไปเข้าโมเดล ( sample, num_classes )
    for i in range( 0, len(index) ) :
        feature[i] = x[ index[i] ]
        label[i] = y[ index[i] ]
    return feature, label  

# pre-procesing data for model
def processing_data( x_data, y_data ) :
    num_classes = 3
    reduced_noise = ReNoise( x_data ) # renoise from audio data
    mfcc_data = audio_to_mfcc( reduced_noise ) # use mfcc function
    normaliz_data, min_max_value = normalization( mfcc_data ) # normalization data 0-1
    X_data = np.expand_dims( normaliz_data, -1 ) # reshape (sample, 3887, 1)
    Y_data = keras.utils.to_categorical( y_data, num_classes ) # แปลง labels เป็น one-hot encoding
    return X_data, Y_data, min_max_value

# pre-procesing data for evaluate model
def processing_test_data( x_data, y_data, MinMax ) :
    num_classes = 3
    reduced_noise = ReNoise( x_data ) # renoise from audio data
    mfcc_data = audio_to_mfcc( reduced_noise ) # use mfcc function
    normaliz_data = normalize_for_test( mfcc_data, MinMax ) # normalization data 0-1
    X_data = np.expand_dims( normaliz_data, -1 ) # reshape (sample, 3887, 1)
    Y_data = keras.utils.to_categorical( y_data, num_classes ) # แปลง labels เป็น one-hot encoding
    return X_data, Y_data

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
    rate = 44100 
    samples = features.shape[0] #เช็คจำนวนของข้อมูล 
    answer = np.zeros( (samples, 299, 13) ) #สร้างเพื่อนำมาใส่ข้อมูลหลังเข้า mfcc function
    for i in range( 0, samples ) :
        answer[ i : i+1 ] = mfcc( features[i], rate ) #นำไปเข้าฟังก์ชั่น MFCC 
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

def scaling_for_test( data_for_scale, min, max ) :
    min = np.float64( min )
    max = np.float64( max )
    rescale_data = ( data_for_scale - min ) / ( max - min )
    return rescale_data

def normalize_for_test( mfcc_feature, min_max_column ) :
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
        scale_data = scaling_for_test( data_to_normalize, min_max_column['min'][column], min_max_column['max'][column] ) # นำค่าที่ได้ไป scaling ให้อยู่ระหว่าง 0 - 1
        # append new scaling data
        for value in range( len(scale_data) ) :
            normalized_data[value][column] = scale_data[value]
    return normalized_data
######################################## Normalization set of function ########################################

def plot( train_data, validation_data, process ) :
    if process == 'loss' :
        title = 'Loss training process'
        ylabel = 'Loss'
    elif process == 'accuracy' :
        title = 'Accuracy training process'
        ylabel = 'Accuracy'
    # Show trained and validation loss
    plt.figure( figsize = (10, 6) ) #(10, 6)
    with plt.style.context( 'seaborn' ) :
        plt.title( title )
        plt.xlabel( 'Epochs' )
        plt.ylabel( ylabel )
        plt.plot( train_data, c = 'forestgreen', linewidth = 2.5, label = 'Training' )
        plt.plot( validation_data, c = 'darkorange', linestyle = '--', label = 'Validation' )
        plt.legend( frameon = True, facecolor = 'white' )
        plt.show()

# Train the model and plot graph
def train_model( model, X_train, Y_train, X_val, Y_val, epochs, batch_size ) :
    logs = model.fit( X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val) ).history # train
    train_accuracy = logs["accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ train_accuracy
    val_accuracy = logs["val_accuracy"][epochs - 1] # เลิอกค่าสุดท้ายของ val_accuracy
    train_loss = logs["loss"][epochs - 1] # เลิอกค่าสุดท้ายของ train_loss
    val_loss = logs["val_loss"][epochs - 1] # เลิอกค่าสุดท้ายของ val_loss
    plot( logs['loss'], logs['val_loss'], 'loss' ) # plot loss process
    plot( logs['accuracy'], logs['val_accuracy'], 'accuracy' ) # plot accuracy process
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

# Model / data parameters
input_shape = ( 3887, 1 ) # ปรับตามขนาดของข้อมูล
num_classes = 3
n_folds = 5

# import data
x_train, y_train = OpenTrainData()

# convert list to numpy array
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
units = 5
model_set = { "model" : [] } # สร้าง dict เก็บโมเดลทั้งหมด

for build in range( n_folds ) :
    model = keras.Sequential(
        [
            layers.LSTM( units = units, input_shape = input_shape, return_sequences = True ), #number of sequence (time steps), number of features
            #layers.LSTM( units = units, return_sequences = True ),
            layers.LSTM( units = units ),
            layers.BatchNormalization(),
            layers.Dense( num_classes, activation = "softmax" )
        ]
    )
    model_set["model"].append( model ) # เก็บโมเดลที่สร้างไว้ใน dict
model_set["model"][0].summary() #สรุปโครงสร้างของโมเดล แสดงเป็นตาราง

# Compile the model
for do in range( n_folds ) :
    model_set["model"][do].compile( loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )

# Train the model
batch_size = 18
epochs = 50
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

# Evaluate the trained model
x_test, y_test = OpenTestData()

# Convert list to numpy array
x_test = np.array( x_test )
y_test = np.array( y_test )

# Pre-processing test data
X_test, Y_test = processing_test_data( x_test, y_test, min_max_value )

score = Best_model.evaluate( X_test, Y_test, verbose = 0 ) #model.evaluate คือการคืนค่า loss และค่า metrics สำหรับโมเดลจากการทดสอบ 
y_predicted = np.zeros( y_test.shape )
print( "Test loss:", score[0] ) #แสดงค่า loss
print( "Test accuracy:", score[1] ) #แสดงค่า accuracy

predicted_classes = Best_model.predict( X_test ) #คาดการณ์จาก input ที่ใส่ไป
for i, predict in enumerate( predicted_classes[ : X_test.shape[0] ] ) : #ให้วนตามจำนวน sample test(X_test.shape[0])
    y_predicted[i] = TurnPredicted( predict )
    #print( "Predicted {} Class {}".format(predict, y_test[i]) )
    print( f"Predicted {predict} Class {y_test[i]}" )

# Show confusion matrix
ConfusionMatrixDisplay.from_predictions( y_test, y_predicted, display_labels = [ "Open", "Close", "Background Noise" ]  )
plt.show()
