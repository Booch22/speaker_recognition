from sklearn.linear_model import LinearRegression #นำเข้าโมเดล
from sklearn.metrics import mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv( "Flavio.csv" ) #เปิดไฟล์ csv
year =  data["Year"] #เลือกคอลัมน์ที่จะใส่ข้อมูลเข้าไปในตัวแปร
numb = data["NumberOfName"] #เลือกคอลัมน์ที่จะใส่ข้อมูลเข้าไปในตัวแปร
#Year = year[30:52] #เลือกแถวที่ต้องการ แถวที่ 30-51
#Numb = numb[30:52] #เลือกแถวที่ต้องการ แถวที่ 30-51
x = np.array(year)
Y = np.array(numb)
X = np.array(x).reshape( -1, 1 ) #เปลี่ยนให้สมาชิกใน aaray แต่ละตัวมีมิติเป็นของตัวเอง

model = LinearRegression() #นำเข้าโมเดล
model.fit( X, Y ) #จับคู่ x, y เพื่อเทรนด์
output = model.predict(X) #ค่าที่คาดการณ์เก็บไว้ใน output
predick = model.predict( [[2025]] ) #คาดการณ์ปี 2025 
print( 'years = 2025 predicted =', predick[0].astype("int32") ) #แสดงผลลัพธ์การทำนาย
print("Mean squared error: %.2f" % mean_squared_error( Y, output ) )

plt.scatter( X, Y ) #พล็อตจุดค่าที่แท้จริง
plt.plot( X, output, color = "red", linewidth = 3 ) #พล็อตค่าที่โมเดลทำนาย
plt.ylabel( "NumberOfName", color = "k" ) #ตัังชื่อแกน y
plt.xlabel( "year", color = "k" ) #ตั้งชื่อแกน x
plt.title( "Flavio" ) #ตั้งชื่อหัวข้อ
plt.show()
