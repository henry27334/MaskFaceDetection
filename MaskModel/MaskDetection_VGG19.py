import pandas as pd # 陣列操作
import numpy as np # 線性代數操作
import seaborn as sns #展示圖片
import os #取得路徑
from sklearn.utils import shuffle #用於模型打亂資料集
import matplotlib.pyplot as plt #展示圖片
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img #準備圖片
import cv2 # 視覺資料庫 用於臉部偵測
from scipy.spatial import distance
import glob
from warnings import filterwarnings
from tensorflow.keras.applications import VGG19 # 分類模組 主要的偵測模組
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from skimage import io
from io import BytesIO
from scipy.spatial import distance
import requests
import PIL

path  = "../input/Face Mask Dataset/"
outPath = "../outputModel/"
dataset = {"image_path":[],"mask_status":[],"where":[]}
IMG_SIZE = 150
MIN_DISTANCE = 0

#使用迴圈 讀取每個資料夾名稱/內容 進而傳回加入資料至dataframe
for where in os.listdir(path):
    for status in os.listdir(path+"/"+where):
        for image in glob.glob(path+where+"/"+status+"/"+"*.png"):
            dataset["image_path"].append(image)
            dataset["mask_status"].append(status)
            dataset["where"].append(where)
            
dataset = pd.DataFrame(dataset)
print(dataset)

withMask = dataset.value_counts("mask_status")[1]
withoutMask = dataset.value_counts("mask_status")[0]

print(f"With Mask: {withMask}\nWithout Mask: {withoutMask}\n")
# sns.countplot(dataset["mask_status"])
# plt.show()

## 穿戴口罩圖片九宮格
# plt.figure(figsize = (14,10))
# for i in range(9): 
#     random = np.random.randint(1,len(dataset))
#     plt.subplot(3,3,i+1)
#     plt.imshow(cv2.imread(dataset.loc[random,"image_path"]))
#     plt.title(dataset.loc[random, "mask_status"], size = 15, color = "brown") 
#     plt.xticks([])
#     plt.yticks([])
    
# plt.show()

train_df = dataset[dataset["where"] == "Train"]
test_df = dataset[dataset["where"] == "Test"]
valid_df = dataset[dataset["where"] == "Validation"]
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)
valid_df = valid_df.sample(frac=1)
print(train_df.head())
print(test_df.head())
print(valid_df.head())

## 數據量長條圖
# plt.figure(figsize = (15,6))
# plt.subplot(1,3,1)
# sns.countplot(train_df["mask_status"])
# plt.title("Train_df", size = 14, color = "orange")

# plt.subplot(1,3,2)
# sns.countplot(test_df["mask_status"])
# plt.title("Test_df", size = 14, color = "red")

# plt.subplot(1,3,3)
# sns.countplot(valid_df["mask_status"])
# plt.title("Validation_df", size = 14, color = "blue")

# plt.show()

datagen = ImageDataGenerator(rescale = 1./255) #圖像預處理物件

def get_generator(dframe):
    generator = datagen.flow_from_dataframe(
                        dataframe=dframe,
                        directory="../input",
                        x_col="image_path",
                        y_col="mask_status",
                        batch_size=60,
                        seed=42,
                        shuffle=False,
                        class_mode="binary",
                        target_size=(150,150))
    return generator

train_generator = get_generator(train_df)
valid_generator = get_generator(valid_df)
test_generator= get_generator(test_df)

def bulid_model():
    model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3)) #呼叫 VGG19模型
    for layer in model.layers[2:]: 
      layer.trainable = False
      
    x=Flatten()(model.output) #將model的output扁平化 
    x2 = Dense(128, activation="relu")(x) #全連接層 unit輸出空間維度 激勵函數使用relu
    output=Dense(1,activation='sigmoid')(x2) #全連接層 最後一層 unit輸出空間維度 激勵函數使用sigmoid
    model=Model(model.input,output) # 將model.input作為Model的輸入 並產生output
    model.summary() 
    
    # save best weights
    checkpoint = ModelCheckpoint("classify_model.h5", save_best_only=True, verbose = 1) #儲存最好的權重值
    model.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"]) 
    
    history = model.fit_generator(train_generator,
                              validation_data  = valid_generator, 
                              epochs = 5, 
                              steps_per_epoch=(len(train_generator.labels) / 80) ,
                              validation_steps=(len(valid_generator.labels)/80), 
                              callbacks =[checkpoint])
    
    return model, history

def showAccLoss(history):
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label = "train accuracy", color = "red")
    plt.plot(history.history["val_accuracy"], label = "validation accuracy", color = "blue")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label = "train loss", color = "red")
    plt.plot(history.history["val_loss"], label = "validation loss", color = "blue")
    
    plt.legend()
    plt.show()

# 創建模型並訓練
model, history = bulid_model()
showAccLoss(history)

# 載入已訓練好的模型
# model = load_model(outPath + "MaskDetect_model.h5")
model.evaluate_generator(test_generator, verbose=1)

predictions = model.predict_generator(test_generator, verbose = 1,workers=-1)

face_model = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread(train_df.loc[np.random.randint(1,len(train_df)),"image_path"]) #隨機抓取訓練圖片並讀取

img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE) #顏色空間轉換

faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #偵測人臉數量

out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #顏色空間轉換

for (x,y,w,h) in faces: 
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,255,255),1)
    
# plt.figure(figsize=(6,6))
# plt.imshow(out_img) 
# plt.show()

model.save(outPath + "MaskDetect_model.h5")

def getFaces(img): #getFaces函數 
    gray_img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_model.detectMultiScale(gray_img, scaleFactor=1.1, 
                                        minNeighbors=8)
    return faces

def newSize(width, height): #newSize函數
    if width < 600:
        return newSize(width * 1.12 , height * 1.12)
    
    if width >= 1200:
        return newSize(width / 1.12 , height / 1.12)
        
    return int(width), int(height)
        
def AdjustSize(f): #調整函數
    img = PIL.Image.open(f)
    width, height = img.size
    new_width, new_height = newSize(width, height)
    
    return (new_width, new_height)   

def Draw(img, face): #Draw 畫出臉部矩型
    (x,y,w,h) = face
    mask_label = {0:'Has Mask',1:'No Mask'}
    # mask_label = {0:'有戴口罩',1:'沒戴口罩'}

    label_color = {0: (0,255,0), 1: (255,0,0)}
    
    crop = img[y:y+h,x:x+w]
    
    crop = cv2.resize(crop,(IMG_SIZE, IMG_SIZE))
    crop = np.reshape(crop,[1,IMG_SIZE,IMG_SIZE,3]) / 255.0
    
    mask_result = model.predict(crop)
            
    pred_label = round(mask_result[0][0])
            
    cv2.putText(img,mask_label[pred_label],
                (x, y-10), cv2.FONT_HERSHEY_TRIPLEX,
                1, label_color[pred_label], 2)
            
    cv2.rectangle(img,(x,y),(x+w,y+h), 
                label_color[pred_label],3)
    
    return img    
    
def maskDetection_url(imgUri):   #maskDetection_url 將上列所有函數合併使用 要測試模型直接呼叫此函數即可
    response = requests.get(imgUri)
    f = BytesIO(response.content)
    
    img = io.imread(f)
    # resize = AdjustSize(f)
    # img = cv2.resize(img, resize)
    faces = getFaces(img)
    
    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2], 
                                          faces[j][:2])
                if dist < MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
                
        for i in range(len(faces)):
            Draw(img, faces[i])
                        
        plt.figure(figsize=(16,14))
        plt.imshow(img)
        plt.show()
            
    else:
        print("No Face!")
        
def maskDetection_image(image):   #maskDetection_imgae 將上列所有函數合併使用 要測試模型直接呼叫此函數即可
    
    img = io.imread(image)
    faces = getFaces(img)
    
    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2], 
                                          faces[j][:2])
                if dist < MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
                
        for i in range(len(faces)):
            Draw(img, faces[i])
                        
        plt.figure(figsize=(16,14))
        plt.imshow(img)
        plt.show()
            
    else:
        print("No Face!")
        

# maskDetection_url("https://specials-images.forbesimg.com/imageserve/1227664783/960x0.jpg?fit=scale")
# maskDetection_url("https://i.cbc.ca/1.5901311.1612993040!/fileImage/httpImage/image.jpg_gen/derivatives/original_780/face-recognition-test.jpg")  
# maskDetection_url("https://media.fromthegrapevine.com/assets/images/2020/3/coronavirus-mask-vatican-0302.jpg.824x0_q71_crop-scale.jpg")  
maskDetection_url("https://pgw.udn.com.tw/gw/photo.php?u=https://uc.udn.com.tw/photo/2021/01/16/1/11221668.jpg&x=0&y=0&sw=0&sh=0&sl=W&fw=800&exp=3600")

maskDetection_image("../input/Face Mask Dataset/Prediction/maskpic_1.jpg")