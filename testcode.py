from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
model=load_model("model.h5")
classes=["Acral_Lentiginous_Melanoma","blue_finger","clubbing","Healthy_Nail","Onychogryphosis","pitting"]
img_path=input("Enter the path of the image to test")
img=image.load_img(img_path,target_size=(128,128))
img_array=image.img_to_array(img)
img_array=img_array/255.0
img_array=np.expand_dims(img_array,axis=0)
result=model.predict(img_array)
predict_classs=np.argmax(result)
print("Prediction:",classes[predict_classs])
print("confidence:",np.max(result)*100,"%")