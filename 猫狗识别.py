# -*- coding: UTF-8 -*-
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from tensorflow.keras import optimizers,losses
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.preprocessing import image
import numpy as np

train_cats_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\train\\cats'
train_dogs_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\train\\dogs'
test_cats_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\test\\cats'
test_dogs_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\test\\dogs'
validation_cats_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\validation\\cats'
validation_dogs_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\validation\\dogs'
train_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\train'
test_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\test'
validation_dir='D:\\DYM\\Python\\DeepLearning\\DownloadFile\\dogs-vs-cats\\dogs-vs-cats-small\\validation'

# 建立模型
model_vgg16=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
model=Sequential()
model.add(model_vgg16)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
mine_loss=losses.binary_crossentropy
rmsprop=optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop,loss=mine_loss,metrics=['acc'])

# 训练、测试图片处理
train_datagen=ImageDataGenerator(
    rotation_range=40,
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator=test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

#模型适配
history=model.fit(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

history=history.history
acc=history['acc']
val_acc=history['val_acc']
epochs=range(1,len(acc)+1)
# 画出训练集和验证集的准确率
plt.plot(epochs,acc,'bo',label='training accuracy')
plt.plot(epochs,val_acc,'b',label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend()
plt.show()

# 照一张图片验证猫狗
img_path='D:/DYM/Python/DeepLearning/DownloadFile/dogs-vs-cats/dogs-vs-cats-small/play/4.jpg'
predict_img=image.load_img(img_path,target_size=(150,150))
img=image.img_to_array(predict_img)
img=img/255
img=np.expand_dims(img,axis=0)
result=model.predict(img)
# print(result)
if result[0,0] >= 0.5:
    print('dog')
else:
    print('cat')
