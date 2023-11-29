#!/usr/bin/env python
# coding: utf-8

# # 1. İlgili Kütüpkanalerin yüklenmesi

# In[1]:


import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib


# In[2]:


print(tf.__version__)


# # 2. Verilerin yüklenmesi

# In[3]:


# Verileri bilgisayarımdan yüklüyorum. Verileri bu adresten indirebilirsiniz.
# Adres -> https://www.kaggle.com/datasets/puneet6060/intel-image-classification
train_dir = pathlib.Path("C:\\Users\\Dell\Desktop\\Intel Image Classification\\seg_train\\seg_train")
test_dir = pathlib.Path("C:\\Users\\Dell\\Desktop\\Intel Image Classification\\seg_test\\seg_test")


# In[4]:


# eğitim verisindeki toplam resim sayısı
image_count = len(list(train_dir.glob('*/*.jpg')))
print(image_count)


# In[5]:


# test verisindeki toplam resim sayısı
image_count = len(list(test_dir.glob('*/*.jpg')))
print(image_count)


# In[6]:


# Eğitim dosyasındaki forest sınıfından örnek bir resim
forest = list(train_dir.glob('forest/*'))
PIL.Image.open(str(forest[0]))


# In[7]:


# Test dosyasındaki street sınıfından örnek bir resim
street = list(test_dir.glob('street/*'))
PIL.Image.open(str(street[0]))


# In[8]:


# veri kümesini oluşturuyoruz, paket boyutunu 32 olarak resim gelişlik ve yükseklik boyutunu orjinal boyutunu korumak için 150
# olarak aldım.
batch_size = 32
img_height = 150
img_width = 150
train_ds = tf.keras.utils.image_dataset_from_directory(
      train_dir,
      seed=120,
      image_size=(img_height, img_width),
      batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory(
      test_dir,
      seed=120,
      image_size=(img_height, img_width),
      batch_size=batch_size)


# In[9]:


# veri setindeki sınıf isimleri
class_names = train_ds.class_names
print(class_names)


# In[10]:


#eğitim veri setindeki ilk dokuz resim
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
      for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


# In[11]:


# burada eğitime girecek bir tensörün boyutunu görebilmekteyiz. İlk satır 32 adet (batch size) 3 kanallı 150x150 boyutunda
# resimden oluştuğunu, ikinci satır bu resimlere karşılık gelen sınıf numaralarının adetini veriyor.
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# # 3. Görsellerin eğitim için hazırlanması

# In[12]:


# resimlerdeki piksel değerleri 0 ile 255 değerleri arasındadır. Bu değerleri eğitmek için uygun değildir. Uygun hale getirmek
# için verilerimizi standartlaşma işlemine tabi tutuyoruz. İşlem sonrasında değerlerin 0 ile 1 arasında olduğunu kontrolünü
# yapıyoruz
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))


# In[13]:


# işlem gören veri paketinden sonra gelecek olan veri paketini ön belleğe alır, böylece modelimizin daha hızlı çalışmasını
# sağlarız.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# # 4. Evrişim Modelinin Oluşturulması

# In[14]:


# modelimiz 5 evrişim katmanından oluşuyor. 1. katmanda 64 kanal, 2. ve 3. katmanda 32 kanal ve 4. ve 5. katmanda 16 kanal
# bulunmakatadır. filtre olarak her katmanda 3x3 luk filtre kullanıldı. Stride default olarak (1) seçildi ve padding uygulanmadı
# aktivasyon fonkiyonu olarak relu seçildi. son gizli katmanda ise 6 adet sınıf olduğu için 6 nöron eklendi.
num_classes = 6

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),    
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),    
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),    
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),    
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),    
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])


# In[15]:


# derlemede, optimizer olarak adam, loss fonksiyonu olarak verimiz 6 sınıf olduğundan Categorical Crossentropy ve başarı 
# ölçüm metriği olarak accuracy seçildi.
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


# # 5. Modelin Eğitim Aşaması

# In[16]:


history=model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=11
    )


# # 6. Model Özeti ve Değerlendirilmesi

# In[17]:


model.summary()


# In[18]:


model_loss=history.history
loss=model_loss["loss"]
val_loss=model_loss["val_loss"]
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(loss,label="loss",color='tab:blue')
ax.plot(val_loss,label="val_loss",color='tab:orange')


# In[19]:


test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print('\nTest accuracy:', test_acc)


# In[20]:


# Sonuç

# Loss ve val_loss değerlerine ve grafiğe baktığımızda 9. epoch'dan sonra modelimiz aşırı öğrenmeye (overfitting) girdiğini 
# görüyoruz. Bu da bizi 9 epoch dan sonra eğitime devam etmenin gerek olmadığını gösteriyor.
# Test accuracy %0.80 olarak hesaplanmıştır. 

# Daha önceki hesaplamalarda 1. ve 2. katmanda 16 kanal, 3. ve 4. katmanda 32 kanal ve 5. katmanda 64 kanal yapıldığında
# veya tüm katmanlar 32 kanal yapıldığında 3. epoch dan sonra aşırı öğrenmeye girdiğini (overfitting) ve accuracy 
# değerinin %0.65-0.75 aralığında olduğu gözlenmiştir.

# Daha iyi bir sonuç için verilerde filtreleme işlemleri yapılabilir, shuffle gibi işlemler uygulanabilir veya dropout katmanı
# eklenebilir.


# # 8. Test Görselleri ile Birkaç Görselin Tahmin Edilmesi

# In[21]:


def plot_image(i, predictions_array, true_label, img):
    true_label = true_label[i]
    img=img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.numpy().astype("uint8"))

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(6))
    plt.yticks([])
    thisplot = plt.bar(range(6), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


# In[22]:


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])


# In[23]:


predictions = probability_model.predict(test_ds)
image_batch, label_batch = next(iter(test_ds))
print(f"Sınıf Adları: 0={class_names[0]}, 1={class_names[1]}, 2={class_names[2]}, 3={class_names[3]}, 4={class_names[4]}, 5={class_names[5]}")
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], label_batch, image_batch)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  label_batch)
plt.show()


# In[24]:


predictions = probability_model.predict(test_ds)
image_batch, label_batch = next(iter(test_ds))
print(f"Sınıf Adları: 0={class_names[0]}, 1={class_names[1]}, 2={class_names[2]}, 3={class_names[3]}, 4={class_names[4]}, 5={class_names[5]}")
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], label_batch, image_batch)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], label_batch)
plt.tight_layout()
plt.show()


# In[ ]:




