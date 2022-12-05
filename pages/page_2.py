import torch
# import torchvision
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms as T
from torchvision import io
# from torchsummary import summary
import json
import numpy as np
import matplotlib.pyplot as plt

# делаем словарь, чтобы по индексу найти название класса
labels = json.load(open('imagenet_class_index.json'))
# функция декодировки
decode = lambda x: labels[str(x)][1]

# загружаем модель
# https://pytorch.org/vision/stable/models.html
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v3_small", pretrained=True)

resize = T.Resize((224, 224))


img = resize(io.read_image('image2.jpg')/255)

plt.imshow(torch.permute(img, (1, 2, 0)))

classes = []
probabilities = model(img.to(device).unsqueeze(0)).tolist()
for _ in range(5):
    index_prop = probabilities[0].index(max(probabilities[0]))
    cl = decode(index_prop)
    classes.append(cl)
    probabilities[0][index_prop] = min(probabilities[0])
classes


import streamlit as st

def main():

    st.subheader('"Это приложение определяет 5 наиболее вероятных классов изображения с помощью модели mobilenet_v3_small" :ru:')
    st.markdown("# Функция ❄️")
    st.sidebar.markdown("Определение класса объекта")
    img = st.file_uploader("Choose an image", ["jpg","jpeg","png"]) #image uploader
    img = resize(io.read_image(img)/255)
    classes = []
    probabilities = model(img.to(device).unsqueeze(0)).tolist()
    for _ in range(5):
        index_prop = probabilities[0].index(max(probabilities[0]))
        cl = decode(index_prop)
        classes.append(cl)
        probabilities[0][index_prop] = min(probabilities[0])

    st.write('Наиболее вероятные классы:', classes)

if __name__ == '__main__':
         main()



st.write("# Страница 2")