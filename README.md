# TUGAS BESAR PENGENALAN POLA

## Studi Kasus: Klasifikasi Kualitas Buah Menggunakan CNN

![Static Badge](https://img.shields.io/badge/windows-11-0078D4?logo=windows11&logoColor=0078D4)
![Static Badge](https://img.shields.io/badge/anaconda-23.7.4-44A833?logo=anaconda&logoColor=44A833)
![Static Badge](https://img.shields.io/badge/python-3.10.14-3776AB?logo=python&logoColor=3776AB)
![Static Badge](https://img.shields.io/badge/tensorflow-2.10.1-orange?logo=tensorflow&logoColor=orange)

### 📖 Deskripsi

Project ini berisi program untuk mengklasifikasikan 6 jenis buah berdasarkan kualitasnya. Dataset yang digunakan dalam project ini adalah dataset FRUITSGB: TOP INDIAN FRUITS WITH yang diperoleh dari IEEE DataPort.

### 💾 Dataset

Dataset berisi 12000 gambar buah berdimensi 256x256x3 (RGB) dengan format .jpg yang terdiri dari 6 jenis buah India yang paling banyak diekspor atau dikonsumsi.
Sebanyak 12000 gambar tersebut dibagi menjadi 12 kelas berbeda, yaitu:

    Bad Apple
    Good Apple
    Bad Banana
    Good Banana
    Bad Guava
    Good Guava
    Bad Lime
    Good Lime
    Bad Orange
    Good Orange
    Bad Pomegranate
    Good Pomegranate

![image](https://github.com/fahmifuady/pattern-recognition/assets/60414175/728b3066-d89c-45ef-b4de-8083eebccb49)

Dataset yang digunakan dapat diunduh di [IEEE DataPort](https://ieee-dataport.org/open-access/fruitsgb-top-indian-fruits-quality)

### 🛠️ Alur Kerja

- Dataset dibagi kedalam direktori train dan val program menggunakan `models/data.py` dengan perbandingan 80:20 dan disimpan dalam direktori `dataset_split/`.
- Arsitektur model dirancang pada file `models/cnn.py` sedangakan model jadi disimpan dengan nama `model.tflite` di root direktori.
- Model ini adalah Convolutional Neural Network (CNN)
  - Menggunakan lapisan konvolusi dan pooling untuk mengekstraksi fitur dari gambar input.
  - Dropout layer digunakan Untuk mencegah overfitting.
  - Fitur-fitur diekstraksi dari gambar diratakan menjadi satu dimensi dan diteruskan ke lapisan dense untuk klasifikasi.
  - Fungsi aktivasi ReLU digunakan di lapisan konvolusi dan dense.
  - Fungsi softmax digunakan di layer output untuk menghasilkan probabilitas kelas.
  - Loss function yang digunakan categorical crossentropy.
  - Optimizer yang digunakan Adam.
- Akurasi yang diperoleh saat evaluasi model dengan 100 epoch sebesar ~80%.
  ![Figure_1](https://github.com/fahmifuady/pattern-recognition/assets/60414175/7a636cd0-4a6a-4ae4-a347-ba6cacbf305e)

### 🚀 Anggota Kelompok

| Nama                          | NIM      | Jobdesc                  |
| :---------------------------- | :------- | :----------------------- |
| 1. Muhammad Husni Fahmi Fuady | 21102049 | Pembuat model & coding   |
| 2. Cheppi Garda Muhamad       | 21102163 | Mencari dataset & coding |
