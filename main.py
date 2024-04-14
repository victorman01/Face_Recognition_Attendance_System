import cv2
import numpy as np
import os
import openpyxl
from datetime import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def registrasi():
   name = input("Masukkan nama untuk registrasi: ")
   if not os.path.exists('images'):
      os.makedirs('images')

   video_capture = cv2.VideoCapture(0)
   while True:
      ret, frame = video_capture.read()
      cv2.imshow('Registrasi', frame)
      if cv2.waitKey(1) & 0xFF == ord('c'):
         image_path = os.path.join('images', f'{name}.jpg')
         cv2.imwrite(image_path, frame)
         print("Gambar registrasi berhasil disimpan.")
         break

   video_capture.release()
   cv2.destroyAllWindows()
   return name, image_path

def training():
   images = []
   labels = []
   label_ids = {}
   current_id = 0
   for root, dirs, files in os.walk("images"):
      for file in files:
         if file.endswith("jpg"):
               path = os.path.join(root, file)
               label = os.path.splitext(os.path.basename(file))[0]
               if label not in label_ids:
                  label_ids[label] = current_id
                  current_id += 1
               id_ = label_ids[label]
               img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
               if img is not None:
                  faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(15, 15))
                  faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                  if faces:
                     x, y, w, h = faces[0]
                     face_roi = img[y:y + h, x:x + w]
                     images.append(face_roi)
                     labels.append(id_)
                  else:
                     print("No faces found")

   labels = np.array(labels)
   print(labels)
   face_recognizer = cv2.face.LBPHFaceRecognizer.create()
   face_recognizer.train(images, labels)
   face_recognizer.save('models/trained_model.yml')
   print("Model training selesai.")

def absensi():
   video_capture = cv2.VideoCapture(0)
   face_recognizer = cv2.face.LBPHFaceRecognizer_create()
   face_recognizer.read('models/trained_model.yml')
   wb = openpyxl.load_workbook('absensi.xlsx')
   sheet = wb.active
   labels = [os.path.splitext(name)[0] for name in os.listdir("images") if name.endswith('.jpg')]

   while True:
      ret, frame = video_capture.read()
      name = "Unknown"
      confidence = 0
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(15,15))
      faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
      
      if faces:
         for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_roi)
            
            if 100 - confidence >= 55:
               name = labels[label]
               confidence_text = f'Confidence: {round(100 - confidence)}%'
            else:
               name = "Unknown"
               confidence_text = ""
            
            text = f'{name} {confidence_text}'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
      cv2.imshow('Video', frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
         if name != "Unknown" and (100-confidence) >= 55:
            waktu_absen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet.append([name, waktu_absen])
            wb.save('absensi.xlsx')
            print(f"Absen berhasil untuk: {name} pada: {waktu_absen}")
            break
         else:
            confidence = 100 - confidence if confidence != 0 else confidence
            print(f"Absensi gagal untuk name: {name} confidence: {round(confidence)}%")
      elif key == ord('w'):
         print(f"Absensi dibatalkan")
         break

   video_capture.release()
   cv2.destroyAllWindows()

while True:
   print("Menu:")
   print("1. Registrasi")
   print("2. Training Model")
   print("3. Absensi")
   print("4. Keluar")

   pilihan = input("Pilih menu: ")

   if pilihan == '1':
      name, image_path = registrasi()
      if name is not None:
         print(f"Registrasi berhasil untuk: {name}")

   elif pilihan == '2':
      training()

   elif pilihan == '3':
      absensi()

   elif pilihan == '4':
      print("Selamat Tinggal!")
      break

   else:
      print("Pilihan tidak valid. Silakan pilih lagi.")
