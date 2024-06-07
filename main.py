
import tkinter as tk
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd

window = tk.Tk()
window.geometry('1366x768')
window.title("Распознование лиц")
window.configure(background='#9dc7c7')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(
	window, text="Распознование лиц",
	bg="#116062", fg="#f7fafc", width=50,
	height=3, font=('times', 30, 'bold'))

message.place(x=100, y=50)

lbl = tk.Label(window, text="ID",
			width=20, height=2, fg="#116062",
			bg="#f7fafc", font=('times', 15, ' bold '))
lbl.place(x=400, y=270)

txt = tk.Entry(window,
			width=20, bg="#f7fafc",
			fg="#116062", font=('times', 15, ' bold '))
txt.place(x=700, y=285)

lbl2 = tk.Label(window, text="Имя",
				width=20, fg="#116062", bg="#f7fafc",
				height=2, font=('times', 15, ' bold '))
lbl2.place(x=400, y=370)

txt2 = tk.Entry(window, width=20,
				bg="#f7fafc", fg="#116062",
				font=('times', 15, ' bold '))
txt2.place(x=700, y=385)


'''row2 = ['Id', 'Name']
with open(r'UserDetails\ UserDetails.csv', 'a+') as csvFile:
    writer = csv.writer(csvFile)
    # Запись строки в csv-файл
    writer.writerow(row2)
csvFile.close()'''

# Функция, приведенная ниже, используется для проверки
# является ли текст ниже числом или нет?
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass

	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass

	return False
# Take Images - это функция, используемая для создания
# выборки изображений, которая используется для
# обучения модели. Требуется 60 изображений
# каждого нового пользователя.


def TakeImages():

	# Для распознавания изображения используются как идентификатор, так и имя
	Id = (txt.get())
	name = (txt2.get())

	# Проверяем, является ли идентификатор цифровым, а имя - алфавитным.
	if(is_number(Id) and name.isalpha()):
		# Открывается основная камера
		cam = cv2.VideoCapture(0)
		# Указание пути к файлу haarcascade
		harcascadePath = "dataset\haarcascade_frontalface_default.xml"
		# Создаем классификатор на основе файла haarcascade.
		detector = cv2.CascadeClassifier(harcascadePath)
		# Инициализируем номер образца (количество изображений) как 0
		sampleNum = 0
		while(True):
			# Просмотр видеозаписей, снятых камерой, кадр за кадром
			ret, img = cam.read()
			# Преобразование изображения в оттенки серого
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Он преобразует изображения в разные размеры
			# (уменьшает в 1,3 раза), а 5 указывает
			# количество раз, когда выполняется масштабирование
			faces = detector.detectMultiScale(gray, 1.3, 5)

			# Для создания прямоугольника вокруг изображения
			for (x, y, w, h) in faces:
				# Указываем координаты изображения, а также
				# цвет и толщину прямоугольника.
				# увеличиваем количество образцов для каждого изображения.
				cv2.rectangle(img, (x, y), (
					x + w, y + h), (255, 0, 0), 2)
				sampleNum = sampleNum + 1

				# сохраняем захваченное лицо в папке dataset
				# Тренируем изображение, так как изображение нуждается в тренировке
				# сохраняем в этой папке
				cv2.imwrite(
					"TrainingImage\ "+name + "."+Id + '.' + str(
						sampleNum) + ".jpg", gray[y:y + h, x:x + w])
				# отобразите захваченный кадр
				# и нарисуйте прямоугольник вокруг него.
				cv2.imshow('frame', img)
			#  подождите 100 миллисекунд
			if cv2.waitKey(100) & 0xFF == ord('q'):
				break
			# прервать, если номер выборки больше 60
			elif sampleNum > 60:
				break
		# освобождение ресурсов
		cam.release()
		# закрытие всех окон
		cv2.destroyAllWindows()
		# Отображение сообщения для пользователя
		res = "Изображениея сохранены: ID: " + Id + " Name: " + name
		# Создаем запись для пользователя в строке csv-файла
		row = [Id, name]
		with open(r'UserDetails\UserDetails.csv', 'a+') as csvFile:
			writer = csv.writer(csvFile)
			# Запись строки в csv-файл
			writer.writerow(row)
		csvFile.close()
		message.configure(text=res)
	else:
		if(is_number(Id)):
			res = "Введите Имя"
			message.configure(text=res)
		if(name.isalpha()):
			res = "Введите ID"
			message.configure(text=res)

# Тренировка изображений, сохраненных в папке с обучающими изображениями.

def TrainImages():
	# Гистограмма локального бинарного шаблона - это средство распознавания лиц
	# алгоритм внутри модуля OpenCV, используемый для обучения распознавателя набора данных
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	# Указание пути к файлу HaarCascade
	harcascadePath = "dataset\haarcascade_frontalface_default.xml"
	# создание детектора лиц
	detector = cv2.CascadeClassifier(harcascadePath)
	# Сохранение обнаруженных лиц в переменных
	faces, Id = getImagesAndLabels("TrainingImage")
	# Сохранение обученных лиц и их соответствующих идентификаторов
	# в модели с именем "trainer.yml".
	recognizer.train(faces, np.array(Id))
	recognizer.save("TrainingImageLabel\Trainer.yml")
	# Отображение сообщения
	res = "Модель обучена"
	message.configure(text=res)


def getImagesAndLabels(path):
	# получает пути ко всем файлам в папке
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	faces = []
	# создаем пустой список ID
	Ids = []
	# теперь перебираем все пути к изображениям и загружаем
	# ID и изображения, сохраненные в папке
	for imagePath in imagePaths:
		# загрузка изображения и преобразование его в цветовую гамму
		pilImage = Image.open(imagePath).convert('L')
		# Теперь мы преобразуем изображение PIL в массив numpy
		imageNp = np.array(pilImage, 'uint8')
		# получение ID из изображения
		Id = int(os.path.split(imagePath)[-1].split(".")[1])
		# извлеките лицо из образца обучающего изображения
		faces.append(imageNp)
		Ids.append(Id)
	return faces, Ids


# Для этапа тестирования

def TrackImages():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	# Считывание обученной модели
	recognizer.read("TrainingImageLabel\Trainer.yml")
	harcascadePath = "dataset\haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(harcascadePath)
	# получаем имя из файла "userdetails.csv"
	df = pd.read_csv(r"UserDetails\UserDetails.csv")
	cam = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	while True:
		ret, im = cam.read()
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.2, 5)
		for(x, y, w, h) in faces:
			cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
			Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
			if(conf < 50):
				aa = df.loc[df['Id'] == Id]['Name'].values
				tt = str(Id)+"_"+aa
			else:
				Id = 'Unknown'
				tt = str(Id)
			if(conf > 75):
				noOfFile = len(os.listdir("ImagesUnknown"))+1
				cv2.imwrite("ImagesUnknown\Image" +
							str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
			cv2.putText(im, str(tt), (x, y + h),
						font, 1, (255, 255, 255), 2)
		cv2.imshow('im', im)
		if (cv2.waitKey(10)==27):
			break
	cam.release()
	cv2.destroyAllWindows()


takeImg = tk.Button(window, text="Новый пользователь",
					command=TakeImages, fg="#f7fafc", bg="#116062",
					width=20, height=3, activebackground="#1d1d8f",
					font=('times', 15, ' bold '))
takeImg.place(x=130, y=520)
trainImg = tk.Button(window, text="Обучение",
					command=TrainImages, fg="#f7fafc", bg="#116062",
					width=20, height=3, activebackground="#1d1d8f",
					font=('times', 15, ' bold '))
trainImg.place(x=430, y=520)
trackImg = tk.Button(window, text="Распознать",
					command=TrackImages, fg="#f7fafc", bg="#116062",
					width=20, height=3, activebackground="#1d1d8f",
					font=('times', 15, ' bold '))
trackImg.place(x=730, y=520)
quitWindow = tk.Button(window, text="Выход",
					command=window.destroy, fg="#f7fafc", bg="#116062",
					width=20, height=3, activebackground="#1d1d8f",
					font=('times', 15, ' bold '))
quitWindow.place(x=1030, y=520)


window.mainloop()
