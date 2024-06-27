# Импорт библиотек связанных с распознованием изображений
import sqlite3

import cv2
import numpy as np
import imutils
import pytesseract
import easyocr

# Импорт библиотек связанных с выделением доминантного цвета
from sklearn.cluster import KMeans
from collections import Counter
import datetime

# Импорт библиотек связанных с созданием интерфейса
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os

#Библиотека для работы с базами данных
import sqlite3 as sl

#Возможные символы
possible_letters=['А','В','Е','К','М','Н','О','Р','С','Т','У','Х']
possible_numbers=['1','2','3','4','5','6','7','8','9','0']
global car_num

#Работа с базами данных

def add_instance(carnumb, imgpath, pltcolor, curtime):
    with sqlite3.connect('db/database.db') as db:
        cursor = db.cursor()
        cursor.execute('''INSERT INTO Main(Carnumber, Imagepath, PlateColor, Time)
                          VALUES(?,?,?,?)''', (carnumb, imgpath, pltcolor, curtime))
        db.commit()

def check_registration(number):
    with sqlite3.connect('db/database.db') as db:
        cursor = db.cursor()
        cursor.execute('''SELECT id FROM Registrated_numbers WHERE number = ?''', (number, ))
        if cursor.fetchone():
            return "Зарегистрированный номер\n"
        else:
            return "Незарегестрированный номер\n"
        db.commit()

def pass_car(carnumb, curtime):
    with sqlite3.connect('db/database.db') as db:
        cursor = db.cursor()
        cursor.execute('''INSERT INTO Accepted_cars(Carnumber, Time)
                          VALUES(?,?)''', (carnumb, curtime))
        db.commit()

# Инициализация pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\tesseract.exe"

#Задание функции распознования номеров
def number_recognitor(name):
    # Чтение изображения полученого на входе
    img = cv2.imread(name)

    # Перевод изображения в чёрно-белый
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Задание каскада Хаара
    cascade = cv2.CascadeClassifier('haarcascades\haarcascade_russian_plate_number.xml')

    # Обнаружение номерных знаков
    plates = cascade.detectMultiScale(gray, 1.2, 5)
    print('Количество обнаруженных знаков:', len(plates))

    # Цикл проходящий по всем найденым знакам
    for (x,y,w,h) in plates:

        # Построение ограничивающего прямоугольника вокруг номерного знака
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        gray_plates = gray[y:y+h, x:x+w]
        color_plates = img[y:y+h, x:x+w]

        # Сохранение номерного знака в формате .jpg
        cv2.imwrite('Numberplate.jpg', gray_plates)
        cv2.imwrite(str(datetime.datetime.now())+".jpg", gray_plates)

        #Вывод знака найденого Pytesseract
        text = pytesseract.image_to_string(gray_plates, config='--psm 8')
        print(f'[PyTesseract] Номерные знаки обнаружены: {text.strip()}')

        # Вывод знака найденого EasyOCR
        reader = easyocr.Reader(['en'])
        result=reader.readtext(gray_plates)
        potential_plates=[]
        for i in result:
            potential_plates+=i[-2]
            potential_plates+='$'
        #Определение Цвета
            bgr_image = cv2.imread('Numberplate.jpg')
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            dom_color = get_dominant_color(hsv_image)
            dom_color_hsv = np.full(bgr_image.shape, dom_color, dtype='uint8')
        #Вывод номера, времени, цвета
        try:
            OCR_Answer=formator(potential_plates)
            print('[EasyOCR] Номерные знаки обнаружены: ',nomerator(OCR_Answer))
            current_time = str(datetime.datetime.now())
            #cv2.imwrite(current_time, gray_plates)
            Output.insert(END,'['+ current_time + '] ' +  nomerator(OCR_Answer)+' '+colorname(dom_color_hsv[0][0])+'\n')
            Output.insert(END, check_registration(nomerator(OCR_Answer)))
            add_instance(nomerator(OCR_Answer), current_time, colorname(dom_color_hsv[0][0]), current_time)
            if check_registration(nomerator(OCR_Answer))=="Зарегистрированный номер":
                pass_car(nomerator(OCR_Answer), current_time)
        except:
            Output.insert(END,'error!\n')

        cv2.waitKey(0)
        cv2.waitKey()
    cv2.destroyAllWindows()

#Создание UI
def showimage():
    filename=filedialog.askopenfilename(initialdir=os.getcwd(),title="Выберите изображение",filetypes=(("JPG File","*.jpg"),("PNG File","*.png"),("All file","how are you .txt")))
    number_recognitor(routeToFileName(filename))
    filename='C:/Users/zlygo/PycharmProjects/OpenCV_CarNumbers/Numberplate.jpg'
    img=Image.open(filename)
    img=ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image=img

#Вспомогательная функция
def formator(plates):
    word=''
    words=[]
    for i in plates:
        if i!='$':
            word+=i
        else:
            words.append(word)
            word=''
    words.sort(key=len)
    print(words)
    for k in range(len(words)):
        words[k] = mistake_remover(words[k])
    for j in words:
        if len(j)>14:
            words.remove(j)
    print(words)
    return words[-1]

#Работа с номером
def nomerator(gotten):
    global car_num
    car_num = gotten
    if len(car_num)>6:
        car_num=car_num[:6]
    car_num=car_num.upper()
    for i in range(len(car_num)):
        if (i == 0) or (i == 4) or (i==5):
            if car_num[i] not in possible_letters:
                if car_num[i] == '1': car_num = replace_letter(car_num,i,"T")
                if car_num[i] == '4': car_num = replace_letter(car_num,i,"A")
                if car_num[i] == '5': car_num = replace_letter(car_num,i,"S")
                if car_num[i] == '6': car_num = replace_letter(car_num,i,"G")
                if car_num[i] == '7': car_num = replace_letter(car_num,i,"T")
                if car_num[i] == '8': car_num = replace_letter(car_num,i,"B")
                if car_num[i] == '0': car_num = replace_letter(car_num,i,"O")
                if car_num[i] == 'D': car_num = replace_letter(car_num,i,"O")
        if (i == 1) or (i == 2) or (i==3):
            if car_num[i] not in possible_numbers:
                if car_num[i] == 'I': car_num = replace_letter(car_num,i,"1")
                if car_num[i] == 'J': car_num = replace_letter(car_num,i,"3")
                if car_num[i] == 'A': car_num = replace_letter(car_num,i,"4")
                if car_num[i] == 'S': car_num = replace_letter(car_num,i,"5")
                if car_num[i] == 'G': car_num = replace_letter(car_num,i,"6")
                if car_num[i] == 'T': car_num = replace_letter(car_num,i,"7")
                if car_num[i] == 'Y': car_num = replace_letter(car_num,i,"7")
                if car_num[i] == 'B': car_num = replace_letter(car_num,i,"8")
                if car_num[i] == 'O': car_num = replace_letter(car_num,i,"0")
    return car_num

#Вспомогательная библиотека
def replace_letter(word,position,replacement):
    s = list(word)
    s[position] = replacement
    return "".join(s)
def routeToFileName(route):
    return route.split("/")[-1]

#Определение доминантного цвета
def get_dominant_color(image, k=4):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)

global fixtext

def mistake_remover(word):
    symbols_to_remove = ",!?./[]"
    global fixtext
    fixtext = word
    for symbol in symbols_to_remove:
        fixtext = fixtext.replace(symbol, "")
    return fixtext


#Перевод из цвета в его название
def colorname(input_color):
    if int(input_color[2]) < 180:
        return "White"
    elif int(input_color[0]) < 15:
        return "Red"
    elif int(input_color[0]) < 40:
        return "Yellow"
    elif int(input_color[0]) < 120:
        return "Blue"
    else: return "Color undefined"

#Задание элементов интерфейса
root=Tk()

fram=Frame(root)
fram.pack(side=BOTTOM,padx=15,pady=15)

lbl=Label(root)
lbl.pack()

btn=Button(fram,text="Выбрать изображение",command=showimage)
btn.pack(side=tk.LEFT)

btn2=Button(fram,text="Выход",command=lambda:exit())
btn2.pack(side=tk.LEFT,padx=12)

Output = Text(fram, height = 5, width = 50)
Output.pack(side=tk.LEFT,padx=12)

root.title("Распознование номеров DEMO")
root.geometry("700x425")
root.mainloop()



