import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import torch
from torch.utils import data as data_
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import os
import imageio
import serial
import time
import serial.tools.list_ports

model_path = input("Please enter the path to your model file: ")
# 等待連接建立
time.sleep(2)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

#pt 前就要有CNN
def detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    arduino_ports = []

    for port in ports:
        if 'Arduino' in port.description:  # 檢查描述中是否包含 "Arduino" 字串
            arduino_ports.append(port.device)  # 獲取COM端口

    return arduino_ports

arduino_ports = detect_arduino_port()

if arduino_ports:
    print("找到以下Arduino的COM端口：")
    for port in arduino_ports:
        print(port)
else:
    print("找不到任何Arduino的COM端口。")



# 設定串列連接的埠號和波特率（根據你的Arduino設定調整）
port = port  # 埠號（Windows作業系統通常為COMX，X為埠號；Linux作業系統通常為/dev/ttyUSBX，X為埠號）
baud_rate = 9600  # 波特率（Arduino程式中的Serial.begin的數值）

# 建立串列連接
ser = serial.Serial(port, baud_rate, timeout=1)


def predict(im):
    im = im.resize((28,28))
    #convert rgb to grayscale
    im = im.convert('L')
    im = np.array(im)

    #imageio.imwrite('temp.png', im)
    im = im/255.0

    im_d = torch.from_numpy(im[None, ...][None, ...]).to(device).float()
    output_d = model(im_d)

    #label_d = torch.from_numpy(np.array([label])).to(device)
    output = output_d.detach().cpu().numpy()
    result = np.argmax(output)

   
    prob = softmax(output[0])[result]
   
    print(prob)

    if result == 1:
        spell = 'Incedio🔥'
        command = f"incendio"
        ser.write(command.encode())
        print("hi")
    elif result == 2:
        spell = 'aqua🌊'
        command = f"aqua_{prob}"
        ser.write(command.encode())
        print("his")
    elif result == 3:
        spell = 'leviosa🪶'
    elif result == 4:
        spell = 'arresto🖐️'
    elif result == 5:
        spell = 'alohomora🔓'
    elif result == 6:
        spell = 'lumos💡' 
        command = f"lumos_{prob}"
        ser.write(command.encode())
        print("hi")
    return spell


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
  





# 以下是原始的 predict_digitX 和 predict_digit 函式



model = torch.load(model_path, map_location=torch.device('cpu'))

model.eval()
print(type(model))

global offset
offset = 10

def pc():
    global offset
    offset = 10
    print(offset)
    
def laptop():
    global offset
    offset = 70
    print(offset)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = 0
        self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.button_laptop = tk.Button(self, text="Laptop", command=laptop)
        self.button_pc = tk.Button(self, text="PC", command=pc)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.button_laptop.grid(row=3, column=0, pady=2)
        self.button_pc.grid(row=3, column=1, pady=2)
        self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")


    def classify_handwriting(self):
        global offset
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        #PC: +10 , laptop: +70
        #offset = 10
        #print(offset)
        
        rect1 = (rect[0]+offset, rect[1]+offset, rect[2]+offset, rect[3]+offset)
        im = ImageGrab.grab(rect1)

        spell_str = predict(im)
        self.label.configure(text=spell_str)

    def draw_linesXX(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

    def start_pos(self, event):
        self.x = event.x
        self.y = event.y

    def draw_lines(self, event):
        
        x1 = self.x
        y1 = self.y
        x2 = event.x
        y2 = event.y
        self.canvas.create_line(x1, y1, x2, y2, fill='black', width=8)
        
        self.x = x2
        self.y = y2
    
    


app = App()
app.mainloop()
