#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:07:56 2018

@author: aky15
"""

import tkinter as tk
import cv2
import numpy as np
import dlib
from skimage import draw
predictor_path = "shape_predictor_68_face_landmarks.dat"
cap = cv2.VideoCapture(0)# set blue thresh
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
def nothing(x):
    pass

def btnHelloClicked():
   flag=0
   cv2.namedWindow('image')
   cv2.createTrackbar('R','image',0,255,nothing)
   cv2.createTrackbar('G','image',0,255,nothing)
   cv2.createTrackbar('B','image',0,255,nothing)
   cv2.createTrackbar('mix_ratio','image',0,100,nothing)
   cv2.createTrackbar('illumination','image',0,100,nothing)
   while(1):    # get a frame and show   
    ret, img = cap.read()   
    dets = detector(img, 1)
    b=cv2.getTrackbarPos('B','image')
    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G','image')
    mix_ratio=cv2.getTrackbarPos('mix_ratio','image')
    illu=cv2.getTrackbarPos('illumination','image')
    
    for k, d in enumerate(dets):
        ori_img=img.copy()
        shape = predictor(img, d) 
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        x = landmark[48:68,0]
        y = landmark[48:68,1]
        rr, cc=draw.polygon(y,x)
        a=img[rr,cc]
        COLOR_target=[[[b,g,r]]]
        n=int(a.shape[0]/10)
        if flag==0:
          s=COLOR_target
        if flag==1:
          s=powder(COLOR_target,n)
        draw.set_color(img,[rr,cc],s)
        alpha=mix_ratio/100
        beta=1-alpha
        img_add = cv2.addWeighted(ori_img, alpha, img, beta, illu)
        cv2.imshow('Face', img_add)

    if cv2.waitKey(1) & 0xFF == ord('a'):      
         flag=1
    if cv2.waitKey(1) & 0xFF == ord('r'):      
         flag=0
    if cv2.waitKey(1) & 0xFF == ord('q'):      
        cap.release()
        cv2.destroyAllWindows() 

    

def powder(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        img[j,i,0]= 0  
        img[j,i,1]= 215    
        img[j,i,2]= 255
    return img
    
root = tk.Tk()  
root.title("A Simple Make-Up Demo")
logo = tk.PhotoImage(file="1.gif")  
w1 = tk.Label(root, image=logo).pack(side="right")  
explanation = """
This is a simple make-up demo.
push button A to add powder.
push button R to remove powder.
push button Q to close the camera.
Have fun using it.""" 
labelHello = tk.Label(root, text = explanation, height = 9, width = 50, fg = "black",font = ('微软雅黑',15))
labelHello.pack()

btnCal = tk.Button(root, text = "start",font = ('微软雅黑',15),width=10, height=1 ,command = btnHelloClicked)
btnCal.pack()


root.mainloop()        