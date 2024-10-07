# -*- coding: utf-8 -*-
"""
Simple code for selecting object and background zone for the segmentation
"""

from tkinter import *
from PIL import Image, ImageTk
current_image = 0

def interactive_segmentation(list_url):
    loaded_image = {}
    global mode
    mode = True #True = background, False = object
    global current_image 
    current_image = 0
    background_list = [set() for _ in range(len(list_url))]
    object_list = [set() for _ in range(len(list_url))]
    
    
    def get_x_and_y(event):
        global lasx, lasy
        lasx, lasy = event.x, event.y
    
    def change_mode():
        global mode
        mode = not(mode)
        
    def draw_smth(event):
        global lasx, lasy, mode
        if mode:
            color = "blue"
        else:
            color = "red"
        canvas.create_line((lasx, lasy, event.x, event.y), 
                      fill=color, 
                      width=2)
        lasx, lasy = event.x, event.y
        if mode:
            background_list[current_image].add((lasx,lasy))
        else:
            object_list[current_image].add((lasx,lasy))

    def done():
        app.destroy()
        return(background_list,object_list)
    
    def go_left():
        global current_image
        if current_image != 0:
            current_image -= 1
        else:
            current_image = len(list_url)-1
        
        canvas.itemconfig(image_container,image=loaded_image["img" + str(current_image)])
    
    def go_right():
        global current_image
        if current_image != len(list_url)-1:
            current_image += 1
        else:
            current_image = 0
        
        canvas.itemconfig(image_container,image=loaded_image["img" + str(current_image)])

        
    app = Tk()
    app.geometry("400x400")
    


    
    canvas = Canvas(app, bg='black')
    canvas.pack(anchor='nw', fill='both', expand=1)

    canvas.bind("<Button-1>", get_x_and_y)
    canvas.bind("<B1-Motion>", draw_smth)
    
    for i in range(len(list_url)):
        loaded_image["img" + str(i)] =  PhotoImage(file=list_url[i])
    print(loaded_image)
    image_container = canvas.create_image(0,0,image = loaded_image["img0"], anchor="nw")
    
    left_button = Button(app, text ="<<<", command = go_left)
    right_button = Button(app, text =">>>", command = go_right)
    mode_button = Button(app, text="Mode", command=change_mode)
    done_button = Button(app, text ="Done", command = done)
    
    left_button.pack()
    right_button.pack()
    mode_button.pack()
    done_button.pack()
    
    app.mainloop()

li_url = ["front1.png"]
interactive_segmentation(li_url)