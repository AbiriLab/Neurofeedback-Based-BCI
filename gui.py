import time
from tkinter import *
from PIL import Image, ImageTk
import customtkinter
from customtkinter import *
import turtle
from turtle import *


customtkinter.set_appearance_mode("system")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


class RootWindow():

    def __init__(self, master):

        # self.patientWindow = None


        # left frame
        self.frame_left = customtkinter.CTkFrame(master)

        button1 = customtkinter.CTkButton(self.frame_left, text="Open Patient Window", command=self.open)
        button1.grid(row=0, column=0, columnspan=2, padx=10, pady=20)

        user_label = customtkinter.CTkLabel(self.frame_left, text='User Number:')
        user_label.grid(row=1, column=0,)
        self.user_entry_var = customtkinter.StringVar()
        user_entry = customtkinter.CTkEntry(self.frame_left, textvariable=self.user_entry_var)
        user_entry.grid(row=1, column=1, padx=10)

        trial_label = customtkinter.CTkLabel(self.frame_left, text='Trial Number:')
        trial_label.grid(row=2, column=0)
        self.trial_entry_var = customtkinter.StringVar()
        trial_entry = customtkinter.CTkEntry(self.frame_left, textvariable=self.trial_entry_var)
        trial_entry.grid(row=2, column=1, padx=10)

        button1 = customtkinter.CTkButton(self.frame_left, text="Submit User/Trial Data", command=self.update_user_data)
        button1.grid(row=3, column=0, columnspan=2, padx=10, pady=20)

        collect_data_box = customtkinter.CTkCheckBox(self.frame_left, text="Record EMG Data", command=self.record_data)
        collect_data_box.grid(row=4, column=0, columnspan=2, padx=50, pady=10)

        self.run_trial_but = customtkinter.CTkButton(self.frame_left, text="Run Trial", )
        self.run_trial_but.grid(row=5, column=0, columnspan=2, padx=10, pady=20)

        self.frame_left.grid(padx=40, pady=50, row=0, column=0)

        # right frame
        self.frame_right = customtkinter.CTkFrame(master)

        stop_but = customtkinter.CTkButton(self.frame_right, text="Stop Trial", command=self.stop_trial)
        stop_but.grid(padx=50, pady=20, row=4, column=0)



        self.frame_right.grid(padx=20, pady=50, row=0, column=1)

    def open(self):
        self.patientWindow = PatientWindow()

    def update_user_data(self):
        # update csv file with user number and trial num
        # uses .get() with entry text variables
        trial_data = [self.user_entry_var.get(), self.trial_entry_var.get()]
        with open('exo_data.csv','w') as file:
            writer = csv.writer(file)
            writer.writerow(['User Number', 'Trial Number'])

            writer.writerow(trial_data)
        print("User and Trial Data Collected")

    def record_data(self):
        # update bool value to update "data collection" function that will take some type of data (eeg,emg)
        pass



    def stop_trial(self):
        self.patientWindow.destroy_patient_window()


class PatientWindow():
    def __init__(self):

        self.top = customtkinter.CTkToplevel()
        self.top.geometry("600x850")
        self.top.title("Exoskeleton Patient View")
        self.frame = customtkinter.CTkFrame(self.top)

        im = Image.open("images/please-wait2.png")
        resized = im.resize((600, 600), Image.Resampling.LANCZOS)
        ph = ImageTk.PhotoImage(resized)

        self.label = Label(self.frame, image=ph, width=600, height=600)
        self.label.image = ph  # need to keep the reference of your image to avoid garbage collection
        self.label.grid(row=0, column=0)
        self.frame.grid(row=0, column=0)

        w = 600
        h = 250

        self.frame_bottom = customtkinter.CTkFrame(self.top)
        self.canvas = Canvas(self.frame_bottom, width=w, height=h, bg='gray')
        self.canvas.grid(row=0, column=0)

        x = w // 2
        y = h // 2
        cursor = self.canvas.create_oval(x, y, x+10, y+10, fill='red')

        # left_wall = self.canvas.create_rectangle(125, 90, 100, 165, fill='pink')

        self.frame_bottom.grid(row=1, column=0)




    def stop_image(self):
        stop_img = Image.open("images/stop.gif")
        stop_img_resized = stop_img.resize((600, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(stop_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img

    def right_arrow(self):
        right_img = Image.open("images/right-arrow.gif")
        right_img_resized = right_img.resize((600, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(right_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img

    def left_arrow(self):
        left_img = Image.open("images/left-arrow.gif")
        left_img_resized = left_img.resize((600, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(left_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img

    def please_wait(self):
        wait_img = Image.open("images/please-wait.png")
        wait_img_resized = wait_img.resize((600, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(wait_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img

    def good_job_img(self):
        good_img = Image.open("images/good_job.jpg")
        good_img_resized = good_img.resize((600, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(good_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img

    def destroy_patient_window(self):
        self.top.destroy()
        self.top.update()


if __name__ == "__main__":
    root = CTk()
    root.geometry("690x500")
    root.title("Controls")
    main_window = RootWindow(root)
    root.mainloop()