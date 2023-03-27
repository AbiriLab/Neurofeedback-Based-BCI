from tkinter import ttk
from tkinter import *
import csv
import pandas as pd
import numpy as np
from image_display_v2 import *
import threading


class RootWindow:

    def __init__(self, master):
        self.image_window = None
        self.block = 0
        self.image_window_open = False
        self.patient_progress = ['', '0', '0', '0', '00000000']
        self.patient_index = 0
        self.patient_data_list = []
        self.pre_eval = 0
        self.neuro = 0
        self.post_eval = 0
        self.seq = None
        self.top = None
        self.r = [1, 2, 3, 4, 5, 6, 7, 8]
        with open('pat_progess_v2.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.patient_data_list.append(row)
        print(self.patient_data_list)
        df = pd.read_csv('pat_progess_v2.csv')
        print(df)

        self.frame_1 = tk.Frame(master)

        self.eeg_label = tk.Label(self.frame_1, text="Recording Device:", font=15)
        self.eeg_label.grid(row=1, column=0, pady=15, padx=5)
        eeg_list = ["Unicorn EEG", "gtec EEG"]
        self.curr_eeg = StringVar()
        self.phase_box = ttk.Combobox(self.frame_1, values=eeg_list, state='readonly', font=15,
                                      textvariable=self.curr_eeg)
        self.phase_box.set("EEG Recording Device")
        self.phase_box.grid(row=1, column=1, pady=15, padx=5)

        self.patient_name_label = tk.Label(self.frame_1, text="Patient Data:", font=15)
        self.patient_name_label.grid(row=2, column=0, pady=15, padx=5)
        self.patient_name_data = tk.StringVar()
        self.patient_name_entry = tk.Entry(self.frame_1, width=30, font=15)
        self.patient_name_entry.grid(row=2, column=1, pady=15, padx=5)

        self.phase_label = tk.Label(self.frame_1, text="Current Phase:", font=15)
        self.phase_label.grid(row=5, column=0, pady=15, padx=5)
        self.curr_phase = tk.StringVar()
        self.curr_phase_num = tk.Label(self.frame_1, textvariable=self.curr_phase, font=15)
        self.curr_phase_num.grid(row=5, column=1, pady=15, padx=5)
        plist = ["Pre-Evaluation", "Neurofeedback", "Post-Evaluation"]
        self.phase_box = ttk.Combobox(self.frame_1, values=plist, state='readonly', font=15, textvariable=self.curr_phase)
        self.phase_box.set("Select the Phase")
        self.phase_box.grid(row=4, column=1, pady=15, padx=5)

        self.block_label = tk.Label(self.frame_1, text="Current Block", font=10)
        self.block_label.grid(row=6, column=0, pady=15, padx=5)
        self.curr_block = StringVar()
        self.block_num = tk.Label(self.frame_1, textvariable=self.curr_block, font=15)
        self.block_num.grid(row=6, column=1, columnspan=2, pady=15, padx=5)

        self.phase_prog_lab = tk.Label(self.frame_1, text="Phase Progress:", font=15)
        self.phase_prog_lab.grid(row=7, column=0, pady=15, padx=5)
        self.progress = tk.IntVar()
        self.phase_prog = ttk.Progressbar(self.frame_1, variable=self.progress, length=250)
        self.progress.set(0)
        self.phase_prog.grid(row=7, column=1, pady=15, padx=5)

        self.frame_1.grid(padx=30, pady=50, row=0, column=0)

        self.frame_2 = tk.Frame(master)

        self.create_trial_but = Button(self.frame_2, text="Create Trial", bg="green", font=15, command=self.create_trial)
        self.create_trial_but.grid(row=1, column=0, columnspan=2, pady=15, padx=5)

        self.add_pat_data_but = tk.Button(self.frame_2, text="Add Patient Data", bg='light blue', font=15,
                                          command=self.add_patient_data)
        self.add_pat_data_but.grid(row=2, column=0, columnspan=2, pady=15, padx=5)

        self.open_img_win_but = Button(self.frame_2, text="Open Image Window", bg='light blue', font=15,
                                       command=self.open_image_win)
        self.open_img_win_but.grid(row=3, column=0, columnspan=2, pady=15, padx=5)

        self.start_trial_but = Button(self.frame_2, text="Start Trial", bg="green", font=15, command=self.start_trial)
        self.start_trial_but.grid(row=4, column=0, pady=15, padx=5)

        self.end_trial_but = Button(self.frame_2, text="End Trial", bg="red", font=15, command=self.end_trial)
        self.end_trial_but.grid(row=4, column=1, pady=15, padx=5)

        # self.update_trial_but = Button(self.frame_2, text="Update Trial", bg="light green", font=15, command=self.update_patient_data)
        # self.update_trial_but.grid(row=5, column=0, columnspan=2, pady=15, padx=5)

        self.frame_2.grid(padx=30, pady=50, row=0, column=1)

        self.frame_3 = tk.Frame(master)

        self.single_block = tk.Label(self.frame_3, text="Single Block", font=8)
        self.single_block.grid(row=2, column=0, pady=5, padx=3)
        blist = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.single_block_num_var = IntVar()
        self.block_box = ttk.Combobox(self.frame_3, values=blist, state='readonly', font=10,
                                      textvariable=self.single_block_num_var, width=12)
        self.block_box.set("Block to Run")
        self.block_box.grid(row=2, column=1, pady=5, padx=3)

        self.start_block_but = Button(self.frame_3, text="Start Block", bg="light green", font=8, command=self.start_block)
        self.start_block_but.grid(row=3, column=0, pady=5, padx=3)

        self.end_block_but = Button(self.frame_3, text="End Block", bg='#FB586C', font=8, command=self.end_block)
        self.end_block_but.grid(row=3, column=1, pady=5, padx=3)

        self.frame_3.grid(padx=30, pady=50, row=0, column=2)

    def create_trial(self):
        patient_name = self.patient_name_entry.get()
        self.patient_progress[0] = patient_name

        for data_list in self.patient_data_list:
            if patient_name == data_list[0]:
                self.patient_index = self.patient_data_list.index(data_list)
                self.patient_progress = data_list
                self.pre_eval = int(self.patient_progress[1])
                self.neuro = int(self.patient_progress[2])
                self.post_eval = int(self.patient_progress[3])
                self.seq = self.patient_progress[4]
                # self.patient_data_list.append(self.patient_progress)
                print(f"Data Already Exists: {self.patient_progress}")
                if self.curr_phase.get() == 'Pre-Evaluation':
                    self.block = self.pre_eval
                elif self.curr_phase.get() == "Neurofeedback":
                    self.block = self.neuro
                elif self.curr_phase.get() == "Post-Evaluation":
                    self.block = self.post_eval
                self.curr_block.set(str(self.block))
                self.progress.set(round(self.block * 12.5))

                return

        self.pre_eval, self.neuro, self.post_eval = 0, 0, 0
        random.shuffle(self.r)
        randomized_blocks = self.r
        self.seq = " ".join(str(x) for x in randomized_blocks)
        self.patient_progress[4] = self.seq
        self.patient_data_list.append(self.patient_progress)
        self.patient_index = len(self.patient_data_list) - 1
        print(f"New data created, please click 'Add Data'")
        # self.update_gui()

    def add_patient_data(self):
        with open('pat_progess_v2.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.patient_data_list:
                writer.writerow(row)
        print(f"New Data Added: {self.patient_progress}")

    def open_image_win(self):
        self.top = Toplevel()
        self.top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        self.top.title("Image Slideshow")
        # print(list(self.seq.replace(" ", "")))
        self.image_window = DisplayImage(self.top, self.block, list(self.seq.replace(" ", "")))
        self.image_window.single_block = False
        self.image_window.create_img_arr()
        self.image_window.next_image()
        self.image_window_open = True
        print("Image Window Opened")
        self.top.mainloop()

    def start_trial(self):
        if self.image_window_open:

            self.image_window.pause = False
            self.image_window.instructions_image()
            # self.image_window.pause = False
            # self.image_window.next_image()
        else:
            self.add_patient_data()
            self.open_image_win()
        if self.curr_phase.get() == 'Pre-Evaluation':
            if self.block > 8:  # may need to change to curr_block
                self.block = 1
                self.curr_block.set(str(self.block))
            self.pre_eval = self.block
            self.patient_progress[1] = str(self.block)
        elif self.curr_phase.get() == 'Post-Evaluation':
            if self.block > 8:
                self.block = 1
                self.curr_block.set(str(self.block))
            self.post_eval = self.block
            self.patient_progress[3] = str(self.block)
        self.update_gui()
        # could call function here to show instructions since trial order created before this so could auto choose right image for 5 secs then switch to normal trial photos
        # self.image_window.next_image()
        self.update_patient_data()

    def end_trial(self):
        self.update_patient_data()
        self.image_window.close_window()

    def update_gui(self):
        self.block += 1
        self.curr_block.set(str(self.block))
        self.progress.set(round(self.block * 12.5))

    def update_patient_data(self):

        self.add_patient_data()
        # self.patient_progress[4] = self.seq
        # self.patient_data_list[self.patient_index] = self.patient_progress
        # with open('NF_Patient_Progress.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for row in self.patient_data_list:
        #         writer.writerow(row)
        #
        # # self.update_gui()
        # print(f'Patient Data Updated post trial run: {self.patient_progress}')

    def start_block(self):
        # config block or count then run next image function
        block_to_run = self.single_block_num_var.get()
        if self.image_window_open:
            self.image_window.close_window()

        self.top = Toplevel()
        self.top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        self.top.title("Image Slideshow")
        self.image_window = DisplayImage(self.top, self.block, block_to_run)
        self.image_window.single_block = True
        self.image_window.create_img_arr()
        self.image_window.pause = False
        self.image_window.next_image()
        self.image_window_open = True
        print("Image Window Opened")
        self.top.mainloop()

        self.curr_block.set(str(self.block))

    def end_block(self):
        self.image_window.close_window()


if __name__ == "__main__":
    root = Tk()
    root.geometry("%dx%d+%d+%d" % (1000, 500, 100, 100))
    root.title("Root Window Controls")
    main_window = RootWindow(root)
    root.mainloop()
