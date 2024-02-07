from tkinter import ttk
from tkinter import *
import csv
import pandas as pd
import numpy as np
import pygds as g
from image_display_gt import *
import time

#main GUI window
class RootWindow:

    def __init__(self, master):
        self.SamplingRate = 256
        self.SerialNumber = 'UR-2020.08.14'
        self.eeg_data = []
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
        self.create_gui(master)

    def create_gui(self, master):
        self.frame_1 = tk.Frame(master)

        self.patient_name_label = tk.Label(self.frame_1, text="Patient Name:", font=15)
        self.patient_name_label.grid(row=1, column=0, pady=15, padx=5)
        self.patient_name_data = tk.StringVar()
        self.patient_name_entry = tk.Entry(self.frame_1, width=30, font=15)
        self.patient_name_entry.grid(row=1, column=1, pady=15, padx=5)

        self.phase_label = tk.Label(self.frame_1, text="Current Phase:", font=15)
        self.phase_label.grid(row=4, column=0, pady=15, padx=5)
        self.curr_phase = tk.StringVar()
        self.curr_phase_num = tk.Label(self.frame_1, textvariable=self.curr_phase, font=15)
        self.curr_phase_num.grid(row=4, column=1, pady=15, padx=5)
        plist = ["Pre-Evaluation", "Neurofeedback", "Post-Evaluation"]
        self.phase_box = ttk.Combobox(self.frame_1, values=plist, state='readonly', font=15, textvariable=self.curr_phase)
        self.phase_box.set("Select the Phase")
        self.phase_box.grid(row=3, column=1, pady=15, padx=5)

        self.block_label = tk.Label(self.frame_1, text="Last Block\n Completed:", font=10)
        self.block_label.grid(row=5, column=0, pady=15, padx=5)
        self.curr_block = StringVar()
        self.block_num = tk.Label(self.frame_1, textvariable=self.curr_block, font=15)
        self.block_num.grid(row=5, column=1, columnspan=2, pady=15, padx=5)

        self.phase_prog_lab = tk.Label(self.frame_1, text="Phase Progress:", font=15)
        self.phase_prog_lab.grid(row=6, column=0, pady=15, padx=5)
        self.progress = tk.IntVar()
        self.phase_prog = ttk.Progressbar(self.frame_1, variable=self.progress, length=250)
        self.progress.set(0)
        self.phase_prog.grid(row=6, column=1, pady=15, padx=5)

        self.frame_1.grid(padx=30, pady=50, row=0, column=0)

        self.frame_2 = tk.Frame(master)

        self.create_trial_but = Button(self.frame_2, text="Create Trial", bg="green", font=15, command=self.create_trial)
        self.create_trial_but.grid(row=1, column=0, columnspan=2, pady=15, padx=5)

        self.start_trial_but = Button(self.frame_2, text="Start Trial", bg="green", font=15, command=self.start_trial)
        self.start_trial_but.grid(row=4, column=0, pady=15, padx=5)

        self.end_trial_but = Button(self.frame_2, text="End Trial", bg="red", font=15, command=self.end_trial)
        self.end_trial_but.grid(row=4, column=1, pady=15, padx=5)

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
        global h
        h.Configure(self.SerialNumber,self.SamplingRate, 0)
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
                self.open_image_win()
                return

        self.pre_eval, self.neuro, self.post_eval = 0, 0, 0
        random.shuffle(self.r)
        randomized_blocks = self.r
        self.seq = " ".join(str(x) for x in randomized_blocks)
        self.patient_progress[4] = self.seq
        self.patient_data_list.append(self.patient_progress)
        self.patient_index = len(self.patient_data_list) - 1
        self.add_patient_data()
        print(f"New Data Added: {self.patient_progress}")
        self.open_image_win()
        # self.update_gui()

    def add_patient_data(self):
        with open('pat_progess_v2.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.patient_data_list:
                writer.writerow(row)
        print(f"Full Data added to CSV: {self.patient_data_list}")

    def open_image_win(self):
        global image_window, top
        top = Toplevel()
        top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        top.title("Image Slideshow")
        # print(list(self.seq.replace(" ", "")))
        image_window = DisplayImage(top, self.block, list(self.seq.replace(" ", "")))
        image_window.single_block = False
        image_window.create_img_arr()
        image_window.pleaseWait_image()
        self.image_window_open = True
        print("Image Window Opened")
        top.update()

    def start_trial(self):
        global h, image_window, data, top
        data = []
        image_window.instructions_image()
        top.update()
        time.sleep(5)
        h.getData(self.SamplingRate)
        
        #saves the EEG file below as csv, file name will be what's in the open()
        with open('eeg_data_'+str(image_window.curr_block)+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            key = pd.read_csv('Images/Composite_Images_key/Block'+self.patient_progress[4][(image_window.curr_block-1)*2]+'_key.csv').to_numpy()
            for i in range(len(data)):
                writer.writerow(np.hstack((data[i],key[int((i)/self.SamplingRate)])))                                     
        del data, key
        print('Data saved for block ' + str(image_window.curr_block))


    def end_trial(self):
        global h, image_window
        self.add_patient_data()
        image_window.close_window()
        h.d.Close()
        del h.d
        print('Disconnected')

    def update_gui(self):
        self.block += 1
        self.curr_block.set(str(self.block))
        self.progress.set(round(self.block * 12.5))

    def start_block(self):
        global image_window
        # config block or count then run next image function
        block_to_run = self.single_block_num_var.get()
        if self.image_window_open:
            image_window.close_window()

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

####################################################################################################################################################################################

class EEG_Headset:
    def __init__(self):
        self.samples = []
        self.count = 0

    def Configure(self,SERIALNUMBER, SAMPLINGRATE, NUMBEROFSCANS):
        self.SamplingRate = SAMPLINGRATE
        self.SerialNumber = SERIALNUMBER
        print('Connecting to device ...   ')
        try:
            a=g.ConnectedDevices
            dev_names = [n for n, t, u in g.ConnectedDevices() if t == 1]
            devices = ','.join(dev_names)
            #devices = dev_names[1] + ',' + dev_names[0]
            print('master,slave = ', devices)
            self.d=g.GDS(devices)
            hb=[]
            if len(dev_names) == 2:
                trig = 0
                for c in self.d.Configs:
                    c.SamplingRate = self.SamplingRate
                    c.NumberOfScans = 1
                    c.CommonGround = [1] * 4
                    c.CommonReference = [1] * 4
                    c.ShortCutEnabled =1
                    c.CounterEnabled = 0
                    if trig == 1:
                        c.TriggerEnabled = 1
                        
                    acquireHelp = 0
                    for ch in c.Channels:
                        ch.Acquire = 1 # if acquireHelp == 0 else 0
                        acquireHelp = 1
                        # do not use filters
                        ch.BandpassFilterIndex = 45
                        ch.NotchFilterIndex = 2
            #         # do not use a bipolar channel
                        ch.BipolarChannel = 0  
                        trig = 1
            else:
                self.d.SamplingRate = self.SamplingRate
                self.d.NumberOfScans = 1
                self.d.CommonGround = [1] * 4
                self.d.CommonReference = [1] * 4
                self.d.ShortCutEnabled = 1
                self.d.CounterEnabled = 0
                self.d.TriggerEnabled = 1
                for ch in self.d.Channels:
                    ch.Acquire = 1 # if acquireHelp == 0 else 0
                    # do not use filters
                    ch.BandpassFilterIndex = 45
                    ch.NotchFilterIndex = 2
        #         # do not use a bipolar channel
                    ch.BipolarChannel = 0
            self.d.SetConfiguration()
            print('Connected')
        except:
            print('Could not connect')

    def getData(self, BLOCKSIZE):
        print('Collecting data')
        # if hasattr(self, 'd'):
        temp = self.d.GetData(BLOCKSIZE, GetDataCallback)
        # else:
        #     print('Device not configured or failed to connect')
def GetDataCallback(dataBlock):
    global h, image_window, top, data
    try:
        if len(data) == 0:
            data = np.vstack(dataBlock.copy())
        else:
            data = np.vstack((data, dataBlock.copy()))
        print(h.count)
    except:
        print('Failed to collect data')
    if h.count < image_window.curr_block*40:
        h.count = h.count + 1
        image_window.next_image()
        top.update()
        return True
    else:
        image_window.pleaseWait_image()
        top.update()
        print('Done')
        return False
    

if __name__ == "__main__":
    global h
    h = EEG_Headset()
    root = Tk()
    root.geometry("%dx%d+%d+%d" % (1000, 500, 100, 100))
    root.title("Root Window Controls")
    main_window = RootWindow(root)
    root.mainloop()
