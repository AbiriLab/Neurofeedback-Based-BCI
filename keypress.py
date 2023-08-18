import tkinter as tk

class KeyPressDetector:
    def __init__(self, master):
        self.key_pressed = 0  # Initial state
        master.bind('<Key>', self.on_key_press)

    def on_key_press(self, event=None):
        self.key_pressed = 1

    def check_key_press(self):
        result = self.key_pressed
        self.key_pressed = 0  # Reset after checking
        return result

# # Example usage:
# root = tk.Tk()

# detector = KeyPressDetector(root)

# def check():
#     print(detector.check_key_press())  # This will print 1 if any key was pressed since the last check, and 0 otherwise
#     # root.after(4, check)  # Check every second

# check()
# root.mainloop()
