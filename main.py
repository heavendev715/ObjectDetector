import tkinter as tk
from tkinter import messagebox
import cv2
from gui_buttons import Buttons
from PIL import ImageTk, Image
import pyttsx3

# Initialize buttons
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("bottle", 20, 180)
button.add_button("chair", 20, 260)
button.add_button("spoon", 20, 340)
colors = button.colors

# opencv DNN
net = cv2.dnn.readNet("yolov8-ultralytics.weights","yolov8-ultralytics.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# load class list
classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print("Object list")
print(classes)

# initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
button_person = False

# create window
window = tk.Tk()
window.title("Object Detector")
window.geometry("800x600")

# add title label
title_label = tk.Label(window, text="Object Detector", font=("Helvetica", 24, "bold"), bg="#323232", fg="white", padx=10, pady=10)
title_label.pack(side="top", fill="x")

# add description label
desc_label = tk.Label(window, text="This is a simple object detector using OpenCV and\n YOLOv4-tiny model. The program allows you to detect objects\n in real-time video from your camera using a pre-trained model.\n Simply click on the buttons to toggle the detection of specific objects.", font=("Helvetica", 14), padx=20, pady=20)
desc_label.pack()

# create background image
bg_image = Image.open("download2.jpg")
bg_image = bg_image.resize((800, 600), Image.LANCZOS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

# add background label
bg_label = tk.Label(window, image=bg_image_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# add title label
title_label = tk.Label(window, text="Object Detector", font=("Helvetica", 24, "bold"), bg="#323232", fg="white", padx=10, pady=10)
title_label.pack(side="top", fill="x")

# add description label
desc_label = tk.Label(window, text="This is a simple object detector using OpenCV and\n YOLOv8-ultralytics model. The program allows you to detect objects\n in real-time video from your camera using a pre-trained model.\n Simply click on the buttons to toggle the detection of specific objects.", font=("Helvetica", 14), padx=20, pady=20)
desc_label.pack()

# text to speech
engine = pyttsx3.init()

# add start button
def start_detection():
    messagebox.showinfo("Starting Detection", "Click OK to start object detection.")

    def click_button(event, x, y, flags, params):
        global button_person
        if event == cv2.EVENT_LBUTTONDOWN:
            button.button_click(x, y)

    # create window
    cv2.namedWindow("Object Detector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Object Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Object Detector", click_button)

    while True:
        # getting frames
        ret, frame = cap.read()
        # zoom out
        frame = cv2.resize(frame, (0, 0), fx=2.0, fy=2.0)
        # get active buttons list
        active_buttons = button.active_buttons_list()
        print("Active buttons", active_buttons)
        # object detection
        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]
            color = colors[class_id]
            if class_name in active_buttons:
                cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 3)
                # add text displaying the coordinates of the top-left corner of the bounding box
                cv2.putText(frame, f"({x}, {y})", (x, y - 40), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                # Speak aloud the detected object
                engine.say(f"I see a {class_name}")
                engine.runAndWait()
        # display buttons
        button.display_buttons(frame)
        cv2.imshow("Object Detector", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

start_button = tk.Button(window, text="Start Detection", command=start_detection, bg="green", fg="white", font=("Arial", 16), padx=10, pady=10)
start_button.pack(pady=30)

window.mainloop()
