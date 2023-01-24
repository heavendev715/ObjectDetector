import cv2
from gui_buttons import Buttons

#Initialize buttons

button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("bottle", 20, 180)

colors = button.colors


#opencv DNN

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

#load class list
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print("Object list")
print(classes)

#initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

button_person = False

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)


# create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    #getting frames
    ret, frame = cap.read()

    #get active buttons list
    active_buttons = button.active_buttons_list()
    print("Active buttons", active_buttons)

    # object detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold= 0.3, nmsThreshold = 0.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 3)


    #display buttons
    button.display_buttons(frame)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

