import cv2
import numpy as np
import math
import time
from threading import Thread
from PyQt5 import QtWidgets,QtCore,QtGui, uic
time_count=0
time_minute=0
START_POINT = 150
END_POINT =40
tb=[]
cost_time=20
number_vehicle = 0
Auutien=[0,0,0]
CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
           "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
           "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
           "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# Define vehicle class
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

# get it at https://pjreddie.com/darknet/yolo/
YOLOV3_CFG = "C:/Users/thi/Documents/yolov3-tiny.cfg.txt"
YOLOV3_WEIGHT = "C:/Users/thi/Documents/yolov3-tiny.weights"
# YOLOV3_CFG = "C:/Users/thi/Downloads/yolov3-spp.cfg.txt"
# YOLOV3_WEIGHT = "C:/Users/thi/Downloads/yolov3-spp.weights"
CONFIDENCE_SETTING = 0.4
YOLOV3_WIDTH = 416
YOLOV3_HEIGHT = 416
Xanh1=30
MAX_DISTANCE = 20
exitFlag = 0
'''các thông số giao diện để điều chỉnh  
1. thuật toán yolo
2.start point
3.end point
4.skip frame
5.thông số điều chỉnh tính chính xác của việc chọn
6. time hiển thị
7. hệ số điều chỉnh tg đếm cost_time
'''
def count_():
    # chossetime()
    global time_count,time_minute,tb,Xanh1,number_vehicle,Auutien,cost_time
    
    # time.sleep(0.001)#1ms
    time_minute += 1
    # print(time_minute)
    if time_minute ==cost_time:
        time_count += 1
        tb.append(time_count)
        print (len(tb))
        time_minute=0
    if len(tb)==Xanh1-10: # xác đinh trước tầm 10s tt tiếp theo
        trungbinh=0
        for i in (tb):
            trungbinh=trungbinh+i
        trungbinh=trungbinh/len(tb)
        if number_vehicle >Xanh1-15:#đường đông (-15) tham số điều chỉnh
            Auutien=[1,0,0]
            print("uu tien 1 ")
            tb.clear()
            number_vehicle=0
        elif 5<trungbinh<10 and Xanh1/2<number_vehicle<Xanh1-15:# đường bình thường
            Auutien=[0,1,0]
            print("uu tien 2 ")
            tb.clear()
            number_vehicle=0
        else:   #đường vắng
            Auutien=[0,0,1]
            print("uu tien 3 ")
            tb.clear()
            number_vehicle=0

# def chossetime():
#     global Auutien,Buutien
#     if Auutien[0]==1 and Buutien[0]==1:
#         Xanh1=60
#         Do1=60
#         Auutien[0]==0
#         Buutien[0]==0
#     elif Auutien[0]==1 and Buutien[1]==1:
#         Xanh1=70
#         Do1=40
#         Auutien[0]==0
#         Buutien[1]==0
#     elif Auutien[0]==1 and Buutien[2]==1:
#         Xanh1=90
#         Do1=30
#         Auutien[0]==0
#         Buutien[2]==0
#     elif Auutien[1]==1 and Buutien[0]==1:
#         Xanh1=40
#         Do1=70
#         Auutien[1]==0
#         Buutien[0]==0
#     elif Auutien[1]==1 and Buutien[1]==1:
#         Xanh1=45
#         Do1=45
#         Auutien[1]==0
#         Buutien[1]==0
#     elif Auutien[1]==1 and Buutien[2]==1:
#         Xanh1=50
#         Do1=35
#         Auutien[1]==0
#         Buutien[2]==0
#     elif Auutien[2]==1 and Buutien[0]==1:
#         Xanh1=30
#         Do1=90
#         Auutien[2]==0
#         Buutien[0]==0
#     elif Auutien[2]==1 and Buutien[1]==1:
#         Xanh1=35
#         Do1=50
#         Auutien[2]==0
#         Buutien[1]==0
#     elif Auutien[2]==1 and Buutien[2]==1:
#         Xanh1=30
#         Do1=30
#         Auutien[2]==0
#         Buutien[2]==0






   
def get_output_layers(net):
    """
    Get output layers of darknet
    :param net: Model
    :return: output_layers
    """
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    except:
        print("Can't get output layers")
        return None

# hàm dùng để nhận dạng
def detections_yolo3(net, image, confidence_setting, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    """
    Detect object use yolo3 model
    :param net: model
    :param image: image
    :param confidence_setting: confidence setting
    :param yolo_w: dimension of yolo input
    :param yolo_h: dimension of yolo input
    :param frame_w: actual dimension of frame
    :param frame_h: actual dimension of frame
    :param classes: name of object
    :return:
    boxes: toa độ
    class_id: loại phương tiện
    confidences: tỷ lệ phán đoán 
    """
    img = cv2.resize(image, (yolo_w, yolo_h))
    blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
    net.setInput(blob)
    layer_output = net.forward(get_output_layers(net))
    # print("layers",layer_output)
    boxes = []
    class_ids = []
    confidences = []

    for out in layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_setting and class_id in VEHICLE_CLASSES:
                # print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                center_x = int(detection[0] * frame_w)
                center_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, class_ids, confidences

# hàm vẽ các đường nhận dạng được và label chúng
def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height):
    """
    Draw bounding box and put classe text and confidence
    :param classes: name of object
    :param colors: color for object
    :param img: immage
    :param class_id: class_id of this object
    :param confidence: confidence
    :param x: top, left
    :param y: top, left
    :param width: width of bounding box
    :param height: height of bounding box
    :return: None
    """
    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))

# kiểm tra phương tiện đi qua line
def check_location(box_y, box_height, height):
    """
    Check center point of object that passing end line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :param height: height of image
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > height - END_POINT:
        return True
    else:
        return False

#kiểm tra phương tiện vào line 
def check_start_line(box_y, box_height):
    """
    Check center point of object that passing start line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > START_POINT:
        return True
    else:
        return False

#đếm xe
def counting_vehicle(video_input, video_output, skip_frame=1):
    global END_POINT, time_count,number_vehicle
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)

    # Read first frame
    cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]
    
    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    
    tmp_vehicle=0
    while cap.isOpened():
        count_()
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
           

            if check:
                box_x, box_y, box_width, box_height = box # lấy các giá trị của mảng box và dán lần lượt vào các biến

                draw_prediction(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height):
                    number_vehicle += 1
                    # This object passed the end line
                    time_count=0
                        
                else:

				    # Increment the minute total
                    # thread1 = Thread(count_())
                    # thread1.start()
                    list_object.append(obj)

        
       # điều khiển các khung fame kt, skip_frame càng lớn, tốc độ chạy càng nhanh độ chính xác càng giảm 
        if number_frame % skip_frame == 0:
            # Detect object and check new object
            boxes, class_ids, confidences = detections_yolo3(net, frame, CONFIDENCE_SETTING, YOLOV3_WIDTH,
                                                             YOLOV3_HEIGHT, width, height, classes=CLASSES)



            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height):
                        # Append new object to list
                        new_tracker = cv2.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }
                        list_object.append(new_object)
                        # Draw new object

                        draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)
        # Put summary text
        cv2.putText(frame, "Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw start line
        #cv2.line(hình,(tọa độ x,y điểm 1),(tọa độ x,y điểm cuối),màu,độ dày)
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (204, 90, 208), 3)
        # Draw end line
        cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (255, 0, 0), 2)
        # Show frame
        cv2.imshow("CAMERA", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    counting_vehicle("C:/Users/thi/Pictures/video_traffic/CSGT hộ tống Đoàn Thủ Tướng từ miền tây ra sân bay Tân Sơn Nhất_Trim.mp4", 'vehicles.avi')
    # counting_vehicle("C:/Users/thi/Pictures/video_traffic/Racing boy bị CSGT nện một gậy đau điếng phải bỏ chạy, vì cố xông vào đoàn xe VIP Chủ tịch Quốc hội.mp4", 'vehicles.avi')
    
   