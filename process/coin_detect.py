import numpy as np
import cv2

import cv2
import numpy as np
import sys

def detect_onnx(img):
    COIN_DIAMETER = 2.7

    def build_model(is_cuda):
        net = cv2.dnn.readNet("pengukuran/best.onnx")
        if is_cuda:
            print("Attempting to use CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    SCORE_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.4
    CONFIDENCE_THRESHOLD = 0.4

    def detect(image, net):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds

    class_list = ["circle"]

    def wrap_detection(input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= CONFIDENCE_THRESHOLD:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if classes_scores[class_id] > SCORE_THRESHOLD:

                    confidences.append(confidence)
                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, NMS_THRESHOLD)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def format_yolov5(frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    net = build_model(is_cuda)

    frame = cv2.imread(img)

    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    if len(confidences) > 0:
        max_confidence_index = np.argmax(confidences)
        highest_confidence_box = boxes[max_confidence_index]
        width_of_highest_confidence = highest_confidence_box[2]
        print(COIN_DIAMETER/width_of_highest_confidence)
        return COIN_DIAMETER/width_of_highest_confidence
    else:
        return None
# def detect_yolo(img_path):
#     COIN_DIAMETER = 2.7  # cm

#     # Load model
#     model = torch.hub.load('pengukuran', 'custom', path='model/best.pt', source='local', force_reload=True)

#     # Inference
#     results = model(img_path)

#     # Convert results to pandas DataFrame
#     df = results.pandas().xyxy[0]
#     df = df[df['confidence'] > 0.6]

#     # Calculate additional properties
#     df['x'] = df['xmax'] - df['xmin']
#     df['y'] = df['ymax'] - df['ymin']
#     df['x_tengah'] = (df['xmin'] + df['xmax']) / 2
#     df['y_tengah'] = (df['ymin'] + df['ymax']) / 2

#     # Sort and reset index
#     df = df.sort_values('x')
#     df = df.reset_index()

#     # Get coordinates of the first detected object
#     x = df['x_tengah'][0]
#     y = df['y_tengah'][0]
#     xmin = df['xmin'][0]
#     ymin = df['ymin'][0]
#     xmax = df['xmax'][0]
#     ymax = df['ymax'][0]

#     # Calculate width and height
#     width = xmax - xmin
#     height = ymax - ymin

#     # Calculate coefficient
#     if width > height:
#         width = height
#     coefficient = COIN_DIAMETER / width

#     # Read image
#     # img = cv2.imread(img_path)

#     # Draw bounding box on the image
#     # cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

#     # # Display the image
#     # cv2.imshow('image', img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return coefficient

#print(detect('images/baby_up.jpeg'))

#coin detect using opencv Houghcircles
def detect_cv(image_path):
    COIN_DIAMETER = 2.7 #cm
    # Baca gambar
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Konversi gambar ke skala abu-abu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur gambar untuk mengurangi noise
    gray_blurred = cv2.medianBlur(gray, 5)

    # Deteksi lingkaran menggunakan HoughCircles

    circles = cv2.HoughCircles(gray_blurred,
               cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
           param2 = 30, minRadius = 1, maxRadius = 40)

    # Jika lingkaran terdeteksi
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Urutkan lingkaran berdasarkan radius (semakin besar, semakin "akurat")
        best_circle = max(circles, key=lambda c: c[2])
        x, y, r = best_circle

        # Gambar lingkaran dan pusatnya
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # Konversi BGR ke RGB untuk plt.imshow
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Tampilkan gambar dengan lingkaran yang terdeteksi
        #plt.figure(figsize=(8, 8))
        #plt.imshow(img_rgb)
        #plt.axis("on")
        #plt.show()

        #coefficient coin
        coefficient = COIN_DIAMETER / (r * 2.0)

        # Kembalikan koordinat pusat lingkaran terbaik
        return coefficient
    else:
        print("Tidak ada lingkaran yang terdeteksi.")
        return None
    
# detect_yolo('images/tes1/baby_up.jpeg')