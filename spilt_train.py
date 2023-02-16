import cv2
import numpy as np
import torch
from utils.general import non_max_suppression,scale_coords
from utils.plots import Annotator,Colors
from utils.augmentations import letterbox


def load_image(img_size = 640,augment = True,path):
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / float(max(h0, w0))  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def load_model():
    model = torch.hub.load('E:/yolov5-master/yolov5_v6/yolov5', 'custom',
                           path='E:/yolov5-master/yolov5_v6/yolov5/screw_best.pt', source='local')
    # model = torch.load('E:/yolov5-master/yolov5-master/best.pt')
    names = model.names
    colors = Colors()
    return model,names,colors

def detect_screw(path,model,names,colors):
    model.eval()
    # Read an input image
    img = cv2.imread(path)
    img0 = img.copy()
    img,(h0,w0),(h,w) = load_image(path)
    # Pre-process the image
    # img =  cv2.resize(img, (640, 640))
    img,ratio,pad = letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, stride=32)
    img = np.array(img, dtype=np.float32) / 255.0
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # img = np.ascontiguousarray(img)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    # Convert into torch
    # img = torch.from_numpy(img).float()
    # img /= 255  # 0 - 255 to 0.0 - 1.0
    img = torch.tensor(img, dtype=torch.float32)
    # if len(img.shape) == 3:
    #     im = img[None] # expand for batch dim
    img = img.to('cuda')

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred)
    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):  # per image
        img1 = img0.copy()
        # color = (30, 30, 30)
        txt_color = (255, 255, 255)
        # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        # h_size, w_size = img.shape[-2:]
        # h, w = img1.shape[:2]
        # gain = (w_size / w, h_size / h,w_size / w, h_size / h)
        # print(h_size, w_size)
        lw = max(round(sum(img.shape) / 2 * 0.003), 2)
        if len(det):
            # # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # gain_tensor = torch.tensor(gain)
            # gain_tensor = gain_tensor.to(det.device)
            # det[:, :4] /= gain_tensor
            #
            # # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                # if conf>0.4:
                # if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label =  f'{names[c]} {conf:.2f}'
                        color = colors(c, True)
                        box = xyxy
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        cv2.rectangle(img1, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                        if label:
                            tf = max(lw - 1, 1)  # font thickness
                            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                            outside = p1[1] - h >= 3
                            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                            cv2.rectangle(img1, p1, p2, color, -1, cv2.LINE_AA)  # filled
                            cv2.putText(img1,
                                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                        0,
                                       lw / 3,
                                        txt_color,
                                        thickness=tf,
                                        lineType=cv2.LINE_AA)

                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())

                        # confidence_scere = conf
                        class_index = cls
                        object_name = names[int(cls)]

                        print('bounding box is', x1, y1, x2, y2)
                        print('class index is', class_index)
                        print('detected object name is', object_name)
                        #find center
                        original_img = img0
                        cropped_img = img0[y1:y2,x1:x2, :]
                        # Turn to Gray
                        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
                        # Subtract the blurred image from the original image
                        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
                        edges = cv2.Canny(sharpened, 50, 150)
                        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=30, minRadius=0,
                                                   maxRadius=0)
                        if circles is None:
                            print("no circles detected")
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                # outer circle
                                cv2.circle(img1, (i[0]+x1, i[1]+y1), i[2], (0, 0, 255), 2)
                                # center
                                cv2.circle(img1, (i[0]+x1, i[1]+y1), 1, (255, 0, 0), 2)
                                print(i[0]+x1, i[1]+y1, i[2])
                        # cv2.imwrite('test.png',cropped_img)


    det = det.tolist()
    print(det)
    cv2.namedWindow("Screw Detection",cv2.WINDOW_NORMAL)
    cv2.imshow("Screw Detection", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
