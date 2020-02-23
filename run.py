import cv2
from kcf_tracker import *

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1  # 间隔


def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # 滑动
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # 左键放开
        selectingObject = False
        if(abs(x-ix) > 10 and abs(y-iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        onTracking = False
        if(w > 0):
            ix, iy = x-w/2, y-h/2
            initTracking = True


if __name__ == '__main__':
    # 读取视频流
    capture = cv2.VideoCapture(0)
    # 定义跟踪器
    tracker = KCFtracker(True)
    # 获取检测框
    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_boundingbox)

    while(capture.isOpened()):
        ret, frame = capture.read()
        if not ret:
            break
        if(selectingObject):
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)

        elif(initTracking):
            cv2.rectangle(frame, (ix, iy), (ix+w, iy+h), (0, 255, 255), 2)
            # 初始化跟踪器
            tracker.init(frame, ix, iy, w, h)

            initTracking = False
            onTracking = True

        elif(onTracking):
            x1,y1=tracker.update(frame)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 255), 2)

        cv2.imshow('tracking', frame)
        cv2.waitKey(inteval)
capture.release()
cv2.destroyAllWindows()
