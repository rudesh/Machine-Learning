import numpy as np
import cv2


def absDiff(image1, image2):
    if image1.shape != image2.shape:
        print("image size mismatch")
        return 0

    else:
        height, width, dummy = image1.shape
        # Compute absolute difference.
        diff = cv2.absdiff(image1, image2)
        a = cv2.split(diff)
        # Sum up the differences of the 3 channels with equal weights.
        # You can change the weights to different values.
        sum = np.zeros((height, width), dtype=np.uint8)
        for i in (1, 2, 3):
            ch = a[i - 1]
            cv2.addWeighted(ch, 1.0 / i, sum, float(i - 1) / i, gamma=0.0, dst=sum)
        return sum



#used the same from the previous lab answers
def setBackground(image, diff, threshold, bgcolor):
    ret, mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    fg = image.copy()
    fg[mask != 1] = bgcolor

    return fg




def average_cap(video, sec):

    #capture the video
    cap = cv2.VideoCapture(video)

    #frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    #capture frame
    ret, f = cap.read()

    #seconds
    if sec != 0:
        fps = fps * 3
    else:
        fps = fps * 10

    count = 0
    resFrame = None

    #save the video
    out = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (f.shape[1], f.shape[0]))
    avg = np.float32(f)

    #get the frames
    while (fps - count) > 0:
        print((fps - count))
        ret, t1Frame = cap.read()

        if ret:
            cv2.accumulateWeighted(f, avg, 0.01)
            t2Frame = cv2.convertScaleAbs(avg)
            # ret, t2Frame = cap.read()
            # cv2.accumulateWeighted(f, avg, 0.5)
            resFrame = cv2.addWeighted(t1Frame, 0.6, t2Frame, 0.4, 0)
        out.write(resFrame)
        count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()




def averaged_video(video):

    cap = cv2.VideoCapture(video)
    ret, f = cap.read()
    out = cv2.VideoWriter("extracted.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (f.shape[1], f.shape[0]))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ret, frameZero = cap.read()
            diff = absDiff(frame, frameZero)
            re = setBackground(frameZero, diff, 30, (0, 0, 0))
            out.write(re)
            cv2.imshow("RESULT", re)

        k = cv2.waitKey(25) & 0xff
        if k == 5:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


video = "traffic.avi"
sec = 0

#calling the method using traffic video
average_cap(video, sec)

average_cap_video = "result.avi"

averaged_video(average_cap_video)