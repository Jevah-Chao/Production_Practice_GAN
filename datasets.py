import cv2


# 人脸数据采集
def datasets():
    mydict = {"0": "Zhendong_You", "1": "Baoze_Wang"}
    return mydict
    print('准备开始人脸数据采集')

    mydict = {}
    while True:
        print("输入'q'停止添加")
        face_id = input('请设置新的人脸id(id为数字)：')
        if face_id == 'q':
            break
        face_name = input('请输入新的人脸name(name为英文或字母)：')
        if face_name == 'q':
            break
        mydict[face_id] = face_name
        # print(mydict)
        count = 0
        vid_cam = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

        while True:
            _, image_frame = vid_cam.read()
            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite("./dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow('frame', image_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif count > 60:
                print('%s:人脸数据采集完成！' % (mydict[face_id]))
                break
        vid_cam.release()
        cv2.destroyAllWindows()
    print(mydict)
    return mydict