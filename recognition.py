import cv2

# 人脸识别
def recognition(mydict):

    # print("按'q'退出！")
    mydict = mydict
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer/trainer.yml')
    cascadePath = "./data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x - 4, y - 4), (x + w + 4, y + h + 4), (225, 222, 101), 3)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 80:
                if str(Id) in mydict or True:
                    Id = mydict[str(Id)]
            else:
                Id = "-"

            cv2.rectangle(im, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 0), 2)

            cv2.rectangle(im, (x - 6, y - 34), (x + w + 6, y - 4), (225, 222, 101), -1)
            cv2.putText(im, str(Id), (x + 2, y - 12), font, 0.8, (88, 88, 61), 3)
            cv2.putText(im, str(Id), (x + 2, y - 12), font, 0.8, (214, 214, 214), 2)
        cv2.imshow('im', im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()