import cv2


def generate_dataset():
    face_cascade = cv2.CascadeClassifier(
        r"C:\Users\Jai\DATA SCIENCE 3\Mood Detection\haarcascade_frontalface_default.xml"
    )

    def crop_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None
        for x, y, w, h in faces:
            cropped_face = img[y : y + h, x : x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0

    while True:
        ret, frame = cap.read()

        if crop_image(frame) is not None:
            img_id += 1
            face = cv2.resize(crop_image(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = (
                "generated_dataset/face." + str(id) + "." + str(img_id) + ".jpg"
            )
            cv2.imwrite(file_name_path, face)

            cv2.imshow("Captured Face", cv2.resize(frame, (1000, 600), face))
            if cv2.waitKey(1) == 13 or int(img_id) == 100:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting images completed...")


generate_dataset()
