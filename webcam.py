import cv2 as cv

cap = cv.VideoCapture(0)  # Try 1 if 0 doesn't work

if not cap.isOpened():
    print("❌ Error: Could not open webcam. Try using cv2.VideoCapture(1).")
else:
    while True:
        ret, frame = cap.read()
        cv.imshow("Live Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()