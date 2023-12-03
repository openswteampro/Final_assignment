import cv2

# 동영상 파일 경로
video_path = '../data/video.mp4'

# HOG 디텍터 생성
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 동영상 열기
cap = cv2.VideoCapture(video_path)

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 보행자 검출
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)

    # 검출된 보행자 표시
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow('Pedestrian Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()