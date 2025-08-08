# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Cử chỉ tay mà bạn muốn thu thập dữ liệu
gestures = ["up", "down", "left", "right", "fist", "open_hand"]
data = []

print("Bắt đầu thu thập dữ liệu cử chỉ tay...")

for gesture in gestures:
    print(f"\nHãy thực hiện cử chỉ: {gesture}")
    input("Nhấn Enter để bắt đầu thu thập dữ liệu...")

    num_samples = 0
    while num_samples < 1000:  # Thu thập 50 mẫu cho mỗi cử chỉ
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lấy tọa độ landmark và chuẩn hóa
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                # Lưu landmark và nhãn
                data.append(np.append(landmarks, gesture))
                num_samples += 1

                # Vẽ landmark lên khung hình để theo dõi
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        cv2.putText(frame, f"Gesture: {gesture} | Samples: {num_samples}/1000", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Lưu dữ liệu vào file CSV
df = pd.DataFrame(data)
df.to_csv("hand_landmarks.csv", index=False)
print("\nThu thập dữ liệu hoàn tất và đã lưu vào hand_landmarks.csv")

cap.release()
cv2.destroyAllWindows()