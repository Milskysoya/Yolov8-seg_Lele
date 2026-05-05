# main.py
import cv2
from hand_detector import HandDetector
from gesture_drawer import GestureDrawer
from config import CUSTOM_IMAGES, GESTURE_WINDOW_WIDTH, GESTURE_WINDOW_HEIGHT

def main():
    print("=" * 50)
    print("  HAND GESTURE RECOGNITION")
    print("=" * 50)

    detector = HandDetector()
    drawer = GestureDrawer(width=GESTURE_WINDOW_WIDTH, height=GESTURE_WINDOW_HEIGHT)

    if CUSTOM_IMAGES:
        print("Loading custom images...")
        drawer.load_custom_images(CUSTOM_IMAGES)
    else:
        print("Menggunakan gambar default")

    print("Membuka webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Tidak bisa membuka webcam!")
        return

    print("Webcam berhasil dibuka")
    print("Tekan 'q' untuk keluar")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_hands(rgb_frame)

        finger_count_first = 0  # default untuk gesture display

        if results.multi_hand_landmarks and results.multi_handedness:
            total_fingers = 0

            for i, (hand_landmarks, handedness_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                detector.draw_landmarks(frame, hand_landmarks)

                handedness_label = handedness_info.classification[0].label
                tangan_asli = "Kanan" if handedness_label == "Left" else "Kiri"

                gesture_name, finger_count = detector.recognize_gesture(
                    hand_landmarks.landmark,
                    handedness_label
                )
                total_fingers += finger_count

                if i == 0:
                    finger_count_first = finger_count

                h_pos = hand_landmarks.landmark[0]
                x_pos = int(h_pos.x * w)
                y_pos = int(h_pos.y * h)

                cv2.putText(
                    frame,
                    f"{tangan_asli}: {gesture_name} ({finger_count})",
                    (max(x_pos - 80, 0), max(y_pos - 20, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                f"Total Jari: {total_fingers}/10",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2
            )

        # Tampilkan gesture image di window terpisah
        gesture_img = drawer.create_gesture_image(finger_count_first)
        cv2.imshow('Gesture Display', gesture_img)
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program ditutup. Terima kasih!")

if __name__ == "__main__":
    main()
