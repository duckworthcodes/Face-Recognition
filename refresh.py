import cv2
import imutils
import winsound
import threading
import logging
import numpy as np

#Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchoolSecuritySystem:
    def __init__(self, camera_index=0, frame_width=640, frame_height=480):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        _, self.start_frame = self.cap.read()
        self.start_frame = self._process_frame(self.start_frame)

        self.alarm = False
        self.alarm_mode = False
        self.alarm_counter = 0

        # Load pre-trained MobileNet SSD model
        self.net = cv2.dnn.readNetFromCaffe("C:\Work And Projects\Spotify Automation\deploy.prototxt", "C:\Work And Projects\Spotify Automation\MobileNetSSD_deploy.caffemodel")

        # Create a threading.Event to signal when the threads should stop
        self.stop_event = threading.Event()

        # Create a thread for the main functionality
        self.main_thread = threading.Thread(target=self.run, daemon=True)

    def _process_frame(self, frame):
        frame = imutils.resize(frame, width=500)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame

    def beep_alarm(self):
        for _ in range(5):
            if not self.alarm_mode:
                break
            logger.info("ALARM!!!!!")
            winsound.Beep(2500, 1000)
        self.alarm = False

    def detect_persons(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Set the input to the neural network
        self.net.setInput(blob)

        # Forward pass and get detection
        detections = self.net.forward()

        # Filter out weak detections
        rects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                rects.append(box.astype("int"))

        return rects

    def run(self):
        while not self.stop_event.is_set():
            _, frame = self.cap.read()
            processed_frame = self._process_frame(frame)

            if self.alarm_mode:
                frame_diff = cv2.absdiff(processed_frame, self.start_frame)
                threshold = cv2.threshold(frame_diff, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]
                self.start_frame = processed_frame

                if threshold.sum() > self.MOTION_THRESHOLD:
                    self.alarm_counter += 1
                else:
                    if self.alarm_counter > 0:
                        self.alarm_counter -= 1

                    # Detect persons and draw boxes
                    rects = self.detect_persons(frame)
                    for rect in rects:
                        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

                cv2.imshow("School Security Cam", frame)
            else:
                cv2.imshow("School Security Cam", processed_frame)

            if self.alarm_counter > self.ALARM_THRESHOLD:
                if not self.alarm:
                    self.alarm = True
                    threading.Thread(target=self.beep_alarm).start()

            key_pressed = cv2.waitKey(1)
            if key_pressed == ord("t"):
                self.alarm_mode = not self.alarm_mode
                self.alarm_counter = 0
                logger.info("Alarm mode changed to: %s", self.alarm_mode)
            elif key_pressed == ord("q"):
                self.alarm_mode = False
                logger.info("Exiting the school security system.")
                break

    def start(self):
        # Start the main thread
        self.main_thread.start()

    def stop(self):
        # Set the stop event to signal threads to stop
        self.stop_event.set()
        # Wait for the main thread to finish
        self.main_thread.join()

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Constants
    SchoolSecuritySystem.THRESHOLD_VALUE = 25
    SchoolSecuritySystem.MOTION_THRESHOLD = 300
    SchoolSecuritySystem.ALARM_THRESHOLD = 20

    security_system = SchoolSecuritySystem(camera_index=0, frame_width=640, frame_height=480)
    security_system.start()

    try:
        # Keep the main thread running until KeyboardInterrupt is received
        while True:
            pass
    except KeyboardInterrupt:
        # Stop the system when KeyboardInterrupt (Ctrl+C) is received
        security_system.stop()