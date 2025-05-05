import cv2
import imutils
import winsound
import threading
import logging
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchoolSecuritySystem:
    def __init__(self, camera_index=0, frame_width=640, frame_height=480):
        # Initialize camera with error checking
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.error("Failed to open camera!")
            raise RuntimeError("Could not open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Read initial frame
        ret, self.start_frame = self.cap.read()
        if not ret:
            logger.error("Failed to read initial frame!")
            raise RuntimeError("Could not read from camera")
            
        self.start_frame = self._process_frame(self.start_frame)
        logger.info("Camera initialized successfully")

        self.alarm = False
        self.alarm_counter = 0

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
        while not self.stop_event.is_set():
            if self.alarm_counter <= self.ALARM_THRESHOLD:
                break
            logger.info("ALARM!!!!!")
            winsound.Beep(2500, 1000)
        self.alarm = False

    def run(self):
        logger.info("Starting security system...")
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera!")
                break

            processed_frame = self._process_frame(frame)

            # Check for motion
            frame_diff = cv2.absdiff(processed_frame, self.start_frame)
            threshold = cv2.threshold(frame_diff, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate motion level for debugging
            motion_level = threshold.sum()
            if motion_level > self.MOTION_THRESHOLD:
                logger.info(f"Motion detected! Level: {motion_level}")
                self.alarm_counter += 1
            else:
                if self.alarm_counter > 0:
                    self.alarm_counter -= 1

            # Update the reference frame
            self.start_frame = processed_frame

            # Show the original color frame
            cv2.imshow("Security Camera", frame)

            if self.alarm_counter > self.ALARM_THRESHOLD:
                if not self.alarm:
                    self.alarm = True
                    logger.info("Triggering alarm!")
                    threading.Thread(target=self.beep_alarm).start()

            key_pressed = cv2.waitKey(1)
            if key_pressed == ord("q"):
                logger.info("Exiting the security system.")
                break

    def start(self):
        self.main_thread.start()

    def stop(self):
        self.stop_event.set()
        self.main_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Constants - Adjusted for better sensitivity
    SchoolSecuritySystem.THRESHOLD_VALUE = 25
    SchoolSecuritySystem.MOTION_THRESHOLD = 1000  # Lowered threshold for more sensitivity
    SchoolSecuritySystem.ALARM_THRESHOLD = 5      # Lowered threshold for faster alarm

    try:
        security_system = SchoolSecuritySystem(camera_index=0, frame_width=640, frame_height=480)
        security_system.start()

        # Keep the main thread running until KeyboardInterrupt is received
        while True:
            pass
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        if 'security_system' in locals():
            security_system.stop()
