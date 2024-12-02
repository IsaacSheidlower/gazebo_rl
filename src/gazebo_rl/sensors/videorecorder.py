import cv2
import threading
import queue
import logging
import time

class VideoRecorder:
    def __init__(self, video_file, codec='H264', fps=30, frame_size=(640, 480), max_queue_size=100):
        """
        Initializes the VideoRecorder.

        Parameters:
            video_file (str): Path to the output video file.
            codec (str): FourCC code for the video codec.
            fps (float): Frames per second.
            frame_size (tuple): Frame size as (width, height).
            max_queue_size (int): Maximum size of the frame queue.
        """
        self.video_file = video_file
        self.codec = codec
        self.fps = fps
        self.frame_size = frame_size
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.is_recording = False
        self.writer_thread = None
        self.logger = logging.getLogger(self.__class__.__name__ + str(time.time()))
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def start(self):
        print('videorecorder start')
        """Starts the video recording."""
        if self.is_recording:
            self.logger.warning("Recording is already in progress.")
            return
        self.is_recording = True
        self.writer_thread = threading.Thread(target=self._video_writer)
        self.writer_thread.start()
        self.logger.info(f"Video recording started. id {id(self)}")

    def stop(self):
        """Stops the video recording."""
        if not self.is_recording:
            self.logger.warning("Recording is not in progress.")
            return
        self.is_recording = False
        # Signal the writer thread to exit
        self.frame_queue.put(None)
        self.writer_thread.join()
        self.logger.info("Video recording stopped.")

    def write_frame(self, frame):
        """Adds a frame to the queue to be written to the video file.

        Parameters:
            frame (numpy.ndarray): The frame to write.
        """
        if not self.is_recording:
            self.logger.warning("Recording has not started yet. Call start() before writing frames.")
            return
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            self.logger.warning("Frame queue is full; dropping frame.")

    def _video_writer(self):
        """The internal method that runs in a separate thread to write frames to the video file."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out = cv2.VideoWriter(self.video_file, fourcc, self.fps, self.frame_size)
            self.logger.info(f"{self.video_file=}")
            if not out.isOpened():
                self.logger.error("Failed to open VideoWriter.")
                return
            self.logger.info("VideoWriter opened successfully.")
            while True:
                frame = self.frame_queue.get()
                if frame is None:
                    self.logger.info("none frame")
                    break
                # self.logger.info(f"before write {str(frame.shape)}")
                try:
                    out.write(frame)
                except Exception as e:
                    self.logger.error(f"Failed to write frame {e}")
                # self.logger.info("after write")
                self.frame_queue.task_done()
        except Exception as e:
            self.logger.error(f"An exception occurred in the writer thread: {e}")
        finally:
            out.release()
            self.logger.info("VideoWriter released.")

    def is_alive(self):
        """Checks if the writer thread is still running.

        Returns:
            bool: True if the writer thread is alive, False otherwise.
        """
        return self.writer_thread.is_alive() if self.writer_thread else False
    



def main():
    # Set up logging to display debug messages
    logging.basicConfig(level=logging.DEBUG)

    # Parameters for VideoRecorder
    video_file = 'output_test.mp4'  # Output file name
    codec = 'mp4v'                  # Codec to use ('XVID', 'MJPG', 'mp4v', etc.)
    fps = 20.0                      # Frames per second
    frame_size = (640, 480)         # Frame size (width, height)

    # Create an instance of VideoRecorder
    recorder = VideoRecorder(
        video_file=video_file,
        codec=codec,
        fps=fps,
        frame_size=frame_size,
        max_queue_size=100
    )

    # Start the recorder
    recorder.start()

    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open the camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from the camera.")
                break

            # Resize frame to match the frame_size if necessary
            if (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size)

            # Write the frame to the recorder
            recorder.write_frame(frame)

            # Display the frame (optional)
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit signal received. Exiting...")
                break

    except Exception as e:
        logging.error(f"An exception occurred: {e}")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        # Stop the recorder
        recorder.stop()
        logging.info("Resources released and recorder stopped.")

if __name__ == '__main__':
    main()