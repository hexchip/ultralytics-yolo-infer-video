#!/usr/bin/env python3

import argparse
import math
import cv2
import threading
import queue
from ultralytics import YOLO

class YoloVideoInferencer:
    DEFAULT_FPS = 30
    MAX_QUEUE_SIZE = 30
    STATUS_MARGIN = 30
    FONT_CONFIG = {
        'primary': (cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2),
        'secondary': (cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    }
    SPEED_STEPS = [1.0, 1.25, 1.5, 1.75, 2.0]

    def __init__(self, model_path, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"视频文件打开失败: {video_path}")
        
        self.model = YOLO(model_path)
        
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.base_fps = original_fps if original_fps > 0 else self.DEFAULT_FPS
        self.current_speed_index = 0
        self.speed_multiplier = self.SPEED_STEPS[self.current_speed_index]
        self.current_fps = self.base_fps * self.speed_multiplier
        self.frame_interval = math.ceil(1000 / self.current_fps)
        
        self.frame_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.processed_frame_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        
        self.running = True
        self.paused = False

    def _handle_input_key(self, delay):

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            self.running = False
        elif key == ord(' '):
            self.paused = not self.paused
            while self.paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    self.paused = False
                elif key == ord('q'):
                    self.running = False
                    break
        elif key == ord('+'):
            if self.current_speed_index < len(self.SPEED_STEPS) - 1:
                self.current_speed_index += 1
                self.speed_multiplier = self.SPEED_STEPS[self.current_speed_index]
                self.current_fps = self.base_fps * self.speed_multiplier
                self.frame_interval = math.ceil(1000 / self.current_fps)
        elif key == ord('-'):
            if self.current_speed_index > 0:
                self.current_speed_index -= 1
                self.speed_multiplier = self.SPEED_STEPS[self.current_speed_index]
                self.current_fps = self.base_fps * self.speed_multiplier
                self.frame_interval = math.ceil(1000 / self.current_fps)

    def _frame_reader(self):
        while self.running: 
            success, frame = self.cap.read()
            if not success:
                break
            
            is_put = False
            while not is_put:
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                    is_put = True
                except queue.Full:
                    if not self.running:
                        break
        
        # end
        if self.running:
            self.frame_queue.put(None)

    def _frame_processor(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if frame is None:
                # end
                self.processed_frame_queue.put(None)
                self.frame_queue.task_done()
                break
            
            results = self.model.predict(frame, conf=0.5, imgsz=(384, 800))
            annotated_frame = results[0].plot()
            
            self.frame_queue.task_done()

            is_put = False
            while not is_put:
                try:
                    self.processed_frame_queue.put(annotated_frame, timeout=0.1)
                    is_put = True
                except queue.Full:
                    if not self.running:
                        break

    def _display_loop(self):

        while self.running:
            has_frame = False
            try:
                frame = self.processed_frame_queue.get(timeout=0.01)
                has_frame = True
            except queue.Empty:
                has_frame = False

            if not has_frame:
                self._handle_input_key(10)
                continue

            # end
            if frame is None:
                self.processed_frame_queue.task_done()
                break

            self._render_ui(frame)

            cv2.imshow("YOLO Inference", frame)

            self._handle_input_key(self.frame_interval)

            self.processed_frame_queue.task_done()

    def _render_ui(self, frame):
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (self.STATUS_MARGIN, self.STATUS_MARGIN), 
                   *self.FONT_CONFIG['primary'])
        cv2.putText(frame, f"Speed: x{self.speed_multiplier}", 
                   (self.STATUS_MARGIN, self.STATUS_MARGIN + 25),
                   *self.FONT_CONFIG['secondary'])
        cv2.putText(frame, "[SPACE] Pause/Resume", 
                   (self.STATUS_MARGIN, self.STATUS_MARGIN + 50),
                   *self.FONT_CONFIG['secondary'])

    def run(self):
        try:
            frame_reader_thread = threading.Thread(target=self._frame_reader)
            frame_processor_thread = threading.Thread(target=self._frame_processor)
            
            frame_reader_thread.start()
            frame_processor_thread.start()
            
            self._display_loop()
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video using YOLO model.")
    parser.add_argument("--modelPath", required=True, help="Name of the YOLO model file.")
    parser.add_argument("--videoPath", required=True, help="Name of the video file.")
    
    try:
        args = parser.parse_args()
        Inferencer = YoloVideoInferencer(
            model_path=args.modelPath,
            video_path=args.videoPath,
        )
        Inferencer.run()
    except Exception as e:
        print(f"[Error] {str(e)}")
        exit(1)