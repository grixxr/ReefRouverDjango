import cv2
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import base64
import os
import csv  # For saving CSV files
import time  # For generating unique filenames

model_path = os.path.join(os.path.dirname(__file__), 'models', 'ReefRouver.pt')

# Create a "recordings" directory if it doesn't exist
recordings_dir = os.path.join(os.getcwd(), 'recordings')
os.makedirs(recordings_dir, exist_ok=True)


class VideoFeedConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept WebSocket connection
        await self.accept()

        # Load YOLO model
        self.model = YOLO(model_path)

        # Initialize recording flag and storage
        self.recording = False
        self.video_writer = None
        self.csv_data = [["Object", "Confidence", "Coordinates"]]  # Changed to CSV

        # Start sending video frames
        self.running = True
        asyncio.create_task(self.stream_video())

    async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code {close_code}")
        self.running = False
        # Only stop recording when the WebSocket is closed, not during the 'stop_recording' action
        if self.recording:
            self.stop_recording()

    async def receive(self, text_data):
        # Listen for messages from the frontend
        if text_data == "start_recording":
            self.start_recording()
        elif text_data == "stop_recording":
            self.stop_recording()

    def start_recording(self):
        # Initialize video writer
        self.recording = True
        self.csv_data = [["Object", "Confidence", "Coordinates"]]  # Initialize with header
        # Generate a unique filename using the current timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Define the folder path for recordings and ensure it exists
        self.csv_path = os.path.join(recordings_dir, f"inference_results_{timestamp}.csv")
        self.video_path = os.path.join(recordings_dir, f"recorded_inference_{timestamp}.mp4")

        # Create a new CSV file with the unique filename
        print(f"New CSV file created: {self.csv_path}")

        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480)
        )

    def stop_recording(self):
        # Finalize video and CSV file
        print("Stopping recording and saving CSV file...")  # Debugging line
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        try:
            # Save data to CSV if there are any detections
            if self.csv_data and len(self.csv_data) > 1:
                with open(self.csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write the header and then the data
                    writer.writerows(self.csv_data)
                print(f"Data saved to {self.csv_path}")  # Debugging line
            else:
                print("No detection results to save.")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    async def stream_video(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        frame_skip = 1  # Process every frame
        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            resized_frame = cv2.resize(frame, (640, 480))
            results = self.model.predict(source=resized_frame, conf=0.4)
            annotated_frame = results[0].plot()

            if self.recording and self.video_writer:
                self.video_writer.write(annotated_frame)

                # Append detection results to CSV data
                for detection in results[0].boxes:
                    class_id = int(detection.cls)
                    class_name = results[0].names[class_id]
                    # Format coordinates as string
                    coordinates = ', '.join(map(str, detection.xyxy.tolist()))
                    self.csv_data.append([
                        class_name,
                        round(detection.conf.item(), 2),  # Fix: Convert Tensor to float and round
                        coordinates
                    ])
                print(f"Detection added: {self.csv_data[-1]}")  # Debugging line

            # Encode and send frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            await self.send(text_data=frame_base64)

            # Control FPS
            await asyncio.sleep(1 / 75)

        cap.release()
