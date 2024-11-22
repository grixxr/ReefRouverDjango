import cv2
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import base64
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'ReefRouver.pt')


class VideoFeedConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept WebSocket connection
        await self.accept()

        # Load YOLO model
        self.model = YOLO(model_path)

        # Start sending video frames
        self.running = True
        asyncio.create_task(self.stream_video())

    async def disconnect(self, close_code):
        # Stop streaming video when the WebSocket is disconnected
        self.running = False

    async def stream_video(self):
        # Open the webcam feed
        camera = cv2.VideoCapture(0)

        while self.running:
            success, frame = camera.read()
            if not success:
                break

            # Predict using YOLO model
            results = self.model.predict(source=frame, conf=0.4)
            annotated_frame = results[0].plot()

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send frame to WebSocket client
            await self.send(text_data=frame_base64)

            # Sleep to control frame rate (e.g., 30 FPS)
            await asyncio.sleep(1 / 30)

        # Release camera resource
        camera.release()
