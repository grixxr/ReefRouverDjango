import cv2
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import base64
import os
import numpy as np
import aiohttp

# Path to your YOLO model
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

    async def fetch_frames(self, url):
        """
        Fetch video frames from an MJPEG HTTP stream using aiohttp.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                buffer = b""
                async for chunk in response.content.iter_chunked(1024):
                    buffer += chunk

                    # Detect JPEG boundaries
                    start = buffer.find(b"\xff\xd8")  # JPEG start
                    end = buffer.find(b"\xff\xd9")   # JPEG end

                    if start != -1 and end != -1:
                        jpeg_data = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        # Decode frame
                        frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        yield frame

    async def stream_video(self):
        # URL of the MJPEG stream
        video_url = "http://10.0.254.17:8081"

        # Fetch and process frames
        frame_skip = 10  # Process every 2nd frame
        frame_count = 0

        async for frame in self.fetch_frames(video_url):
            if not self.running:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Downscale frame for faster YOLO inference
            resized_frame = cv2.resize(frame, (640, 480))

            # Predict using YOLO model
            results = self.model.predict(source=resized_frame, conf=0.4)
            annotated_frame = results[0].plot()

            # Encode the annotated frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send the frame to WebSocket client
            await self.send(text_data=frame_base64)

            # Sleep to control FPS (e.g., 15 FPS)
            await asyncio.sleep(1 / 75)
