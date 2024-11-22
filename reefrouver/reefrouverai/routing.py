from django.urls import path
from reefrouverai.consumers import VideoFeedConsumer

websocket_urlpatterns = [
    path('ws/video_feed/', VideoFeedConsumer.as_asgi()),
]