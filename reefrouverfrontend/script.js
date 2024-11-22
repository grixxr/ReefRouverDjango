document.addEventListener("DOMContentLoaded", function () {
    const videoElement = document.querySelector('img[alt="Live Feed"]');
    const ws = new WebSocket('ws://127.0.0.1:8000/ws/video_feed/');

    ws.onopen = function () {
        console.log("WebSocket connection opened.");
    };

    ws.onmessage = (event) => {
        const frameBase64 = event.data;
        if (videoElement) {
            videoElement.src = `data:image/jpeg;base64,${frameBase64}`;
        }
    };

    ws.onclose = () => {
        console.error('WebSocket closed');
    };
});


