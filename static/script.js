const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');

        // Start webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            });

        function autoCaptureAndSend() {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob, 'webcam.jpg');

                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById('result').textContent = 'Detected Emotion: ' + data.emotion;
                })
                .catch(err => {
                    document.getElementById('result').textContent = 'Error detecting emotion';
                });
            }, 'image/jpeg');
        }

        // Run every 1/2 seconds
        setInterval(autoCaptureAndSend, 500);