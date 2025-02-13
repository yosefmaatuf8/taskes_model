from flask import Flask, render_template
from extensions import socketio  # Import the shared SocketIO instance
from logger import DualLogger
import sys
import time
from stream_recorder import StreamRecorder  # Assuming this handles streaming

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Initialize SocketIO with eventlet support
socketio.init_app(app, async_mode='eventlet')

# Override stdout after initializing SocketIO
sys.stdout = DualLogger(sys.stdout)

@app.route('/')
def index():
    return render_template('index.html')

def stream_logs():
    """Function to start the stream recorder and monitor logs."""
    recorder = StreamRecorder()
    recorder.monitor_and_record()

if __name__ == '__main__':
    # Start the recording process in a background task instead of using threading.Thread
    socketio.start_background_task(target=stream_logs)

    # Run the app using eventlet
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
