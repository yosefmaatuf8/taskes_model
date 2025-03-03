import sys
from extensions import socketio  # ייבוא המופע המשותף
from logger import DualLogger
from src.stream_recorder import StreamRecorder

if __name__ == '__main__':
    # אתחול socketio
    socketio.init_app(None)  # אם אין לך Flask פה, תוכל להעביר None

    # רק אחרי שהsocketio מאותחל, שנה את sys.stdout
    sys.stdout = DualLogger(sys.stdout)

    # Instantiate the StreamRecorder
    recorder = StreamRecorder()

    # Start the recording process
    recorder.monitor_and_record()
