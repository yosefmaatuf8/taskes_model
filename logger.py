from extensions import socketio  # Import shared SocketIO instance

class DualLogger:
    def __init__(self, original_stdout):
        self.terminal = original_stdout

    def write(self, message):
        # Write message to the terminal (console)
        self.terminal.write(message)
        self.terminal.flush()

        # If the message is not just a newline, emit it via Socket.IO
        if message.strip():
            try:
                socketio.emit('log_message', {'data': message}, namespace='/')
                socketio.sleep(0)  # Ensures messages are sent immediately
            except Exception as e:
                self.terminal.write(f"\n[SocketIO Error] {e}\n")
                self.terminal.flush()

    def flush(self):
        self.terminal.flush()
