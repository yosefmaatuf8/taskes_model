import subprocess
import threading

def monitor_ffmpeg_logs(process):
    """
    Monitors ffmpeg's stderr output.
    It prints each log line and checks for key phrases that indicate streaming activity or errors.
    """
    while True:
        line = process.stderr.readline()
        if not line:
            break
        decoded_line = line.decode("utf-8", errors="replace")
        print(decoded_line, end="")  # Print the log line
        
        # Look for patterns that indicate active streaming.
        # For example, ffmpeg often prints lines starting with "frame=" or "bitrate="
        if "frame=" in decoded_line or "bitrate=" in decoded_line:
            print("✅ Streaming appears to be active!")
        
        # Check if there are any error messages in the logs.
        if "error" in decoded_line.lower():
            print("⚠️ Error detected in ffmpeg logs:", decoded_line)

def start_streaming():
    """
    Starts ffmpeg to stream audio from an input file to a local RTMP endpoint.
    Adjust the input source and RTMP URL to your environment.
    """
    # Define the RTMP endpoint as a variable.
    rtmp_endpoint = "rtmp://localhost/live/stream"
    print(f"Using RTMP endpoint: {rtmp_endpoint}")

    # Configure the ffmpeg command:
    # - '-re' reads the input in real-time.
    # - '-i sample.wav' specifies the input file. Replace with your audio source if needed.
    # - '-c:a aac' encodes audio with AAC.
    # - '-b:a 128k' sets the audio bitrate.
    # - '-f flv' specifies the output format required for RTMP streaming.
    command = [
        "ffmpeg",
        "-re",                    # Read input at native frame rate
        "-i", "sample.wav",       # Input audio file (replace with your source)
        "-c:a", "aac",            # Use AAC audio codec
        "-b:a", "128k",           # Audio bitrate
        "-f", "flv",              # Output format for RTMP
        rtmp_endpoint             # RTMP endpoint variable
    ]
    
    # Start the ffmpeg process, capturing stderr for log monitoring.
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    # Launch a separate thread to monitor the ffmpeg logs.
    log_thread = threading.Thread(target=monitor_ffmpeg_logs, args=(process,))
    log_thread.start()
    
    # Wait for the ffmpeg process to complete.
    process.wait()
    log_thread.join()

if __name__ == "__main__":
    start_streaming()
