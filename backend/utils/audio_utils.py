import os
import requests
import subprocess
import uuid

# Configuration
MAX_SIZE_MB = 10
TIMEOUT_SEC = 10
SUPPORT_EXTENSIONS = ('.mp3', '.wav', '.m4a')

def download_audio(url, download_dir):
    """
    Downloads an audio file from a URL.
    Returns the local file path.
    """
    # 1. Validate Extension
    clean_url = url.split('?')[0].lower().strip()
    print(f"DEBUG: Validating URL: '{clean_url}' against {SUPPORT_EXTENSIONS}")
    if not any(clean_url.endswith(ext) for ext in SUPPORT_EXTENSIONS):
        raise ValueError(f"Unsupported file format. Please provide a link ending in {', '.join(SUPPORT_EXTENSIONS)}.")

    # 2. Setup Temporary File
    os.makedirs(download_dir, exist_ok=True)
    ext = os.path.splitext(clean_url)[1]
    local_path = os.path.join(download_dir, f"dl_{uuid.uuid4().hex}{ext}")

    # 3. Stream Download with size limit
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=TIMEOUT_SEC, headers=headers)
        response.raise_for_status()

        # Check content length before downloading
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large. Maximum size is {MAX_SIZE_MB}MB.")

        downloaded_size = 0
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_SIZE_MB * 1024 * 1024:
                        raise ValueError(f"File too large. Maximum size is {MAX_SIZE_MB}MB.")
                    f.write(chunk)
        
        return local_path
    except requests.exceptions.RequestException as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(f"Failed to download audio: {str(e)}")
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise e

def convert_to_standard_wav(input_path, output_dir):
    """
    Converts any audio file to 16000Hz, Mono, WAV format using FFmpeg.
    If FFmpeg is missing and the input is already a .wav, returns the input_path directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"std_{uuid.uuid4().hex}.wav")

    # Check if ffmpeg is available
    ffmpeg_available = False
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        ffmpeg_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        ffmpeg_available = False

    if not ffmpeg_available:
        if input_path.lower().endswith('.wav'):
            print(f"DEBUG: FFmpeg not found, but file is .wav. Skipping conversion.")
            return input_path
        else:
            raise RuntimeError("FFmpeg is not installed on this system. Audio conversion for non-WAV files is not possible. Please install FFmpeg.")

    # ffmpeg command:
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-y',
        output_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        # If conversion fails but it was already a wav, try to fallback
        if input_path.lower().endswith('.wav'):
             return input_path
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during audio conversion: {str(e)}")
