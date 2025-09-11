import cv2
import requests
import numpy as np
import time
import os
import json
import platform
import subprocess
import websocket
import threading
from datetime import datetime

BACKEND_URL = 'http://10.135.209.220:4000'
WS_URL = 'ws://10.135.209.220:4000'
CACHE_DIR = 'media_cache'
CONFIG_FILE = 'player_config.json'
SCHEDULE_CACHE_FILE = 'current_schedule.json'
DEVICE_INFO_FILE = 'device_info.json' 

os.makedirs(CACHE_DIR, exist_ok=True)

class PlayerManager:
    def __init__(self):
        self.player_id = None
        self.token = None
        self.ws = None
        self.connected = False
        self.config = self.load_config()
        self.device_info = self.detect_device_info()
        self.force_content_refresh = False  # Flag for forced refresh
        
    def load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
        
        return {
            'name': f"Display-{platform.node()}",
            'location': 'Unknown Location'
        }
    
    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def detect_device_info(self):
        device_info = {
            'screenWidth': 1920,
            'screenHeight': 1080,
            'orientation': 'landscape',
            'deviceType': 'display',
            'os': platform.system(),
            'osVersion': platform.release(),
            'pythonVersion': platform.python_version(),
            'hostname': platform.node()
        }
        
        try:
            if platform.system() == 'Windows':
                import tkinter as tk
                root = tk.Tk()
                device_info['screenWidth'] = root.winfo_screenwidth()
                device_info['screenHeight'] = root.winfo_screenheight()
                root.destroy()
            elif platform.system() == 'Linux':
                try:
                    result = subprocess.run(['xrandr'], capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if '*' in line:
                                resolution = line.split()[0]
                                if 'x' in resolution:
                                    width, height = map(int, resolution.split('x'))
                                    device_info['screenWidth'] = width
                                    device_info['screenHeight'] = height
                                    break
                except:
                    pass
            
            if device_info['screenWidth'] > device_info['screenHeight']:
                device_info['orientation'] = 'landscape'
            else:
                device_info['orientation'] = 'portrait'
                
            total_pixels = device_info['screenWidth'] * device_info['screenHeight']
            if total_pixels >= 1920 * 1080:
                device_info['deviceType'] = 'large_display'
            elif total_pixels >= 1280 * 720:
                device_info['deviceType'] = 'display'
            else:
                device_info['deviceType'] = 'small_display'
                
        except Exception as e:
            print(f"Could not detect display info: {e}")
        
        with open(DEVICE_INFO_FILE, 'w') as f:
            json.dump(device_info, f, indent=2)
        
        return device_info
    
    def register_player(self):
        try:
            payload = {
                'deviceInfo': self.device_info,
                'location': self.config.get('location', 'Unknown Location'),
                'name': self.config.get('name', f"Display-{platform.node()}")
            }
            
            response = requests.post(f'{BACKEND_URL}/players/register', 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.player_id = data['playerId']
                self.token = data['token']
                
                self.config['playerId'] = self.player_id
                self.config['token'] = self.token
                self.save_config()
                
                print(f"Registered as player: {self.player_id}")
                return True
            else:
                print(f"Registration failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Registration error: {e}")
            return False
    
    def authenticate(self):
        if not self.config.get('playerId') or not self.config.get('token'):
            return False
            
        try:
            payload = {
                'playerId': self.config['playerId'],
                'token': self.config['token']
            }
            
            response = requests.post(f'{BACKEND_URL}/players/auth', 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                self.player_id = self.config['playerId']
                self.token = self.config['token']
                print(f"Authenticated as player: {self.player_id}")
                return True
            else:
                print("Authentication failed, need to re-register")
                return False
                
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def connect_websocket(self):
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self.handle_ws_message(data)
                except Exception as e:
                    print(f"WebSocket message error: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
                self.connected = False
            
            def on_close(ws, close_status_code, close_msg):
                print("WebSocket connection closed")
                self.connected = False
            
            def on_open(ws):
                print("WebSocket connected")
                ws.send(json.dumps({
                    'type': 'player-connect',
                    'playerId': self.player_id,
                    'token': self.token
                }))
            
            self.ws = websocket.WebSocketApp(WS_URL,
                                           on_message=on_message,
                                           on_error=on_error,
                                           on_close=on_close,
                                           on_open=on_open)
            
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            time.sleep(2)
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
    
    def handle_ws_message(self, data):
        message_type = data.get('type')
        
        if message_type == 'connection-confirmed':
            self.connected = True
            print("WebSocket connection confirmed")
            
        elif message_type == 'connection-rejected':
            print(f"WebSocket connection rejected: {data.get('reason')}")
            self.connected = False
            
        elif message_type == 'player-deleted':
            print("Player has been removed from the system. Shutting down...")
            self.shutdown()
            
        elif message_type == 'content-changed':
            print("Content change notification received - will refresh on next check")
            self.force_content_refresh = True
            
        elif message_type == 'command':
            self.handle_remote_command(data.get('command'), data.get('data'))
            
        elif message_type == 'config-update':
            self.handle_config_update(data.get('config'))
    
    def handle_remote_command(self, command, data):
        print(f"Received command: {command}")
        
        if command == 'restart':
            print("Restart command received")
            
        elif command == 'update_config':
            print("Config update command received")
            if data:
                self.config.update(data)
                self.save_config()
                
        elif command == 'refresh_content':
            print("Refresh content command received")
            self.force_content_refresh = True
    
    def handle_config_update(self, config):
        if config:
            self.config.update(config)
            self.save_config()
            print("Configuration updated")
    
    def send_heartbeat(self):
        try:
            if self.ws and self.connected:
                self.ws.send(json.dumps({
                    'type': 'player-heartbeat',
                    'playerId': self.player_id,
                    'timestamp': datetime.now().isoformat()
                }))
        except Exception as e:
            print(f"Heartbeat error: {e}")
    
    def send_status(self, status):
        try:
            if self.ws and self.connected:
                self.ws.send(json.dumps({
                    'type': 'player-status',
                    'playerId': self.player_id,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }))
        except Exception as e:
            print(f"Status update error: {e}")
    
    def shutdown(self):
        self.connected = False
        if self.ws:
            self.ws.close()

player_manager = PlayerManager()

def connect_to_cms():
    print("Connecting to CMS...")
    
    if not player_manager.authenticate():
        if not player_manager.register_player():
            print("Failed to connect to CMS")
            return False
    
    player_manager.connect_websocket()
    return True

def make_full_url(path):
    if path.startswith('http://') or path.startswith('https://'):
        return path
    return f'{BACKEND_URL}{path}'

def download_media_file(media_item):
    try:
        url = make_full_url(media_item['url'])
        filename = f"{media_item['id']}_{os.path.basename(media_item['url'])}"
        local_path = os.path.join(CACHE_DIR, filename)
        
        if os.path.exists(local_path):
            return local_path
        
        print(f"Downloading {media_item['name']}...")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return local_path
        else:
            print(f"Failed to download {media_item['name']}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error downloading {media_item['name']}: {e}")
        return None

def fetch_schedule():
    try:
        headers = {}
        if player_manager.token:
            headers['Authorization'] = f'Bearer {player_manager.token}'
            
        url = f'{BACKEND_URL}/player-schedule/{player_manager.player_id}'
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            schedule_data = resp.json()
            
            with open(SCHEDULE_CACHE_FILE, 'w') as f:
                json.dump(schedule_data, f, indent=2)
            
            return schedule_data
        else:
            print(f"Failed to fetch schedule: HTTP {resp.status_code}")
            return load_cached_schedule()
    except Exception as e:
        print(f"Failed to fetch schedule: {e}")
        return load_cached_schedule()

def load_cached_schedule():
    try:
        if os.path.exists(SCHEDULE_CACHE_FILE):
            with open(SCHEDULE_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load cached schedule: {e}")
    
    return {
        'playerId': player_manager.player_id,
        'currentSchedule': None,
        'playlist': None,
        'media': [],
        'serverTime': datetime.now().isoformat(),
        'contentHash': 0
    }

def download_all_media(media_list):
    local_media = []
    for media_item in media_list:
        if media_item['type'] in ['image', 'video']:
            local_path = download_media_file(media_item)
            if local_path:
                media_copy = media_item.copy()
                media_copy['local_path'] = local_path
                local_media.append(media_copy)
            else:
                print(f"Skipping {media_item['name']} - download failed")
        elif media_item['type'] == 'text':
            local_media.append(media_item)
    
    return local_media

def resize_for_device(image, device_info):
    if image is None:
        return None
    
    screen_width = device_info['screenWidth']
    screen_height = device_info['screenHeight']
    
    h, w = image.shape[:2]
    
    scale_w = screen_width / w
    scale_h = screen_height / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    start_y = (screen_height - new_h) // 2
    start_x = (screen_width - new_w) // 2
    background[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return background

def get_image(media_item, device_info):
    try:
        if 'local_path' in media_item and os.path.exists(media_item['local_path']):
            img = cv2.imread(media_item['local_path'])
            if img is not None:
                return resize_for_device(img, device_info)
        
        url = make_full_url(media_item['url'])
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            bytes_img = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
            return resize_for_device(img, device_info)
    except Exception as e:
        print(f"Image error: {e}")
    return None

def get_text_image(text, device_info):
    screen_width = device_info['screenWidth']
    screen_height = device_info['screenHeight']
    
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    base_font_scale = max(screen_width / 1920, screen_height / 1080)
    font_scale = base_font_scale * 1.5
    thickness = max(2, int(base_font_scale * 3))
    
    words = text.split()
    lines = []
    current_line = []
    max_width = screen_width - 100
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if text_width < max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    line_height = int(60 * base_font_scale)
    total_height = len(lines) * line_height
    start_y = max(line_height, (screen_height - total_height) // 2)
    
    for i, line in enumerate(lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (screen_width - text_width) // 2
        y = start_y + i * line_height
        
        cv2.putText(img, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

def play_video_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def main():
    print(f"Digital Signage Player Starting...")
    print(f"Device: {player_manager.device_info['deviceType']} - {player_manager.device_info['screenWidth']}x{player_manager.device_info['screenHeight']}")
    
    if not connect_to_cms():
        print("Failed to connect to CMS, exiting...")
        return
    
    cv2.namedWindow("Media Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Media Player", 
                     player_manager.device_info['screenWidth'], 
                     player_manager.device_info['screenHeight'])
    
    try:
        cv2.setWindowProperty("Media Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        pass
    
    last_schedule_check = 0
    last_heartbeat = 0
    current_media_list = []
    current_index = 0
    start_display = time.time()
    current_content = None
    current_type = None
    current_duration = 5
    cap = None
    last_content_hash = 0  # Track content changes
    
    print(f"Player {player_manager.player_id} ready for content...")
    player_manager.send_status("ready")
    
    while True:
        now = time.time()
        
        # Check if player was shut down
        if not player_manager.connected and current_media_list:
            print("Player disconnected, shutting down...")
            break
        
        if cv2.getWindowProperty("Media Player", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        if now - last_heartbeat >= 30:
            player_manager.send_heartbeat()
            last_heartbeat = now
        
        # Check for schedule updates every 5 seconds OR when forced refresh is requested
        should_check_schedule = (now - last_schedule_check >= 5) or player_manager.force_content_refresh
        
        if should_check_schedule:
            schedule_data = fetch_schedule()
            new_media_list = schedule_data.get('media', [])
            new_content_hash = schedule_data.get('contentHash', 0)
            
            # Only reset playlist if content actually changed
            content_changed = (new_content_hash != last_content_hash) or player_manager.force_content_refresh
            
            if content_changed:
                print("Content changed detected - updating playlist...")
                player_manager.send_status("downloading")
                
                current_media_list = download_all_media(new_media_list)
                current_index = 0  # Reset to start of new playlist
                current_content = None
                cap = None
                last_content_hash = new_content_hash
                player_manager.force_content_refresh = False
                
                if schedule_data.get('currentSchedule'):
                    schedule_name = schedule_data['currentSchedule'].get('name', 'Unknown')
                    playlist_name = schedule_data.get('playlist', {}).get('name', 'Unknown')
                    print(f"Playing: {schedule_name} -> {playlist_name}")
                    print(f"Loaded {len(current_media_list)} media items")
                    player_manager.send_status(f"playing:{playlist_name}")
                else:
                    print("No active schedule")
                    player_manager.send_status("idle")
            else:
                # Content hasn't changed, just continue with current playlist
                if new_media_list != current_media_list:
                    print("Schedule data updated but content unchanged - continuing playlist")
            
            last_schedule_check = now
        
        if not current_media_list:
            no_content_img = get_text_image("Waiting for content...", player_manager.device_info)
            cv2.imshow("Media Player", no_content_img)
            
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q') or key == 27:
                break
            continue
        
        # Load next media if needed (duration expired or no current content)
        if current_content is None or (now - start_display >= current_duration):
            if current_index >= len(current_media_list):
                current_index = 0  # Loop back to start
                
            media = current_media_list[current_index]
            current_type = media.get('type')
            current_duration = media.get('playlistDuration', media.get('duration', 5))
            
            print(f"[{current_index + 1}/{len(current_media_list)}] Loading: {media.get('name', 'Unknown')} ({current_type}) for {current_duration}s")
            
            if current_type == "image":
                current_content = get_image(media, player_manager.device_info)
            elif current_type == "text":
                text_content = media.get('url', media.get('name', 'Text Content'))
                current_content = get_text_image(text_content, player_manager.device_info)
            elif current_type == "video":
                video_path = media.get('local_path') or make_full_url(media.get('url', ''))
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open video: {video_path}")
                    cap = None
                    current_content = None
                else:
                    current_content = "video"
                    if current_duration <= 0:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if fps > 0:
                            current_duration = frame_count / fps
                        else:
                            current_duration = 30
            else:
                print(f"Unsupported media type: {current_type}")
                current_content = None
            
            start_display = now
            current_index = (current_index + 1) % len(current_media_list)
        
        # Display content
        if current_type == "image" and current_content is not None and isinstance(current_content, np.ndarray):
            cv2.imshow("Media Player", current_content)
        elif current_type == "text" and current_content is not None and isinstance(current_content, np.ndarray):
            cv2.imshow("Media Player", current_content)
        elif current_type == "video" and cap is not None:
            frame = play_video_frame(cap)
            if frame is None:
                cap.release()
                cap = None
                current_content = None
                continue
            
            resized_frame = resize_for_device(frame, player_manager.device_info)
            cv2.imshow("Media Player", resized_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    # Cleanup
    player_manager.send_status("offline")
    if player_manager.ws:
        player_manager.ws.close()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Player stopped.")

if __name__ == "__main__":
    main()
