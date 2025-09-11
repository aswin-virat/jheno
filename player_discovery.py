# player_discovery.py
import socket
import json
import threading
import time

class PlayerDiscovery:
    def __init__(self, cms_url):
        self.cms_url = cms_url
        self.discovery_port = 8888
        self.running = False
        
    def start_discovery_server(self):
        """Start UDP discovery server"""
        self.running = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.discovery_port))
        
        print(f"Discovery server listening on port {self.discovery_port}")
        
        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                message = json.loads(data.decode())
                
                if message.get('type') == 'player_discovery':
                    # Respond with CMS connection info
                    response = {
                        'type': 'cms_response',
                        'cms_url': self.cms_url,
                        'registration_endpoint': f'{self.cms_url}/players/register'
                    }
                    
                    sock.sendto(json.dumps(response).encode(), addr)
                    print(f"Responded to player discovery from {addr}")
                    
            except Exception as e:
                print(f"Discovery error: {e}")
        
        sock.close()
    
    def broadcast_discovery(self):
        """Broadcast discovery message (for player side)"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        message = {
            'type': 'player_discovery',
            'hostname': socket.gethostname()
        }
        
        try:
            sock.sendto(json.dumps(message).encode(), ('<broadcast>', self.discovery_port))
            
            # Wait for response
            sock.settimeout(5)
            data, addr = sock.recvfrom(1024)
            response = json.loads(data.decode())
            
            if response.get('type') == 'cms_response':
                return response
                
        except socket.timeout:
            print("No CMS found on network")
        except Exception as e:
            print(f"Discovery broadcast error: {e}")
        finally:
            sock.close()
            
        return None