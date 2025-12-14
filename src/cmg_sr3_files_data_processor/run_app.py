#!/usr/bin/env python3
"""
Standalone execution script for SR3 Data Processor Web Application
Configurable server settings for localhost or network access
"""

import argparse
import subprocess
import sys
import os
import time
import signal
import socket
from pathlib import Path

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def check_ngrok_available():
    """Check if ngrok is available (via pyngrok or system installation)"""
    try:
        import pyngrok
        return True, 'pyngrok'
    except ImportError:
        try:
            result = subprocess.run(['ngrok', 'version'], 
                                  capture_output=True, 
                                  timeout=2,
                                  check=False)
            if result.returncode == 0:
                return True, 'system'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False, None

def start_ngrok_tunnel(port):
    """Start ngrok tunnel and return public URL"""
    try:
        import pyngrok.conf
        from pyngrok import ngrok
        
        # Check if ngrok is already authenticated
        # If not, pyngrok will download and setup ngrok, but may require auth token
        try:
            # Start tunnel
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url
            
            return tunnel, public_url
        except Exception as e:
            # Check if it's an authentication error
            error_msg = str(e).lower()
            if 'auth' in error_msg or 'token' in error_msg or 'authtoken' in error_msg:
                print("\n⚠️  ngrok authentication required!")
                print("   To use ngrok tunneling, you need to:")
                print("   1. Sign up at https://dashboard.ngrok.com/signup")
                print("   2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
                print("   3. Run: ngrok config add-authtoken YOUR_TOKEN")
                print("\n   Or set it programmatically:")
                print("   from pyngrok import ngrok")
                print("   ngrok.set_auth_token('YOUR_TOKEN')")
                return None, None
            raise
    except ImportError:
        # Fallback to system ngrok if pyngrok not available
        try:
            process = subprocess.Popen(
                ['ngrok', 'http', str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(2)  # Wait for ngrok to start
            
            # Try to get URL from ngrok API
            try:
                import urllib.request
                import json
                response = urllib.request.urlopen('http://127.0.0.1:4040/api/tunnels', timeout=2)
                data = json.loads(response.read().decode())
                if data.get('tunnels'):
                    public_url = data['tunnels'][0]['public_url']
                    return process, public_url
            except:
                pass
            
            # If API doesn't work, return process handle
            # User will need to check ngrok web interface
            return process, None
        except FileNotFoundError:
            return None, None
    except Exception as e:
        return None, None

def stop_ngrok_tunnel(tunnel):
    """Stop ngrok tunnel"""
    try:
        if hasattr(tunnel, 'disconnect'):
            # pyngrok tunnel object
            tunnel.disconnect()
        elif hasattr(tunnel, 'terminate'):
            # subprocess process object
            tunnel.terminate()
            tunnel.wait()
    except:
        pass

def _is_port_available(host: str, port: int) -> bool:
    """
    Best-effort check if we can bind to (host, port) on this machine.

    On Windows, Streamlit failing with "Port XXXX is already in use" is common when
    a previous server instance is still running. We try to pick the next available
    port automatically instead of failing hard.
    """
    # "localhost" can resolve to IPv6 on some machines; binding to 127.0.0.1 is a
    # reliable indicator for local-only availability.
    bind_host = '127.0.0.1' if host == 'localhost' else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((bind_host, port))
        return True
    except OSError:
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='SR3 Data Processor Web Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on localhost only (default)
  python run_app.py
  
  # Run on localhost with custom port
  python run_app.py --port 8502
  
  # Run on network (accessible from other machines on your network)
  python run_app.py --host 0.0.0.0
  
  # Enable internet tunneling (public URL for internet access)
  python run_app.py --tunnel
  
  # Enable tunneling with custom port
  python run_app.py --tunnel --port 8502
  
  # Run in headless mode
  python run_app.py --server-headless
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind to. Use "0.0.0.0" for network access, "localhost" for local only (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to bind to (default: 8501)'
    )
    
    parser.add_argument(
        '--server-headless',
        action='store_true',
        help='Run in headless mode (no browser auto-open)'
    )
    
    parser.add_argument(
        '--tunnel',
        action='store_true',
        help='Enable internet tunneling (creates public URL for internet access)'
    )
    
    parser.add_argument(
        '--tunnel-service',
        type=str,
        default='ngrok',
        choices=['ngrok'],
        help='Tunneling service to use (default: ngrok)'
    )
    
    args = parser.parse_args()

    # If the requested port is already in use, auto-select the next free one.
    requested_port = args.port
    if not _is_port_available(args.host, args.port):
        for candidate in range(args.port + 1, args.port + 51):
            if _is_port_available(args.host, candidate):
                args.port = candidate
                break

        if args.port != requested_port:
            print(f"⚠️  Port {requested_port} is already in use. Switching to port {args.port}.")
            print("   Tip: you can force a port via `python run_app.py --port <PORT>`")
            print()
        else:
            print(f"❌ Error: Port {requested_port} is already in use and no free port was found nearby.")
            print("   Close the existing server or pass a different port with --port.")
            sys.exit(1)
    
    # Ensure we're in the script's directory for portable execution
    # This allows the app to work regardless of where the folder is located
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    # Check if app.py exists
    app_file = script_dir / 'app.py'
    if not app_file.exists():
        print(f"❌ Error: app.py not found at {app_file}")
        sys.exit(1)
    
    # Verify that app.py can be imported (early error detection)
    # Add script directory to sys.path for import verification
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    try:
        # Test import to catch path issues early
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", app_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load app.py from {app_file}")
    except Exception as e:
        print(f"❌ Error: Cannot import app.py: {e}")
        print(f"   Script directory: {script_dir}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}...")
        sys.exit(1)
    
    # Handle tunneling mode
    tunnel = None
    tunnel_url = None
    if args.tunnel:
        # Force localhost mode when using tunnel
        args.host = 'localhost'
        
        # Check if tunneling service is available
        if args.tunnel_service == 'ngrok':
            available, method = check_ngrok_available()
            if not available:
                print("❌ Error: ngrok is not available")
                print("\nTo enable internet tunneling, please install ngrok:")
                print("  Option 1 (Recommended): pip install pyngrok")
                print("  Option 2: Download from https://ngrok.com/download")
                print("           and add to PATH")
                sys.exit(1)
            
            print("🔗 Starting ngrok tunnel...")
            tunnel, tunnel_url = start_ngrok_tunnel(args.port)
            
            if tunnel and tunnel_url:
                print(f"✅ Tunnel established!")
                print(f"🌐 Public URL: {tunnel_url}")
                print()
            elif tunnel:
                print("⚠️  Tunnel started but URL unavailable")
                print("   Check ngrok web interface at http://127.0.0.1:4040")
                print()
            else:
                print("❌ Failed to start tunnel")
                sys.exit(1)
    
    # Build streamlit command
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        str(app_file),
        '--server.address',
        args.host,
        '--server.port',
        str(args.port)
    ]
    
    if args.server_headless:
        cmd.extend(['--server.headless', 'true'])
    
    # Security warnings
    if args.tunnel:
        print("=" * 60)
        print("⚠️  SECURITY WARNING: Internet Tunneling Enabled")
        print("=" * 60)
        print("   Your application is now publicly accessible on the internet!")
        print("   - No authentication is enabled (open access)")
        print("   - Anybody with the URL can access your application")
        print("   - Use only for testing/demo purposes")
        print("   - Stop the server when done testing")
        print("=" * 60)
        print()
    elif args.host == '0.0.0.0':
        print("⚠️  WARNING: Running in network mode (0.0.0.0)")
        print("   The application will be accessible from other machines on your network.")
        print("   Make sure your firewall is properly configured.")
        print()
    
    # Display startup information
    print("=" * 60)
    print("🛢️  SR3 Data Processor Web Application")
    print("=" * 60)
    
    if args.tunnel and tunnel_url:
        print(f"🌐 Public URL: {tunnel_url}")
        print(f"📍 Local URL: http://localhost:{args.port}")
        print(f"🔒 Access: Internet (via tunnel)")
    else:
        local_url = f"http://{args.host}:{args.port}"
        print(f"🌐 Local Web Page: {local_url}")
        if args.host == 'localhost' or args.host == '127.0.0.1':
            print(f"🔒 Access: Localhost only (browser will open automatically)")
        else:
            print(f"🔒 Access: Network (all interfaces)")
            print(f"⚠️  Available from: http://{args.host}:{args.port}")
    
    print("=" * 60)
    print()
    if not args.server_headless and (args.host == 'localhost' or args.host == '127.0.0.1'):
        print("🚀 Starting Streamlit server...")
        print("🌐 Browser will open automatically in a few seconds")
    else:
        print("🚀 Starting Streamlit server...")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Setup signal handler for cleanup
    def cleanup_handler(signum, frame):
        if tunnel:
            print("\n\n🛑 Stopping tunnel...")
            stop_ngrok_tunnel(tunnel)
        print("👋 Server stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, cleanup_handler)
    
    try:
        # Run streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        cleanup_handler(None, None)
    except subprocess.CalledProcessError as e:
        if tunnel:
            stop_ngrok_tunnel(tunnel)
        print(f"\n❌ Error running Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        if tunnel:
            stop_ngrok_tunnel(tunnel)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if tunnel:
            stop_ngrok_tunnel(tunnel)

if __name__ == "__main__":
    main()

