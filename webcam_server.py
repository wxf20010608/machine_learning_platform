#!/usr/bin/env python3
"""
简单的网络摄像头服务器
用于在本地启动摄像头服务，供Docker容器访问
"""

import cv2
import threading
import time
from flask import Flask, Response, render_template_string
import argparse

app = Flask(__name__)

# 全局变量
camera = None
frame_buffer = None
lock = threading.Lock()

def get_camera_frame():
    """获取摄像头帧"""
    global camera, frame_buffer
    
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if ret:
            with lock:
                frame_buffer = frame.copy()
        time.sleep(0.033)  # 约30fps

def generate_frames():
    """生成视频流"""
    global frame_buffer
    
    while True:
        if frame_buffer is not None:
            with lock:
                frame = frame_buffer.copy()
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    """主页"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>网络摄像头服务器</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
            .info { background-color: #d1ecf1; color: #0c5460; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>网络摄像头服务器</h1>
            <div class="status success">
                <strong>状态:</strong> 运行中
            </div>
            <div class="status info">
                <strong>摄像头URL:</strong> <code>http://localhost:8080/video</code>
            </div>
            <div class="status info">
                <strong>使用说明:</strong> 在Streamlit应用中输入上述URL作为摄像头源
            </div>
            <h2>实时预览</h2>
            <img src="/video" alt="摄像头预览" />
        </div>
    </body>
    </html>
    """
    return html

@app.route('/video')
def video_feed():
    """视频流端点"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """状态检查端点"""
    global camera
    if camera is not None and camera.isOpened():
        return {"status": "running", "camera": "connected"}
    else:
        return {"status": "error", "camera": "disconnected"}

def main():
    global camera
    
    parser = argparse.ArgumentParser(description='启动网络摄像头服务器')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口 (默认: 8080)')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print(f"正在启动网络摄像头服务器...")
    print(f"摄像头设备: {args.camera}")
    print(f"服务器地址: http://{args.host}:{args.port}")
    
    # 初始化摄像头
    try:
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            print(f"错误: 无法打开摄像头 {args.camera}")
            return
        
        print("摄像头初始化成功!")
        
        # 启动摄像头线程
        camera_thread = threading.Thread(target=get_camera_frame, daemon=True)
        camera_thread.start()
        
        # 启动Flask服务器
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
    except Exception as e:
        print(f"启动服务器时出错: {e}")
    finally:
        if camera is not None:
            camera.release()
        print("服务器已关闭")

if __name__ == '__main__':
    main() 