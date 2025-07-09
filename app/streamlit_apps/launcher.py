import subprocess
import os
import signal
import sys
import time

def start_streamlit_app(app_name, port):
    script_path = os.path.join(os.path.dirname(__file__), f'{app_name}.py')
    
    # 检查文件是否存在
    if not os.path.exists(script_path):
        print(f"警告: 找不到应用文件 {script_path}")
        return None
        
    # 修改Streamlit启动参数，确保可以通过域名访问
    process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', script_path,
         '--server.port', str(port),
         '--server.address', '0.0.0.0',
         '--server.enableCORS', 'false',
         '--server.enableXsrfProtection', 'false',
         '--browser.gatherUsageStats', 'false',
         '--server.headless', 'true'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    )
    return process

def main():
    # 应用名称与端口映射
    apps = {
        'knn_camera_app_pytorch': 8501,  # 对应/knn
        'kmeans_app': 8502,              # 对应/kmeans
        'linear_regression_app': 8503,   # 对应/linear-regression
        'logistic_regression_app': 8504, # 对应/logistic-regression
        'random_forest_app': 8505        # 对应/random-forest
    }
    
    processes = []
    try:
        for app_name, port in apps.items():
            process = start_streamlit_app(app_name, port)
            if process:
                processes.append(process)
                print(f'Started {app_name} on port {port}')
                time.sleep(1)  # 给每个应用启动一点间隔时间
            else:
                print(f'Failed to start {app_name}')
        
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for process in processes:
            if sys.platform == 'win32':
                process.terminate()
            else:
                os.kill(process.pid, signal.SIGTERM)
        print('\nAll Streamlit apps have been stopped')
