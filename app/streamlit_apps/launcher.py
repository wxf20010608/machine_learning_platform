import subprocess
import os
import signal
import sys
import time
import logging
import threading
from datetime import datetime
import requests

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('streamlit_launcher.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class StreamlitProcessManager:
    def __init__(self):
        self.processes = {}
        self.apps_config = {
            'knn_camera_app_pytorch': 8501,
            'kmeans_app': 8502,
            'linear_regression_app': 8503,
            'logistic_regression_app': 8504,
            'random_forest_app': 8505
        }
        self.restart_enabled = True
        self.health_check_interval = 30  # 30秒检查一次
        
    def start_streamlit_app(self, app_name, port):
        """启动单个Streamlit应用"""
        script_path = os.path.join(os.path.dirname(__file__), f'{app_name}.py')
        
        # 检查文件是否存在
        if not os.path.exists(script_path):
            logging.warning(f"找不到应用文件 {script_path}")
            return None
            
        try:
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
            
            # 存储进程信息
            self.processes[app_name] = {
                'process': process,
                'port': port,
                'start_time': datetime.now(),
                'restart_count': 0,
                'status': 'starting'
            }
            
            logging.info(f'启动 {app_name} (端口 {port})')
            return process
            
        except Exception as e:
            logging.error(f"启动 {app_name} 失败: {e}")
            return None
    
    def check_process_health(self, app_name):
        """检查进程健康状态"""
        if app_name not in self.processes:
            return False
            
        process_info = self.processes[app_name]
        process = process_info['process']
        port = process_info['port']
        
        # 检查进程是否还在运行
        if process.poll() is not None:
            logging.warning(f"{app_name} 进程已退出 (退出码: {process.returncode})")
            return False
        
        # 检查HTTP健康状态
        try:
            response = requests.get(f'http://localhost:{port}/_stcore/health', timeout=5)
            if response.status_code == 200:
                process_info['status'] = 'healthy'
                return True
            else:
                logging.warning(f"{app_name} HTTP健康检查失败: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            # 应用可能还在启动中
            if process_info['status'] == 'starting':
                startup_time = (datetime.now() - process_info['start_time']).seconds
                if startup_time < 60:  # 给60秒启动时间
                    return True
            logging.warning(f"{app_name} HTTP连接失败: {e}")
            return False
    
    def restart_app(self, app_name):
        """重启应用"""
        if app_name not in self.processes:
            return False
            
        process_info = self.processes[app_name]
        process_info['restart_count'] += 1
        
        # 如果重启次数过多，停止自动重启
        if process_info['restart_count'] > 5:
            logging.error(f"{app_name} 重启次数过多，停止自动重启")
            return False
        
        logging.info(f"重启 {app_name} (第 {process_info['restart_count']} 次)")
        
        # 停止旧进程
        old_process = process_info['process']
        try:
            if sys.platform == 'win32':
                old_process.terminate()
            else:
                os.kill(old_process.pid, signal.SIGTERM)
        except Exception as e:
            logging.warning(f"停止旧进程失败: {e}")
        
        # 启动新进程
        port = process_info['port']
        new_process = self.start_streamlit_app(app_name, port)
        return new_process is not None
    
    def monitor_processes(self):
        """监控所有进程的健康状态"""
        while self.restart_enabled:
            for app_name in list(self.processes.keys()):
                if not self.check_process_health(app_name):
                    if self.restart_enabled:
                        self.restart_app(app_name)
            
            time.sleep(self.health_check_interval)
    
    def start_all_apps(self):
        """启动所有应用"""
        for app_name, port in self.apps_config.items():
            process = self.start_streamlit_app(app_name, port)
            if process:
                logging.info(f'成功启动 {app_name} (端口 {port})')
                time.sleep(2)  # 给每个应用启动间隔时间
            else:
                logging.error(f'启动 {app_name} 失败')
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        logging.info("进程监控已启动")
    
    def stop_all_apps(self):
        """停止所有应用"""
        self.restart_enabled = False
        logging.info("正在停止所有应用...")
        
        for app_name, process_info in self.processes.items():
            try:
                process = process_info['process']
                if sys.platform == 'win32':
                    process.terminate()
                else:
                    os.kill(process.pid, signal.SIGTERM)
                logging.info(f'已停止 {app_name}')
            except Exception as e:
                logging.error(f'停止 {app_name} 失败: {e}')
        
        logging.info('所有Streamlit应用已停止')

def start_streamlit_app(app_name, port):
    """保持向后兼容的函数"""
    manager = StreamlitProcessManager()
    return manager.start_streamlit_app(app_name, port)

def main():
    """主函数，使用新的进程管理器"""
    manager = StreamlitProcessManager()
    
    try:
        logging.info("启动Streamlit应用管理器...")
        manager.start_all_apps()
        
        # 保持脚本运行
        logging.info("所有应用已启动，进入监控模式...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在关闭所有应用...")
        manager.stop_all_apps()

if __name__ == "__main__":
    main()
