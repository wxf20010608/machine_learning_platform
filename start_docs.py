#!/usr/bin/env python3
"""
文档服务器启动脚本

提供便捷的文档服务启动方式，包括：
1. 启动FastAPI服务器
2. 自动打开浏览器显示文档
3. 提供文档访问链接
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    required_packages = ['fastapi', 'uvicorn', 'requests', 'markdown', 'jinja2']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少必要的依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def start_server():
    """启动FastAPI服务器"""
    print("🚀 启动FastAPI服务器...")
    
    try:
        # 启动服务器
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务器启动
        time.sleep(3)
        
        # 检查服务器是否正常启动
        if process.poll() is None:
            print("✅ 服务器启动成功！")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ 服务器启动失败:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ 启动服务器时出错: {e}")
        return None

def open_docs():
    """打开文档页面"""
    print("\n📖 正在打开文档页面...")
    
    docs_urls = [
        ("Swagger UI", "http://localhost:8000/docs"),
        ("ReDoc", "http://localhost:8000/redoc"),
        ("静态HTML文档", str(Path("docs/API.html").absolute()))
    ]
    
    print("\n🌐 可用的文档链接:")
    for name, url in docs_urls:
        print(f"   {name}: {url}")
    
    # 尝试打开浏览器
    try:
        # 首先尝试打开Swagger UI
        webbrowser.open("http://localhost:8000/docs")
        print("\n✅ 已自动打开Swagger UI")
        
        # 询问是否打开其他文档
        choice = input("\n是否同时打开ReDoc文档? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            webbrowser.open("http://localhost:8000/redoc")
            print("✅ 已打开ReDoc文档")
            
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print("请手动访问上述链接")

def show_usage_info():
    """显示使用说明"""
    print("\n" + "="*60)
    print("📚 机器学习平台 API 文档")
    print("="*60)
    print("\n🎯 主要功能:")
    print("   • 用户认证 (邮箱验证码 + GitHub OAuth)")
    print("   • 文档扫描与图像处理")
    print("   • 机器学习算法 (KNN, K-means, 回归, 随机森林)")
    print("   • 数据可视化 (Streamlit应用)")
    print("   • 健康检查与监控")
    
    print("\n🔧 API测试:")
    print("   1. 在Swagger UI中可以直接测试API")
    print("   2. 大部分API需要JWT Token认证")
    print("   3. 先通过 /auth/send-verification-code 获取验证码")
    print("   4. 再通过 /auth/login 登录获取token")
    
    print("\n📋 文档格式:")
    print("   • 交互式: Swagger UI, ReDoc")
    print("   • 静态: HTML, Markdown, OpenAPI JSON")
    
    print("\n🛠️  开发工具集成:")
    print("   • Postman: 导入 http://localhost:8000/openapi.json")
    print("   • Insomnia: 导入OpenAPI规范")
    print("   • VS Code: 使用REST Client插件")

def main():
    """主函数"""
    print("🚀 机器学习平台文档服务器")
    print("="*40)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查main.py是否存在
    if not Path("main.py").exists():
        print("❌ 找不到main.py文件，请确保在项目根目录运行此脚本")
        return
    
    # 显示使用说明
    show_usage_info()
    
    # 询问是否启动服务器
    choice = input("\n是否启动服务器并打开文档? (y/n): ").lower().strip()
    if choice not in ['y', 'yes', '是']:
        print("👋 再见！")
        return
    
    # 启动服务器
    server_process = start_server()
    if not server_process:
        print("❌ 无法启动服务器，请检查错误信息")
        return
    
    try:
        # 打开文档
        open_docs()
        
        print("\n" + "="*60)
        print("✅ 文档服务器已启动！")
        print("📝 按 Ctrl+C 停止服务器")
        print("="*60)
        
        # 保持服务器运行
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止服务器...")
        server_process.terminate()
        server_process.wait()
        print("✅ 服务器已停止")

if __name__ == "__main__":
    main()
