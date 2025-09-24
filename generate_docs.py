#!/usr/bin/env python3
"""
API文档生成脚本

生成静态HTML版本的API文档，包括：
1. 从OpenAPI JSON生成HTML
2. 导出现有的Markdown文档为HTML
"""

import json
import os
import sys
from pathlib import Path
import markdown
from jinja2 import Template

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_openapi_html():
    """从OpenAPI JSON生成HTML文档"""
    try:
        import requests
        
        # 启动服务器并获取OpenAPI JSON
        base_url = "http://localhost:8000"
        openapi_url = f"{base_url}/openapi.json"
        
        print("正在获取OpenAPI JSON...")
        response = requests.get(openapi_url, timeout=10)
        
        if response.status_code == 200:
            openapi_spec = response.json()
            
            # 保存OpenAPI JSON
            with open("docs/openapi.json", "w", encoding="utf-8") as f:
                json.dump(openapi_spec, f, ensure_ascii=False, indent=2)
            
            print("✅ OpenAPI JSON已保存到 docs/openapi.json")
            return True
        else:
            print(f"❌ 获取OpenAPI JSON失败: {response.status_code}")
            return False
            
    except ImportError:
        print("❌ 需要安装requests库: pip install requests")
        return False
    except Exception as e:
        print(f"❌ 生成OpenAPI文档时出错: {e}")
        return False

def generate_markdown_html():
    """将Markdown文档转换为HTML"""
    try:
        # 读取Markdown文档
        md_file = Path("docs/API.md")
        if not md_file.exists():
            print("❌ docs/API.md 文件不存在")
            return False
            
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # 转换为HTML
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite',
                'markdown.extensions.toc'
            ]
        )
        
        # 创建HTML模板
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>机器学习平台 API 文档</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }
        pre code {
            background: none;
            color: inherit;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #7f8c8d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #7f8c8d;
        }
        .nav-links {
            margin-bottom: 20px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 6px;
        }
        .nav-links a {
            color: #3498db;
            text-decoration: none;
            margin-right: 20px;
        }
        .nav-links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav-links">
            <a href="#overview">概述</a>
            <a href="#auth">认证</a>
            <a href="#scan">扫描</a>
            <a href="#ml">机器学习</a>
            <a href="#health">健康检查</a>
            <a href="http://localhost:8000/docs" target="_blank">Swagger UI</a>
            <a href="http://localhost:8000/redoc" target="_blank">ReDoc</a>
        </div>
        
        {{ content }}
        
        <div class="footer">
            <p>机器学习平台 API 文档 | 生成时间: {{ timestamp }}</p>
            <p>交互式文档: <a href="http://localhost:8000/docs">Swagger UI</a> | <a href="http://localhost:8000/redoc">ReDoc</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        # 渲染HTML
        template = Template(html_template)
        html_output = template.render(
            content=html_content,
            timestamp=__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 保存HTML文件
        output_file = Path("docs/API.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_output)
        
        print("✅ HTML文档已生成: docs/API.html")
        return True
        
    except ImportError:
        print("❌ 需要安装markdown和jinja2库:")
        print("   pip install markdown jinja2")
        return False
    except Exception as e:
        print(f"❌ 生成HTML文档时出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始生成API文档...")
    
    # 确保docs目录存在
    Path("docs").mkdir(exist_ok=True)
    
    success_count = 0
    
    # 生成Markdown转HTML
    if generate_markdown_html():
        success_count += 1
    
    # 生成OpenAPI文档（需要服务器运行）
    print("\n📡 尝试获取OpenAPI文档（需要服务器运行在localhost:8000）...")
    if generate_openapi_html():
        success_count += 1
    else:
        print("💡 提示: 请先启动服务器 (python main.py) 后再运行此脚本获取OpenAPI文档")
    
    print(f"\n✨ 文档生成完成！成功生成 {success_count}/2 种格式")
    print("\n📖 可用的文档:")
    print("   - docs/API.html (静态HTML)")
    print("   - docs/API.md (Markdown)")
    if Path("docs/openapi.json").exists():
        print("   - docs/openapi.json (OpenAPI规范)")
    print("\n🌐 在线文档 (需要服务器运行):")
    print("   - http://localhost:8000/docs (Swagger UI)")
    print("   - http://localhost:8000/redoc (ReDoc)")

if __name__ == "__main__":
    main()
