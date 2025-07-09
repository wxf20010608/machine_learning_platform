import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi.logger import logger
import time
import socket


def send_verification_email(to_email, verification_code):
    # 尝试多次发送
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 硬编码QQ邮箱配置，确保正确
            smtp_server = "smtp.qq.com"
            smtp_port = 587  # 改用587端口，使用TLS
            smtp_user = "2140717632@qq.com"  # 您的QQ邮箱
            smtp_password = "emgwncdmfqwyeifc"  # 您的授权码

            logger.info(f"SMTP配置: 服务器={smtp_server}, 端口={smtp_port}, 用户={smtp_user}, 尝试次数={attempt + 1}")

            # 测试网络连接
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((smtp_server, smtp_port))
                sock.close()
                if result != 0:
                    logger.error(f"无法连接到 {smtp_server}:{smtp_port}")
                    return False
                logger.info(f"网络连接测试成功: {smtp_server}:{smtp_port}")
            except Exception as e:
                logger.error(f"网络连接测试失败: {str(e)}")
                return False

            # 创建消息
            message = MIMEMultipart()
            message['From'] = smtp_user
            message['To'] = to_email
            message['Subject'] = "您的验证码"

            # 邮件内容
            body = f"""
            <html>
            <body>
                <h2>验证码</h2>
                <p>您的登录验证码是: <strong>{verification_code}</strong></p>
                <p>此验证码有效期为10分钟。</p>
            </body>
            </html>
            """
            message.attach(MIMEText(body, 'html'))

            # 使用TLS连接到SMTP服务器，设置较短的超时
            logger.info("尝试TLS连接到SMTP服务器...")
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
            
            # 启用TLS
            logger.info("启用TLS加密...")
            server.starttls()
            
            # 设置调试级别（启用调试）
            server.set_debuglevel(1)

            # 登录
            logger.info(f"尝试以{smtp_user}登录...")
            server.login(smtp_user, smtp_password)

            # 发送邮件
            logger.info(f"发送邮件到{to_email}...")
            server.sendmail(smtp_user, to_email, message.as_string())

            # 关闭连接
            server.quit()
            logger.info("邮件发送成功")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP认证失败: {str(e)}")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP连接失败: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"等待2秒后重试...")
                time.sleep(2)
                continue
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP错误: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"等待2秒后重试...")
                time.sleep(2)
                continue
            return False
        except Exception as e:
            logger.error(f"发送邮件时出错: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"等待2秒后重试...")
                time.sleep(2)
                continue
            return False
    
    logger.error(f"经过{max_retries}次尝试后仍然失败")
    return False


def send_verification_email_backup(to_email, verification_code):
    """备用邮件发送函数，使用不同的连接方式"""
    try:
        smtp_server = "smtp.qq.com"
        smtp_port = 587
        smtp_user = "2140717632@qq.com"
        smtp_password = "emgwncdmfqwyeifc"

        logger.info(f"备用SMTP配置: 服务器={smtp_server}, 端口={smtp_port}, 用户={smtp_user}")

        # 创建消息
        message = MIMEMultipart()
        message['From'] = smtp_user
        message['To'] = to_email
        message['Subject'] = "您的验证码"

        body = f"""
        <html>
        <body>
            <h2>验证码</h2>
            <p>您的登录验证码是: <strong>{verification_code}</strong></p>
            <p>此验证码有效期为10分钟。</p>
        </body>
        </html>
        """
        message.attach(MIMEText(body, 'html'))

        # 使用普通SMTP连接
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.ehlo()
        server.starttls()
        server.ehlo()
        
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, message.as_string())
        server.quit()
        
        logger.info("备用邮件发送成功")
        return True

    except Exception as e:
        logger.error(f"备用邮件发送失败: {str(e)}")
        return False


def test_email_connection():
    """测试邮件连接功能"""
    try:
        smtp_server = "smtp.qq.com"
        smtp_port = 587
        smtp_user = "2140717632@qq.com"
        smtp_password = "emgwncdmfqwyeifc"
        
        logger.info("开始测试邮件连接...")
        
        # 测试网络连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((smtp_server, smtp_port))
        sock.close()
        
        if result == 0:
            logger.info("网络连接测试成功")
        else:
            logger.error(f"网络连接失败，错误代码: {result}")
            return False
        
        # 测试SMTP连接
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.quit()
        
        logger.info("SMTP连接测试成功")
        return True
        
    except Exception as e:
        logger.error(f"邮件连接测试失败: {str(e)}")
        return False

