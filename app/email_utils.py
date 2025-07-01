import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi.logger import logger


def send_verification_email(to_email, verification_code):
    try:
        # 硬编码QQ邮箱配置，确保正确
        smtp_server = "smtp.qq.com"
        smtp_port = 465  # QQ邮箱使用SSL连接，端口465
        smtp_user = "2140717632@qq.com"  # 您的QQ邮箱
        smtp_password = "emgwncdmfqwyeifc"  # 您的授权码

        logger.info(f"SMTP配置: 服务器={smtp_server}, 端口={smtp_port}, 用户={smtp_user}")

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

        # 使用SSL连接到SMTP服务器
        logger.info("尝试SSL连接到SMTP服务器...")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

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

    except Exception as e:
        logger.error(f"发送邮件时出错: {str(e)}")
        return False

