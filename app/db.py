from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
# from app.config import settings

# 直接使用连接字符串（不依赖config）
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

# 创建数据库引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建模型基类
Base = declarative_base()


# 用户模型
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 验证码相关字段
    verification_code = Column(String, nullable=True)
    verification_code_expires = Column(DateTime, nullable=True)

    # GitHub相关字段
    github_id = Column(String, unique=True, nullable=True)
    github_username = Column(String, nullable=True)
    github_avatar = Column(String, nullable=True)


# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



