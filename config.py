# config.py
from pydantic_settings import BaseSettings
from typing import Dict, Any
import os


class Settings(BaseSettings):
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 文件存储配置
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB

    # 模型配置
    MODEL_NAME: str = "deepseek-7b"

    # 默认处理配置
    DEFAULT_CONFIG: Dict[str, Any] = {
        "no_split_threshold": 8000,
        "chunk_size": 7000,
        "chunk_overlap": 500
    }

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT: int = 3600  # 1小时

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"

    # 模型路径（可以通过环境变量覆盖）
    T5_MODEL_PATH: str = os.getenv("T5_MODEL_PATH", "")
    SPACY_MODEL_PATH: str = os.getenv("SPACY_MODEL_PATH", "zh_core_web_md")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()