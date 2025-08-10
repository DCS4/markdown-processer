# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import uuid
import asyncio
import aiofiles
import logging
import traceback
from pathlib import Path
import ftfy
from datetime import datetime

# 导入处理器
from improved_markdown_processor import ImprovedMarkdownProcessor
from config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Markdown Text Processor API",
    description="格式感知的Markdown文档处理服务，支持T5纠错和隐私脱敏",
    version="1.0.0"
)

# 全局处理器实例（启动时初始化）
processor: Optional[ImprovedMarkdownProcessor] = None

# 任务状态存储（生产环境建议使用Redis）
task_status: Dict[str, Dict[str, Any]] = {}


# Pydantic模型
class ProcessRequest(BaseModel):
    markdown_content: str
    config: Optional[Dict[str, Any]] = None
    task_name: Optional[str] = None


class ProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str
    chunks_count: Optional[int] = None
    chunks: Optional[List[str]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    uptime: str


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global processor
    try:
        logger.info("正在初始化Markdown处理器...")
        processor = ImprovedMarkdownProcessor()
        await asyncio.get_event_loop().run_in_executor(None, processor.load_models)
        logger.info("Markdown处理器初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise


# 根路径
@app.get("/")
async def root():
    """根路径，提供API信息"""
    return {
        "message": "Markdown Text Processor API",
        "description": "格式感知的Markdown文档处理服务，支持T5纠错和隐私脱敏",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "endpoints": {
            "upload": "/upload-markdown",
            "process_sync": "/process-sync",
            "process_async": "/process-async",
            "task_status": "/task-status/{task_id}"
        }
    }


# 健康检查
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    try:
        models_loaded = processor is not None and processor.corrector is not None
        return HealthResponse(
            status="healthy" if models_loaded else "unhealthy",
            version="1.0.0",
            models_loaded=models_loaded,
            uptime="running"
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="服务不健康")


# 文件上传处理
@app.post("/upload-markdown")
async def upload_markdown_file(file: UploadFile = File(...)):
    """
    上传Markdown文件并返回文件ID，用于后续处理
    """
    try:
        # 验证文件类型
        if not file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="只支持.md文件")

        # 生成唯一ID
        file_id = str(uuid.uuid4())
        file_path = Path(settings.UPLOAD_DIR) / f"{file_id}.md"

        # 确保上传目录存在
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # 保存文件
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"文件上传成功: {file.filename} -> {file_id}")

        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "message": "文件上传成功"
        }

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


# 同步处理（适合小文件）
@app.post("/process-sync", response_model=ProcessResponse)
async def process_markdown_sync(request: ProcessRequest):
    """
    同步处理Markdown内容，适合小文件或实时处理需求
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="处理器未初始化")

    try:
        start_time = datetime.now()
        task_id = str(uuid.uuid4())

        logger.info(f"开始同步处理任务: {task_id}")

        # 获取配置
        config = request.config or settings.DEFAULT_CONFIG

        # 在线程池中执行处理
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            _process_markdown_content,
            request.markdown_content,
            config
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"同步处理完成: {task_id}, 耗时: {processing_time:.2f}s")

        return ProcessResponse(
            task_id=task_id,
            status="completed",
            message="处理完成",
            chunks_count=len(chunks),
            chunks=chunks,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"同步处理失败: {e}")
        return ProcessResponse(
            task_id=task_id,
            status="failed",
            message="处理失败",
            error=str(e)
        )


# 异步处理（适合大文件）
@app.post("/process-async")
async def process_markdown_async(
        background_tasks: BackgroundTasks,
        request: ProcessRequest
):
    """
    异步处理Markdown内容，适合大文件处理
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="处理器未初始化")

    task_id = str(uuid.uuid4())

    # 初始化任务状态
    task_status[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "任务已创建，等待处理",
        "result": None,
        "error": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }

    # 添加后台任务
    background_tasks.add_task(
        _process_markdown_background,
        task_id,
        request.markdown_content,
        request.config or settings.DEFAULT_CONFIG
    )

    logger.info(f"异步任务已创建: {task_id}")

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务已创建，请使用task_id查询处理状态"
    }


# 查询任务状态
@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    查询任务处理状态
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")

    status_data = task_status[task_id]
    return TaskStatus(**status_data)


# 处理上传的文件
@app.post("/process-file/{file_id}")
async def process_uploaded_file(
        file_id: str,
        background_tasks: BackgroundTasks,
        config: Optional[Dict[str, Any]] = None
):
    """
    处理之前上传的文件
    """
    file_path = Path(settings.UPLOAD_DIR) / f"{file_id}.md"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        # 读取文件内容
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # 创建处理请求
        request = ProcessRequest(
            markdown_content=content,
            config=config,
            task_name=f"file_{file_id}"
        )

        # 异步处理
        return await process_markdown_async(background_tasks, request)

    except Exception as e:
        logger.error(f"处理文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理文件失败: {str(e)}")


# Dify专用接口
@app.post("/dify/process")
async def dify_process_markdown(request: ProcessRequest):
    """
    专为Dify优化的接口，返回格式适合工作流集成
    """
    try:
        # 同步处理（Dify通常需要快速响应）
        response = await process_markdown_sync(request)

        # 转换为Dify友好的格式
        if response.status == "completed":
            return {
                "success": True,
                "data": {
                    "chunks": response.chunks,
                    "chunks_count": response.chunks_count,
                    "processing_time": response.processing_time
                },
                "message": "处理成功"
            }
        else:
            return {
                "success": False,
                "error": response.error,
                "message": response.message
            }

    except Exception as e:
        logger.error(f"Dify接口处理失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "处理失败"
        }


# 内部处理函数
def _process_markdown_content(markdown_content: str, config: Dict[str, Any]) -> List[str]:
    """
    内部处理函数，在线程池中执行
    """
    try:
        # 基础清理
        text = ftfy.fix_text(markdown_content)

        # 分割文本
        chunks = processor.split_text(text, config)

        # 处理每个块
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"处理块 {i + 1}/{len(chunks)}")
            processed_chunk = processor.process_chunk(chunk)
            processed_chunks.append(processed_chunk)

        return processed_chunks

    except Exception as e:
        logger.error(f"内部处理失败: {e}")
        raise


async def _process_markdown_background(task_id: str, markdown_content: str, config: Dict[str, Any]):
    """
    后台处理任务
    """
    try:
        # 更新状态
        task_status[task_id].update({
            "status": "processing",
            "progress": 0.1,
            "message": "开始处理",
            "updated_at": datetime.now()
        })

        start_time = datetime.now()

        # 在线程池中执行处理
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            _process_markdown_content,
            markdown_content,
            config
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # 更新完成状态
        task_status[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "处理完成",
            "result": {
                "chunks": chunks,
                "chunks_count": len(chunks),
                "processing_time": processing_time
            },
            "updated_at": datetime.now()
        })

        logger.info(f"后台任务完成: {task_id}")

    except Exception as e:
        error_msg = f"后台处理失败: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        task_status[task_id].update({
            "status": "failed",
            "message": "处理失败",
            "error": error_msg,
            "updated_at": datetime.now()
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)