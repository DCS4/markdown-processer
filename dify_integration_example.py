# dify_integration_example.py
"""
Dify工作流集成示例
展示如何在Dify中调用Markdown处理服务
"""

import requests
import json
from typing import Dict, List, Any


class MarkdownProcessorClient:
    """Markdown处理服务客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def process_markdown_sync(self, markdown_content: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步处理Markdown内容"""
        payload = {
            "markdown_content": markdown_content,
            "config": config,
            "task_name": "dify_task"
        }

        response = requests.post(
            f"{self.base_url}/dify/process",
            json=payload,
            timeout=300  # 5分钟超时
        )

        return response.json()

    def process_markdown_async(self, markdown_content: str, config: Dict[str, Any] = None) -> str:
        """异步处理Markdown内容，返回任务ID"""
        payload = {
            "markdown_content": markdown_content,
            "config": config,
            "task_name": "dify_async_task"
        }

        response = requests.post(
            f"{self.base_url}/process-async",
            json=payload
        )

        result = response.json()
        return result.get("task_id")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """查询任务状态"""
        response = requests.get(f"{self.base_url}/task/{task_id}")
        return response.json()

    def wait_for_completion(self, task_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """等待任务完成"""
        import time

        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_task_status(task_id)

            if status["status"] == "completed":
                return status["result"]
            elif status["status"] == "failed":
                raise Exception(f"任务失败: {status.get('error', '未知错误')}")

            time.sleep(2)  # 等待2秒后重试

        raise TimeoutError(f"任务超时: {task_id}")


# Dify工作流中的使用示例
def dify_workflow_example():
    """
    在Dify工作流中使用的示例代码
    """

    # 1. 初始化客户端
    client = MarkdownProcessorClient("http://markdown-processor:8000")

    # 2. 检查服务健康状态
    try:
        health = client.health_check()
        if health["status"] != "healthy":
            return {"error": "Markdown处理服务不可用"}
    except Exception as e:
        return {"error": f"无法连接到Markdown处理服务: {str(e)}"}

    # 3. 处理Markdown内容（这里假设从前一个节点获得）
    markdown_content = """
    # 示例文档

    这是一个示例文档，包含了一些需要纠错的文本内容。
    这里可能有一些语法错误或者需要隐私脱敏的信息。

    ## 联系信息
    姓名：张三
    电话：13800138000
    邮箱：zhangsan@example.com
    """

    # 4. 自定义配置（可选）
    config = {
        "no_split_threshold": 5000,
        "chunk_size": 4000,
        "chunk_overlap": 200
    }

    try:
        # 5. 同步处理（适合小文档）
        result = client.process_markdown_sync(markdown_content, config)

        if result["success"]:
            chunks = result["data"]["chunks"]
            chunks_count = result["data"]["chunks_count"]

            return {
                "success": True,
                "chunks": chunks,
                "chunks_count": chunks_count,
                "message": f"处理完成，生成{chunks_count}个文本块"
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "message": "处理失败"
            }

    except requests.exceptions.Timeout:
        # 6. 如果同步处理超时，可以切换到异步处理
        try:
            task_id = client.process_markdown_async(markdown_content, config)
            result = client.wait_for_completion(task_id, max_wait=300)

            return {
                "success": True,
                "chunks": result["chunks"],
                "chunks_count": result["chunks_count"],
                "message": f"异步处理完成，生成{result['chunks_count']}个文本块"
            }

        except Exception as async_error:
            return {
                "success": False,
                "error": str(async_error),
                "message": "异步处理也失败了"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "处理过程中发生错误"
        }


# Dify HTTP节点配置示例
DIFY_HTTP_NODE_CONFIG = {
    "method": "POST",
    "url": "http://markdown-processor:8000/dify/process",
    "headers": {
        "Content-Type": "application/json"
    },
    "body": {
        "markdown_content": "{{previous_node.markdown_content}}",
        "config": {
            "no_split_threshold": 8000,
            "chunk_size": 7000,
            "chunk_overlap": 500
        },
        "task_name": "{{workflow.id}}_{{node.id}}"
    },
    "timeout": 300
}


# 完整的处理流水线示例（从PDF到处理完成）
def complete_pipeline_example():
    """
    完整的处理流水线：PDF -> MinerU -> Markdown处理
    """

    # 假设的MinerU服务客户端
    mineru_client = "http://mineru-service:8001"
    markdown_client = MarkdownProcessorClient("http://markdown-processor:8000")

    def process_pdf_to_chunks(pdf_file_path: str) -> Dict[str, Any]:
        try:
            # 1. 调用MinerU转换PDF
            with open(pdf_file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{mineru_client}/convert-pdf", files=files)

            if response.status_code != 200:
                return {"success": False, "error": "PDF转换失败"}

            markdown_content = response.json()["markdown_content"]

            # 2. 调用Markdown处理服务
            result = markdown_client.process_markdown_sync(markdown_content)

            if result["success"]:
                return {
                    "success": True,
                    "chunks": result["data"]["chunks"],
                    "chunks_count": result["data"]["chunks_count"],
                    "processing_time": result["data"]["processing_time"],
                    "pipeline": "PDF -> MinerU -> Markdown处理"
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "stage": "markdown_processing"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "pipeline_error"
            }

    return process_pdf_to_chunks


if __name__ == "__main__":
    # 测试示例
    client = MarkdownProcessorClient()

    # 测试健康检查
    print("健康检查:", client.health_check())

    # 测试处理
    test_markdown = """
    # 测试文档

    这是一个测试文档，用于验证处理功能。
    联系人：李四，电话：15800000000
    """

    result = client.process_markdown_sync(test_markdown)
    print("处理结果:", json.dumps(result, ensure_ascii=False, indent=2))