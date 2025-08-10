# improved_markdown_processor.py

import os
import re
import ftfy
from typing import List, Tuple
import logging

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Presidio
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# PyCorrector
from pycorrector import T5Corrector

# Markdown处理
from markdown_it import MarkdownIt
from markdown_it.token import Token

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedMarkdownProcessor:
    def __init__(self):
        self.md_parser = MarkdownIt()
        self.corrector = None
        self.analyzer = None
        self.anonymizer = None

    def load_models(self):
        """加载所需的模型和引擎"""
        try:
            logger.info("正在加载T5纠错模型...")
            # 设置环境变量指向本地模型缓存
            import os
            os.environ['HF_HOME'] = '/app/model-cache'
            os.environ['TRANSFORMERS_CACHE'] = '/app/model-cache'
            os.environ['HF_DATASETS_CACHE'] = '/app/model-cache'
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 使用本地缓存的模型初始化T5Corrector
            self.corrector = T5Corrector()
            logger.info("T5模型加载成功")

            logger.info("正在初始化Presidio...")
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "zh", "model_name": "zh_core_web_md"}]
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["zh"])
            self.anonymizer = AnonymizerEngine()
            logger.info("Presidio初始化成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def extract_text_segments(self, text: str) -> List[Tuple[str, bool]]:
        """
        改进的文本提取：返回(文本片段, 是否需要纠错)的元组列表
        """
        tokens = self.md_parser.parse(text)
        segments = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 处理段落内容
            if token.type == 'paragraph_open':
                # 查找对应的inline token
                if i + 1 < len(tokens) and tokens[i + 1].type == 'inline':
                    content = tokens[i + 1].content
                    if content.strip():
                        segments.append((content, True))  # 需要纠错
                i += 2  # 跳过paragraph_open和inline
                continue

            # 处理标题
            elif token.type in ['heading_open']:
                if i + 1 < len(tokens) and tokens[i + 1].type == 'inline':
                    content = tokens[i + 1].content
                    level = token.tag[1]  # h1, h2, etc.
                    formatted_content = f"{'#' * int(level)} {content}"
                    segments.append((formatted_content, False))  # 不纠错
                i += 2
                continue

            # 处理代码块
            elif token.type == 'code_block':
                segments.append((f"```\n{token.content}```", False))  # 不纠错

            # 处理列表项
            elif token.type == 'list_item_open':
                # 简化处理：直接添加原始内容
                if i + 1 < len(tokens):
                    segments.append((tokens[i + 1].content or "", False))

            i += 1

        return segments

    def safe_correct_text(self, text: str) -> str:
        """
        安全的文本纠错，添加错误处理
        """
        try:
            if not text.strip():
                return text

            result = self.corrector.correct(text)
            corrected = result.get('target', text)

            # 基本的质量检查
            if len(corrected) < len(text) * 0.5:  # 如果纠错后长度过短，可能有问题
                logger.warning(f"纠错结果异常短，使用原文: {text[:50]}...")
                return text

            return corrected

        except Exception as e:
            logger.error(f"文本纠错失败: {e}")
            return text

    def anonymize_text(self, text: str) -> str:
        """
        对文本进行脱敏处理
        """
        try:
            results = self.analyzer.analyze(text=text, language="zh")
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "PERSON": OperatorConfig("replace", {"new_value": "[姓名]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[电话号码]"}),
                    "ID": OperatorConfig("replace", {"new_value": "[身份证号]"}),
                    "LOCATION": OperatorConfig("replace", {"new_value": "[地址]"}),
                    "NRP": OperatorConfig("replace", {"new_value": "[组织名]"}),
                    "DATE_TIME": OperatorConfig("keep"),
                    "EMAIL_ADDRESS": OperatorConfig("mask",
                                                    {"masking_char": "*", "chars_to_mask": 4, "from_end": True}),
                }
            ).text
            return anonymized
        except Exception as e:
            logger.error(f"文本脱敏失败: {e}")
            return text

    def process_text_segments(self, segments: List[Tuple[str, bool]]) -> str:
        """
        处理文本片段列表
        """
        processed_segments = []

        for segment, should_correct in segments:
            if should_correct:
                # 先纠错再脱敏
                corrected = self.safe_correct_text(segment)
                anonymized = self.anonymize_text(corrected)
                processed_segments.append(anonymized)
            else:
                # 格式化内容直接脱敏（但通常格式内容不包含敏感信息）
                processed_segments.append(segment)

        return '\n\n'.join(processed_segments)

    def process_chunk(self, chunk_text: str) -> str:
        """
        处理单个文本块
        """
        try:
            # 提取文本片段
            segments = self.extract_text_segments(chunk_text)

            # 处理片段
            processed_text = self.process_text_segments(segments)

            return processed_text

        except Exception as e:
            logger.error(f"处理文本块失败: {e}")
            # 降级处理：直接对整个chunk进行简单处理
            try:
                corrected = self.safe_correct_text(chunk_text)
                return self.anonymize_text(corrected)
            except:
                return chunk_text

    def split_text(self, text: str, config: dict) -> List[str]:
        """
        智能文本分割
        """
        if len(text) < config["no_split_threshold"]:
            logger.info(f"文档长度 {len(text)} 小于阈值，无需分割")
            return [text]

        logger.info(f"文档长度 {len(text)} 大于阈值，启动分割")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", "### ", "## ", "# ", "。", "，", " ", ""]
        )

        chunks = text_splitter.split_text(text)
        logger.info(f"分割完成，共 {len(chunks)} 个块")
        return chunks

    def process_file(self, input_path: str, output_dir: str, config: dict):
        """
        处理单个文件
        """
        filename = os.path.basename(input_path)
        logger.info(f"开始处理文件: {filename}")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            # 基础清理
            text = ftfy.fix_text(raw_text)

            # 分割文本
            chunks = self.split_text(text, config)

            # 处理每个块
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"处理块 {i + 1}/{len(chunks)}")
                processed_chunk = self.process_chunk(chunk)
                processed_chunks.append(processed_chunk)

            # 保存结果
            base_filename = os.path.splitext(filename)[0]
            if len(processed_chunks) == 1:
                output_filename = f"{base_filename}_processed.md"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_chunks[0])
            else:
                for i, chunk in enumerate(processed_chunks):
                    output_filename = f"{base_filename}_chunk_{i + 1:03d}.md"
                    output_path = os.path.join(output_dir, output_filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)

            logger.info(f"文件 {filename} 处理完成")

        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {e}")


def main():
    """主函数"""

    # 配置
    MODEL_CONFIGS = {
        "deepseek-7b": {
            "no_split_threshold": 8000,
            "chunk_size": 7000,
            "chunk_overlap": 500
        },
    }
    CURRENT_MODEL = "deepseek-7b"
    config = MODEL_CONFIGS[CURRENT_MODEL]

    INPUT_FOLDER = "input_markdowns"
    OUTPUT_FOLDER = f"output_chunks_{CURRENT_MODEL}_improved"

    # 创建文件夹
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 检查输入文件
    md_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.md')]
    if not md_files:
        logger.info(f"在 '{INPUT_FOLDER}' 中未找到 .md 文件")
        return

    # 初始化处理器
    processor = ImprovedMarkdownProcessor()
    try:
        processor.load_models()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    # 批量处理
    logger.info(f"开始批量处理 {len(md_files)} 个文件")
    for filename in md_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        processor.process_file(input_path, OUTPUT_FOLDER, config)

    logger.info(f"所有文件处理完成，结果保存在 '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()