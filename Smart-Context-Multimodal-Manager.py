"""
title: 🚀 Advanced Multimodal Context Manager
author: JiangNanGenius
version: 1.0.1
license: MIT
required_open_webui_version: 0.5.17
description: 智能长上下文和多模态内容处理器，支持向量化检索、语义重排序、递归总结等功能
"""
import json
import hashlib
import asyncio
import re
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum

# 导入所需库
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

# 模型配置
MULTIMODAL_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview",
    "doubao-1.5-vision-pro", "doubao-1.5-vision-lite",
    "claude-3", "gemini-pro-vision"
}

# Vision预处理模型配置
VISION_MODELS = [
    "doubao-1.5-vision-pro-250328",
    "doubao-1.5-vision-lite-250328", 
    "doubao-1.5-vision-pro",
    "doubao-1.5-vision-lite"
]

# Token限制配置
MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "doubao-1.5-thinking-pro": 128000,
    "doubao-1.5-vision-pro": 128000,
    "claude-3": 200000,
    "gemini-pro": 128000,
}

# 向量化策略枚举
class VectorStrategy(str, Enum):
    AUTO = "auto"  # 自动选择最佳模型
    MULTIMODAL_FIRST = "multimodal_first"  # 优先使用多模态模型
    TEXT_FIRST = "text_first"  # 优先使用文本模型
    MIXED = "mixed"  # 混合使用两个模型
    FALLBACK = "fallback"  # 主模型失败时使用备用模型
    VISION_TO_TEXT = "vision_to_text"  # 图片转文本后用文本向量

class Filter:
    class Valves(BaseModel):
        # === 🎛️ 基础开关配置 ===
        enable_processing: bool = Field(
            default=True,
            description="🔄 启用长上下文和多模态处理"
        )
        
        enable_multimodal: bool = Field(
            default=True,
            description="🖼️ 启用多模态功能（为不支持图片的模型添加视觉能力）"
        )
        
        enable_vision_preprocessing: bool = Field(
            default=True,
            description="👁️ 启用图片预处理（将图片转换为文本描述）"
        )
        
        force_truncate_first: bool = Field(
            default=True,
            description="✂️ 强制先截断，再判断是否使用处理后的消息"
        )

        # === 📊 调试配置 ===
        debug_level: int = Field(
            default=1,
            description="🐛 调试级别：0=关闭，1=基础，2=详细，3=完整",
            json_schema_extra={"enum": [0, 1, 2, 3]}
        )
        
        show_frontend_progress: bool = Field(
            default=True,
            description="📱 显示前端处理进度"
        )

        # === 🎯 Token管理配置 ===
        default_token_limit: int = Field(
            default=120000,
            description="⚖️ 默认token限制（当模型未配置时使用）"
        )
        
        token_safety_ratio: float = Field(
            default=0.85,
            description="🛡️ Token安全比例（实际限制=模型限制*此比例）"
        )
        
        preserve_last_messages: int = Field(
            default=2,
            description="💾 强制保留的最后消息数量（user+assistant对）"
        )
        
        context_preserve_ratio: float = Field(
            default=0.6,
            description="📝 上下文保留比例（0.6=保留60%原文，40%用于摘要）"
        )

        # === 👁️ Vision预处理配置 ===
        vision_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="👁️ Vision预处理API基础URL"
        )
        
        vision_api_key: str = Field(
            default="",
            description="🔑 Vision预处理API密钥"
        )
        
        vision_model: str = Field(
            default="doubao-1.5-vision-pro-250328",
            description="🧠 Vision预处理模型名称",
            json_schema_extra={"enum": VISION_MODELS}
        )
        
        vision_custom_model: str = Field(
            default="",
            description="🎛️ 自定义Vision模型名称（留空使用预设）"
        )
        
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。描述要准确、具体、完整，便于后续的语义检索。",
            description="👁️ Vision模型提示词模板"
        )

        # === 🌐 向量化服务配置 - 多模态模型 ===
        enable_multimodal_vector: bool = Field(
            default=True,
            description="🖼️ 启用多模态向量模型"
        )
        
        multimodal_vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 多模态向量化API基础URL"
        )
        
        multimodal_vector_api_key: str = Field(
            default="",
            description="🔑 多模态向量化API密钥"
        )
        
        multimodal_vector_model: str = Field(
            default="doubao-embedding-vision-250615",
            description="🧠 多模态向量模型名称"
        )
        
        multimodal_vector_custom_model: str = Field(
            default="",
            description="🎛️ 自定义多模态向量模型名称（留空使用预设）"
        )

        # === 🌐 向量化服务配置 - 文本模型 ===
        enable_text_vector: bool = Field(
            default=True,
            description="📝 启用文本向量模型"
        )
        
        text_vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 文本向量化API基础URL"
        )
        
        text_vector_api_key: str = Field(
            default="",
            description="🔑 文本向量化API密钥"
        )
        
        text_vector_model: str = Field(
            default="doubao-embedding-large-text-250515",
            description="🧠 文本向量模型名称",
            json_schema_extra={"enum": [
                "doubao-embedding-large-text-250515",
                "doubao-embedding-large-text-240915",
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]}
        )
        
        text_vector_custom_model: str = Field(
            default="",
            description="🎛️ 自定义文本向量模型名称（留空使用预设）"
        )

        # === 🎯 向量化策略配置 ===
        vector_strategy: VectorStrategy = Field(
            default=VectorStrategy.AUTO,
            description="🎯 向量化策略选择"
        )
        
        vector_similarity_threshold: float = Field(
            default=0.5,
            description="🎯 向量相似度阈值"
        )
        
        multimodal_similarity_threshold: float = Field(
            default=0.45,
            description="🖼️ 多模态内容相似度阈值（通常设置较低）"
        )
        
        text_similarity_threshold: float = Field(
            default=0.55,
            description="📝 纯文本内容相似度阈值（通常设置较高）"
        )

        # === 🔄 重排序配置 ===
        enable_reranking: bool = Field(
            default=True,
            description="🔄 启用语义重排序"
        )
        
        rerank_api_base: str = Field(
            default="https://api.bochaai.com",
            description="🔄 重排序API基础URL"
        )
        
        rerank_api_key: str = Field(
            default="",
            description="🔑 重排序API密钥"
        )
        
        rerank_model: str = Field(
            default="gte-rerank",
            description="🧠 重排序模型名称",
            json_schema_extra={"enum": [
                "gte-rerank",
                "bocha-semantic-reranker-cn",
                "bocha-semantic-reranker-en"
            ]}
        )
        
        rerank_custom_model: str = Field(
            default="",
            description="🎛️ 自定义重排序模型名称（留空使用预设）"
        )
        
        rerank_top_k: int = Field(
            default=10,
            description="🔝 重排序返回的Top-K数量"
        )

        # === 📑 摘要配置 ===
        summary_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="📝 摘要API基础URL"
        )
        
        summary_api_key: str = Field(
            default="",
            description="🔑 摘要API密钥"
        )
        
        summary_model: str = Field(
            default="doubao-1.5-thinking-pro-250415",
            description="🧠 摘要模型名称"
        )
        
        summary_custom_model: str = Field(
            default="",
            description="🎛️ 自定义摘要模型名称（留空使用预设）"
        )
        
        max_summary_length: int = Field(
            default=3000,
            description="📏 单次摘要最大长度"
        )
        
        max_recursion_depth: int = Field(
            default=3,
            description="🔄 最大递归摘要深度"
        )

        # === ⚡ 性能配置 ===
        max_concurrent_requests: int = Field(
            default=3,
            description="⚡ 最大并发请求数"
        )
        
        request_timeout: int = Field(
            default=60,
            description="⏱️ API请求超时时间（秒）"
        )
        
        chunk_size: int = Field(
            default=1000,
            description="📄 文本分片大小（tokens）"
        )
        
        overlap_size: int = Field(
            default=100,
            description="🔗 分片重叠大小（tokens）"
        )

        @classmethod
        def model_validate(cls, v):
            """验证配置"""
            if isinstance(v, dict):
                # 确保至少启用一个向量模型
                if not v.get('enable_multimodal_vector', True) and not v.get('enable_text_vector', True):
                    v['enable_text_vector'] = True  # 强制启用文本向量模型
                
                # 如果启用vision预处理但没有配置，自动配置
                if v.get('enable_vision_preprocessing', True) and not v.get('vision_api_key'):
                    # 尝试使用多模态向量API配置
                    if v.get('multimodal_vector_api_key'):
                        v['vision_api_key'] = v['multimodal_vector_api_key']
                        v['vision_api_base'] = v.get('multimodal_vector_api_base', v['vision_api_base'])
                    # 或者使用文本向量API配置
                    elif v.get('text_vector_api_key'):
                        v['vision_api_key'] = v['text_vector_api_key']
                        v['vision_api_base'] = v.get('text_vector_api_base', v['vision_api_base'])
            
            return super().model_validate(v)

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAzdjE4bTktOWwtOS05LTkgOSIgLz4KPC9zdmc+"""
        
        # 初始化状态
        self._multimodal_vector_client = None
        self._text_vector_client = None
        self._summary_client = None
        self._vision_client = None
        self._encoding = None
        self.processing_cache = {}
        self.vision_cache = {}  # 缓存图片描述结果

    # === 🛠️ 工具函数 ===
    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        """分级调试日志"""
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            print(f"{prefix} {emoji} {message}")

    def get_encoding(self):
        """获取token编码器"""
        if not TIKTOKEN_AVAILABLE:
            return None
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """精确计算token数量"""
        if not text:
            return 0
        encoding = self.get_encoding()
        if encoding is None:
            return len(text) // 4
        try:
            return len(encoding.encode(text))
        except:
            return len(text) // 4

    def count_message_tokens(self, message: dict) -> int:
        """计算单个消息的token数"""
        content = message.get("content", "")
        role = message.get("role", "")
        total_tokens = 0
        
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += self.count_tokens(text)
                elif item.get("type") == "image_url":
                    # 图片按固定token计算
                    total_tokens += 1000
        elif isinstance(content, str):
            total_tokens = self.count_tokens(content)
        
        total_tokens += self.count_tokens(role) + 4
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数"""
        return sum(self.count_message_tokens(msg) for msg in messages)

    def get_model_token_limit(self, model_name: str) -> int:
        """获取模型的token限制"""
        # 从配置中查找
        limit = MODEL_TOKEN_LIMITS.get(model_name.lower())
        if limit:
            return int(limit * self.valves.token_safety_ratio)
        
        # 模糊匹配
        for model_key, model_limit in MODEL_TOKEN_LIMITS.items():
            if model_key in model_name.lower():
                return int(model_limit * self.valves.token_safety_ratio)
        
        # 使用默认值
        return int(self.valves.default_token_limit * self.valves.token_safety_ratio)

    def is_multimodal_model(self, model_name: str) -> bool:
        """检查是否为多模态模型"""
        return any(mm_model in model_name.lower() for mm_model in MULTIMODAL_MODELS)

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        """检查消息中是否包含图片"""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        return True
        return False

    def has_images_in_content(self, content) -> bool:
        """检查单个内容中是否包含图片"""
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    return True
        return False

    def extract_images_from_content(self, content) -> List[str]:
        """从内容中提取图片URL"""
        images = []
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        images.append(image_url)
        return images

    def validate_vector_config(self) -> Tuple[bool, str]:
        """验证向量化配置"""
        if not self.valves.enable_multimodal_vector and not self.valves.enable_text_vector:
            return False, "至少需要启用一个向量模型"
        
        if self.valves.enable_multimodal_vector and not self.valves.multimodal_vector_api_key:
            return False, "多模态向量模型已启用但缺少API密钥"
        
        if self.valves.enable_text_vector and not self.valves.text_vector_api_key:
            return False, "文本向量模型已启用但缺少API密钥"
        
        return True, "配置验证通过"

    async def send_status(self, __event_emitter__, message: str, done: bool = True, emoji: str = "🔄"):
        """发送状态消息到前端"""
        if __event_emitter__ and self.valves.show_frontend_progress:
            try:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"{emoji} {message}",
                        "done": done,
                    },
                })
            except Exception as e:
                self.debug_log(1, f"发送状态失败: {e}", "❌")

    # === 👁️ Vision预处理 ===
    def get_vision_client(self):
        """获取Vision客户端"""
        if not OPENAI_AVAILABLE:
            return None
        
        if self._vision_client is None:
            api_key = self.valves.vision_api_key
            if not api_key:
                return None
            
            self._vision_client = AsyncOpenAI(
                base_url=self.valves.vision_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout
            )
        
        return self._vision_client

    async def describe_image_with_vision(self, image_url: str, __event_emitter__) -> str:
        """使用Vision模型描述图片"""
        # 检查缓存
        image_hash = hashlib.md5(image_url.encode()).hexdigest()
        if image_hash in self.vision_cache:
            self.debug_log(2, "使用缓存的图片描述", "💾")
            return self.vision_cache[image_hash]
        
        client = self.get_vision_client()
        if not client:
            return "无法处理图片：缺少Vision API配置"
        
        vision_model = self.valves.vision_custom_model or self.valves.vision_model
        
        try:
            await self.send_status(__event_emitter__, f"正在分析图片内容...", False, "👁️")
            
            response = await client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.valves.vision_prompt_template},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            if response.choices:
                description = response.choices[0].message.content.strip()
                # 缓存结果
                self.vision_cache[image_hash] = description
                self.debug_log(2, f"图片描述生成成功: {description[:100]}...", "✅")
                return description
            else:
                return "图片描述生成失败"
        
        except Exception as e:
            self.debug_log(1, f"Vision模型调用失败: {e}", "❌")
            return f"图片处理错误: {str(e)[:100]}"

    async def preprocess_content_for_text_vector(self, content, __event_emitter__) -> str:
        """为文本向量模型预处理内容（将图片转换为文本描述）"""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            processed_parts = []
            has_images = False
            
            for item in content:
                if item.get("type") == "text":
                    processed_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    has_images = True
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url and self.valves.enable_vision_preprocessing:
                        description = await self.describe_image_with_vision(image_url, __event_emitter__)
                        processed_parts.append(f"[图片描述] {description}")
                    else:
                        processed_parts.append("[图片] 无法处理")
            
            if has_images:
                self.debug_log(2, "将多模态内容转换为纯文本", "🔄")
            
            return " ".join(processed_parts)
        
        return str(content)

    # === 🖼️ 多模态处理 ===
    async def process_multimodal_content(self, messages: List[dict], __event_emitter__) -> List[dict]:
        """处理多模态内容"""
        if not self.valves.enable_multimodal:
            return messages
        
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return messages
        
        await self.send_status(__event_emitter__, "检测到图片内容，准备处理...", False, "🖼️")
        
        processed_messages = []
        for message in messages:
            content = message.get("content", "")
            
            if isinstance(content, list):
                # 处理多模态消息
                text_parts = []
                image_descriptions = []
                
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        # 描述图片
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            try:
                                description = await self.describe_image_with_vision(image_url, __event_emitter__)
                                image_descriptions.append(f"[图片描述] {description}")
                            except Exception as e:
                                self.debug_log(1, f"图片描述失败: {e}", "❌")
                                image_descriptions.append("[图片] 处理失败")
                
                # 合并文本和图片描述
                combined_content = " ".join(text_parts + image_descriptions)
                processed_message = message.copy()
                processed_message["content"] = combined_content
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        
        await self.send_status(__event_emitter__, "多模态内容处理完成", True, "✅")
        return processed_messages

    # === 🔗 向量化和检索 ===
    def choose_vector_model(self, content_type: str = "text", has_images: bool = False) -> Tuple[str, str, str, str]:
        """根据策略选择向量模型
        Returns:
            Tuple[api_base, api_key, model_name, model_type]
        """
        strategy = self.valves.vector_strategy
        
        # 获取实际使用的模型名称
        multimodal_model = self.valves.multimodal_vector_custom_model or self.valves.multimodal_vector_model
        text_model = self.valves.text_vector_custom_model or self.valves.text_vector_model
        
        # 关键修复：如果有图片但多模态向量模型不可用，强制使用 VISION_TO_TEXT 策略
        if has_images and not self.valves.enable_multimodal_vector and self.valves.enable_text_vector:
            strategy = VectorStrategy.VISION_TO_TEXT
            self.debug_log(2, "检测到图片但多模态向量模型不可用，自动切换到VISION_TO_TEXT策略", "🔄")
        
        if strategy == VectorStrategy.AUTO:
            # 自动选择：有图片且多模态模型可用时使用多模态，否则使用文本模型
            if has_images and self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    multimodal_model,
                    "multimodal"
                )
            elif self.valves.enable_text_vector:
                return (
                    self.valves.text_vector_api_base,
                    self.valves.text_vector_api_key,
                    text_model,
                    "text"
                )
            elif self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    multimodal_model,
                    "multimodal"
                )
        
        elif strategy == VectorStrategy.MULTIMODAL_FIRST:
            # 优先使用多模态模型
            if self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    multimodal_model,
                    "multimodal"
                )
            elif self.valves.enable_text_vector:
                return (
                    self.valves.text_vector_api_base,
                    self.valves.text_vector_api_key,
                    text_model,
                    "text"
                )
        
        elif strategy == VectorStrategy.TEXT_FIRST or strategy == VectorStrategy.VISION_TO_TEXT:
            # 优先使用文本模型（VISION_TO_TEXT会在向量化时预处理图片）
            if self.valves.enable_text_vector:
                return (
                    self.valves.text_vector_api_base,
                    self.valves.text_vector_api_key,
                    text_model,
                    "text"
                )
            elif self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    multimodal_model,
                    "multimodal"
                )
        
        # 默认返回可用的第一个模型
        if self.valves.enable_text_vector:
            return (
                self.valves.text_vector_api_base,
                self.valves.text_vector_api_key,
                text_model,
                "text"
            )
        elif self.valves.enable_multimodal_vector:
            return (
                self.valves.multimodal_vector_api_base,
                self.valves.multimodal_vector_api_key,
                multimodal_model,
                "multimodal"
            )
        
        raise ValueError("没有可用的向量模型")

    async def vectorize_content(self, content, __event_emitter__, content_type: str = "text", has_images: bool = False) -> Optional[List[float]]:
        """向量化内容（智能处理图片）"""
        if not HTTPX_AVAILABLE:
            return None
        
        try:
            api_base, api_key, model_name, model_type = self.choose_vector_model(content_type, has_images)
        except ValueError as e:
            self.debug_log(1, f"选择向量模型失败: {e}", "❌")
            return None
        
        if not api_key:
            self.debug_log(1, f"向量模型 {model_type} 缺少API密钥", "❌")
            return None
        
        # 预处理内容
        text_content = content
        if model_type == "text" and (has_images or self.has_images_in_content(content)):
            # 文本向量模型需要将图片转换为文本描述
            text_content = await self.preprocess_content_for_text_vector(content, __event_emitter__)
