"""
title: 🚀 Advanced Multimodal Context Manager
author: JiangNanGenius
version: 1.5.2
license: MIT
required_open_webui_version: 0.5.17
description: 智能长上下文和多模态内容处理器，支持向量化检索、语义重排序、递归总结等功能 - 修复JSON解析错误
"""

import json
import hashlib
import asyncio
import re
import base64
import math
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum

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

class VectorStrategy(str, Enum):
    AUTO = "auto"
    MULTIMODAL_FIRST = "multimodal_first"
    TEXT_FIRST = "text_first"
    MIXED = "mixed"
    FALLBACK = "fallback"
    VISION_TO_TEXT = "vision_to_text"

class MultimodalStrategy(str, Enum):
    ALL_MODELS = "all_models"
    NON_MULTIMODAL_ONLY = "non_multimodal_only"
    CUSTOM_LIST = "custom_list"
    SMART_ADAPTIVE = "smart_adaptive"

class Filter:
    class Valves(BaseModel):
        # 基础控制
        enable_processing: bool = Field(default=True, description="🔄 启用所有处理功能")
        excluded_models: str = Field(
            default="", description="🚫 排除模型列表(逗号分隔)"
        )
        
        # 多模态模型配置
        multimodal_models: str = Field(
            default="gpt-4o,gpt-4o-mini,gpt-4-vision-preview,doubao-1.5-vision-pro,doubao-1.5-vision-lite,claude-3,gemini-pro-vision,qwen-vl",
            description="🖼️ 多模态模型列表(逗号分隔)",
        )
        
        # 模型Token限制配置
        model_token_limits: str = Field(
            default="gpt-4o:128000,gpt-4o-mini:128000,gpt-4:8192,gpt-3.5-turbo:16385,doubao-1.5-thinking-pro:128000,doubao-1.5-vision-pro:128000,doubao-seed:50000,doubao:50000,claude-3:200000,gemini-pro:128000",
            description="⚖️ 模型Token限制配置(model:limit格式，逗号分隔)",
        )
        
        # 多模态处理策略
        multimodal_processing_strategy: str = Field(
            default="smart_adaptive",
            description="🖼️ 多模态处理策略 (all_models|non_multimodal_only|custom_list|smart_adaptive)",
        )
        force_vision_processing_models: str = Field(
            default="gpt-4,gpt-3.5-turbo,doubao-1.5-thinking-pro",
            description="🔍 强制进行视觉处理的模型列表(逗号分隔)",
        )
        preserve_images_in_multimodal: bool = Field(
            default=True, description="📸 多模态模型是否保留原始图片"
        )
        always_process_images_before_summary: bool = Field(
            default=True, description="📝 摘要前总是先处理图片"
        )
        
        # 功能开关
        enable_multimodal: bool = Field(default=True, description="🖼️ 启用多模态处理")
        enable_vision_preprocessing: bool = Field(
            default=True, description="👁️ 启用图片预处理"
        )
        enable_smart_truncation: bool = Field(
            default=True, description="✂️ 启用智能截断"
        )
        enable_vector_retrieval: bool = Field(
            default=True, description="🔍 启用向量检索"
        )
        enable_content_maximization: bool = Field(
            default=True, description="📈 启用内容最大化保留"
        )
        
        # 调试和错误处理
        debug_level: int = Field(default=2, description="🐛 调试级别 0-3")
        show_frontend_progress: bool = Field(
            default=True, description="📱 显示处理进度"
        )
        api_error_retry_times: int = Field(
            default=2, description="🔄 API错误重试次数"
        )
        api_error_retry_delay: float = Field(
            default=1.0, description="⏱️ API错误重试延迟(秒)"
        )
        
        # Token管理 - 最大化保留策略
        default_token_limit: int = Field(default=100000, description="⚖️ 默认token限制")
        token_safety_ratio: float = Field(
            default=0.95, description="🛡️ Token安全比例"
        )
        max_processing_iterations: int = Field(
            default=3, description="🔄 最大处理迭代次数"
        )
        min_reduction_threshold: int = Field(
            default=2000, description="📉 最小减少阈值"
        )
        
        # 保护策略
        force_preserve_last_user_message: bool = Field(
            default=True, description="🔒 强制保留用户最后消息"
        )
        preserve_recent_exchanges: int = Field(
            default=3, description="💬 保护最近完整对话轮次"
        )
        max_preserve_ratio: float = Field(
            default=0.75, description="🔒 保护消息最大token比例"
        )
        max_single_message_tokens: int = Field(
            default=20000, description="📝 单条消息最大token"
        )
        
        # Vision配置
        vision_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="👁️ Vision API地址",
        )
        vision_api_key: str = Field(default="", description="🔑 Vision API密钥")
        vision_model: str = Field(
            default="doubao-1.5-vision-pro-250328", description="🧠 Vision模型"
        )
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。保持客观准确，重点突出关键信息。",
            description="👁️ Vision提示词",
        )
        vision_max_tokens: int = Field(
            default=1200, description="👁️ Vision最大输出tokens"
        )
        
        # 多模态向量
        enable_multimodal_vector: bool = Field(
            default=True, description="🖼️ 启用多模态向量"
        )
        multimodal_vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 多模态向量API",
        )
        multimodal_vector_api_key: str = Field(
            default="", description="🔑 多模态向量密钥"
        )
        multimodal_vector_model: str = Field(
            default="doubao-embedding-vision-250615", description="🧠 多模态向量模型"
        )
        
        # 文本向量
        enable_text_vector: bool = Field(default=True, description="📝 启用文本向量")
        text_vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 文本向量API",
        )
        text_vector_api_key: str = Field(default="", description="🔑 文本向量密钥")
        text_vector_model: str = Field(
            default="doubao-embedding-large-text-250515", description="🧠 文本向量模型"
        )
        
        # 向量策略
        vector_strategy: str = Field(
            default="auto",
            description="🎯 向量化策略 (auto|multimodal_first|text_first|mixed|fallback|vision_to_text)",
        )
        vector_similarity_threshold: float = Field(
            default=0.4, description="🎯 基础相似度阈值"
        )
        multimodal_similarity_threshold: float = Field(
            default=0.35, description="🖼️ 多模态相似度阈值"
        )
        text_similarity_threshold: float = Field(
            default=0.45, description="📝 文本相似度阈值"
        )
        vector_top_k: int = Field(default=25, description="🔝 向量检索Top-K数量")
        
        # 重排序
        enable_reranking: bool = Field(default=True, description="🔄 启用重排序")
        rerank_api_base: str = Field(
            default="https://api.bochaai.com", description="🔄 重排序API"
        )
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=20, description="🔝 重排序返回数量")
        
        # 摘要配置
        summary_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3", description="📝 摘要API"
        )
        summary_api_key: str = Field(default="", description="🔑 摘要密钥")
        summary_model: str = Field(
            default="doubao-1.5-thinking-pro-250415", description="🧠 摘要模型"
        )
        max_summary_length: int = Field(
            default=4000, description="📏 摘要最大长度"
        )
        max_recursion_depth: int = Field(
            default=3, description="🔄 最大递归深度"
        )
        
        # 性能配置
        max_concurrent_requests: int = Field(default=3, description="⚡ 最大并发数")
        request_timeout: int = Field(default=60, description="⏱️ 请求超时(秒)")
        chunk_size: int = Field(default=1500, description="📄 分片大小")
        overlap_size: int = Field(default=150, description="🔗 重叠大小")

    def __init__(self):
        print("\n" + "=" * 60)
        print("🚀 Advanced Multimodal Context Manager v1.5.2")
        print("📍 插件正在初始化...")
        print("🔧 修复JSON解析错误...")
        
        self.valves = self.Valves()
        self._vision_client = None
        self._text_vector_client = None
        self._multimodal_vector_client = None
        self._rerank_client = None
        self._encoding = None
        
        self.vision_cache = {}
        self.vector_cache = {}
        self.processing_cache = {}
        self.api_error_cache = {}
        
        # 解析多模态模型配置
        self.multimodal_models = set()
        if self.valves.multimodal_models:
            self.multimodal_models = {
                model.strip().lower()
                for model in self.valves.multimodal_models.split(",")
                if model.strip()
            }
        
        # 解析模型Token限制配置
        self.model_token_limits = {}
        if self.valves.model_token_limits:
            for limit_config in self.valves.model_token_limits.split(","):
                if ":" in limit_config:
                    model, limit = limit_config.split(":", 1)
                    try:
                        self.model_token_limits[model.strip().lower()] = int(
                            limit.strip()
                        )
                    except ValueError:
                        pass
        
        print(f"✅ 插件初始化完成")
        print(f"🔧 错误重试次数: {self.valves.api_error_retry_times}")
        print(f"🔧 错误重试延迟: {self.valves.api_error_retry_delay}秒")
        print("=" * 60 + "\n")

    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            print(f"{prefix} {emoji} {message}")

    def is_model_excluded(self, model_name: str) -> bool:
        if not self.valves.excluded_models or not model_name:
            return False
        
        excluded_list = [
            model.strip().lower()
            for model in self.valves.excluded_models.split(",")
            if model.strip()
        ]
        
        if not excluded_list:
            return False
        
        model_lower = model_name.lower()
        for excluded_model in excluded_list:
            if excluded_model in model_lower:
                self.debug_log(1, f"模型 {model_name} 在排除列表中", "🚫")
                return True
        
        return False

    def get_encoding(self):
        if not TIKTOKEN_AVAILABLE:
            return None
        
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self.debug_log(3, "tiktoken编码器已初始化", "🔧")
            except Exception as e:
                self.debug_log(1, f"tiktoken初始化失败: {e}", "⚠️")
        
        return self._encoding

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        
        text = str(text)
        encoding = self.get_encoding()
        
        if encoding:
            try:
                return len(encoding.encode(text))
            except Exception as e:
                self.debug_log(2, f"token计算失败，使用估算: {e}", "⚠️")
        
        return max(len(text) // 3, len(text.encode("utf-8")) // 4)

    def count_message_tokens(self, message: dict) -> int:
        if not message:
            return 0
        
        content = message.get("content", "")
        role = message.get("role", "")
        total_tokens = 0
        
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total_tokens += self.count_tokens(item.get("text", ""))
                elif item.get("type") == "image_url":
                    total_tokens += 1500  # 图片token估算
        else:
            total_tokens = self.count_tokens(str(content))
        
        # 角色和格式开销
        total_tokens += self.count_tokens(role) + 10
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        if not messages:
            return 0
        return sum(self.count_message_tokens(msg) for msg in messages)

    def get_model_token_limit(self, model_name: str) -> int:
        model_lower = model_name.lower()
        
        # 优先使用配置的限制
        for model_key, limit in self.model_token_limits.items():
            if model_key in model_lower:
                safe_limit = int(limit * self.valves.token_safety_ratio)
                self.debug_log(
                    2, f"模型 {model_name} 限制: {limit} -> {safe_limit}", "⚖️"
                )
                return safe_limit
        
        # 使用默认限制
        safe_limit = int(
            self.valves.default_token_limit * self.valves.token_safety_ratio
        )
        self.debug_log(1, f"未知模型 {model_name}, 使用默认限制: {safe_limit}", "⚠️")
        return safe_limit

    def is_multimodal_model(self, model_name: str) -> bool:
        model_lower = model_name.lower()
        return any(mm in model_lower for mm in self.multimodal_models)

    def should_process_images_for_model(self, model_name: str) -> bool:
        if not self.valves.enable_multimodal:
            return False
        
        model_lower = model_name.lower()
        
        # 检查强制处理列表
        force_list = [
            m.strip().lower()
            for m in self.valves.force_vision_processing_models.split(",")
            if m.strip()
        ]
        
        if any(force_model in model_lower for force_model in force_list):
            self.debug_log(2, f"模型 {model_name} 在强制处理列表中", "🔍")
            return True
        
        # 根据策略判断
        is_multimodal = self.is_multimodal_model(model_name)
        strategy = self.valves.multimodal_processing_strategy.lower()
        
        if strategy == "all_models":
            return True
        elif strategy == "non_multimodal_only":
            return not is_multimodal
        elif strategy == "smart_adaptive":
            return True
        else:
            return not is_multimodal

    def has_images_in_content(self, content) -> bool:
        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)
        return False

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        return any(self.has_images_in_content(msg.get("content")) for msg in messages)

    async def send_status(
        self, __event_emitter__, message: str, done: bool = True, emoji: str = "🔄"
    ):
        self.debug_log(2, f"状态: {message}", emoji)
        if __event_emitter__ and self.valves.show_frontend_progress:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"{emoji} {message}", "done": done},
                    }
                )
            except:
                pass

    # ========== 增强的API调用方法 ==========
    def is_json_response(self, content: str) -> bool:
        """检查响应是否为JSON格式"""
        if not content:
            return False
        
        content = content.strip()
        return content.startswith('{') or content.startswith('[')

    def extract_error_info(self, content: str) -> str:
        """从错误响应中提取关键信息"""
        if not content:
            return "空响应"
        
        # 检查是否为HTML错误页面
        if content.strip().startswith('<!DOCTYPE') or '<html' in content:
            # 尝试提取title
            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
            if title_match:
                return f"HTML错误页面: {title_match.group(1)}"
            return "HTML错误页面"
        
        # 检查是否为JSON错误
        try:
            if self.is_json_response(content):
                error_data = json.loads(content)
                if isinstance(error_data, dict):
                    error_msg = error_data.get('error', error_data.get('message', ''))
                    if error_msg:
                        return f"API错误: {error_msg}"
            return f"响应内容: {content[:200]}..."
        except:
            return f"响应内容: {content[:200]}..."

    async def safe_api_call(self, call_func, call_name: str, *args, **kwargs):
        """安全的API调用包装器"""
        error_key = f"{call_name}_{hash(str(args) + str(kwargs))}"
        
        # 检查错误缓存
        if error_key in self.api_error_cache:
            cache_time, error_msg = self.api_error_cache[error_key]
            if asyncio.get_event_loop().time() - cache_time < 300:  # 5分钟缓存
                self.debug_log(1, f"使用缓存的错误结果: {error_msg}", "⚠️")
                return None
        
        for attempt in range(self.valves.api_error_retry_times + 1):
            try:
                result = await call_func(*args, **kwargs)
                # 清除错误缓存
                if error_key in self.api_error_cache:
                    del self.api_error_cache[error_key]
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否为JSON解析错误
                if "is not valid JSON" in error_msg or "Unexpected token" in error_msg:
                    self.debug_log(1, f"{call_name} JSON解析错误: {error_msg}", "❌")
                    # 记录到错误缓存
                    self.api_error_cache[error_key] = (asyncio.get_event_loop().time(), error_msg)
                    return None
                
                if attempt < self.valves.api_error_retry_times:
                    self.debug_log(
                        1, f"{call_name} 第{attempt+1}次尝试失败，{self.valves.api_error_retry_delay}秒后重试: {error_msg}", "🔄"
                    )
                    await asyncio.sleep(self.valves.api_error_retry_delay)
                else:
                    self.debug_log(1, f"{call_name} 最终失败: {error_msg}", "❌")
                    # 记录到错误缓存
                    self.api_error_cache[error_key] = (asyncio.get_event_loop().time(), error_msg)
                    return None
        
        return None

    # ========== 向量化功能 ==========
    def get_text_vector_client(self):
        if not OPENAI_AVAILABLE:
            return None
        
        if self._text_vector_client:
            return self._text_vector_client
        
        api_key = self.valves.text_vector_api_key
        if not api_key:
            api_key = (
                self.valves.multimodal_vector_api_key or self.valves.vision_api_key
            )
        
        if api_key:
            self._text_vector_client = AsyncOpenAI(
                base_url=self.valves.text_vector_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout,
            )
            self.debug_log(2, "文本向量客户端已创建", "📝")
        
        return self._text_vector_client

    def get_multimodal_vector_client(self):
        if not OPENAI_AVAILABLE:
            return None
        
        if self._multimodal_vector_client:
            return self._multimodal_vector_client
        
        api_key = self.valves.multimodal_vector_api_key
        if not api_key:
            api_key = self.valves.text_vector_api_key or self.valves.vision_api_key
        
        if api_key:
            self._multimodal_vector_client = AsyncOpenAI(
                base_url=self.valves.multimodal_vector_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout,
            )
            self.debug_log(2, "多模态向量客户端已创建", "🖼️")
        
        return self._multimodal_vector_client

    async def _get_text_embedding_impl(self, text: str, __event_emitter__):
        """实际的文本向量获取实现"""
        client = self.get_text_vector_client()
        if not client:
            return None
        
        response = await client.embeddings.create(
            model=self.valves.text_vector_model,
            input=[text[:8000]],
            encoding_format="float",
        )
        
        if response.data:
            return response.data[0].embedding
        return None

    async def get_text_embedding(
        self, text: str, __event_emitter__
    ) -> Optional[List[float]]:
        """获取文本向量"""
        if not text or not self.valves.enable_text_vector:
            return None
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"text_emb_{text_hash}"
        
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        embedding = await self.safe_api_call(
            self._get_text_embedding_impl, "文本向量", text, __event_emitter__
        )
        
        if embedding:
            self.vector_cache[cache_key] = embedding
            self.debug_log(3, f"文本向量获取成功: {len(embedding)}维", "📝")
        
        return embedding

    async def _get_multimodal_embedding_impl(self, content, __event_emitter__):
        """实际的多模态向量获取实现"""
        client = self.get_multimodal_vector_client()
        if not client:
            return None
        
        # 处理输入格式
        if isinstance(content, list):
            input_data = content
        else:
            input_data = [{"type": "text", "text": str(content)[:8000]}]
        
        response = await client.embeddings.create(
            model=self.valves.multimodal_vector_model, input=input_data
        )
        
        if response.data:
            return response.data[0].embedding
        return None

    async def get_multimodal_embedding(
        self, content, __event_emitter__
    ) -> Optional[List[float]]:
        """获取多模态向量"""
        if not content or not self.valves.enable_multimodal_vector:
            return None
        
        # 生成缓存key
        if isinstance(content, list):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        cache_key = f"multimodal_emb_{content_hash}"
        
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        embedding = await self.safe_api_call(
            self._get_multimodal_embedding_impl, "多模态向量", content, __event_emitter__
        )
        
        if embedding:
            self.vector_cache[cache_key] = embedding
            self.debug_log(3, f"多模态向量获取成功: {len(embedding)}维", "🖼️")
        
        return embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    async def vector_retrieve_relevant_messages(
        self, query_message: dict, candidate_messages: List[dict], __event_emitter__
    ) -> List[dict]:
        """基于向量相似度检索相关消息"""
        if not candidate_messages or not self.valves.enable_vector_retrieval:
            return candidate_messages
        
        self.debug_log(
            1, f"开始向量检索: 查询1条，候选{len(candidate_messages)}条", "🔍"
        )
        
        await self.send_status(
            __event_emitter__,
            f"向量检索 {len(candidate_messages)} 条消息...",
            False,
            "🔍",
        )
        
        # 获取查询向量
        query_content = query_message.get("content", "")
        query_vector = None
        strategy = self.valves.vector_strategy.lower()
        
        # 根据策略选择向量化方法
        if self.has_images_in_content(query_content):
            if strategy in ["auto", "multimodal_first"]:
                query_vector = await self.get_multimodal_embedding(
                    query_content, __event_emitter__
                )
            if not query_vector and strategy in ["auto", "fallback"]:
                # 转换为文本再向量化
                text_content = self.extract_text_from_content(query_content)
                if text_content:
                    query_vector = await self.get_text_embedding(
                        text_content, __event_emitter__
                    )
        else:
            text_content = self.extract_text_from_content(query_content)
            if text_content:
                query_vector = await self.get_text_embedding(
                    text_content, __event_emitter__
                )
        
        if not query_vector:
            self.debug_log(1, "查询向量获取失败，返回原始消息", "⚠️")
            return candidate_messages
        
        # 计算候选消息的相似度
        similarities = []
        for i, msg in enumerate(candidate_messages):
            msg_content = msg.get("content", "")
            msg_vector = None
            
            # 为候选消息获取向量
            if self.has_images_in_content(msg_content):
                msg_vector = await self.get_multimodal_embedding(
                    msg_content, __event_emitter__
                )
                if not msg_vector:
                    text_content = self.extract_text_from_content(msg_content)
                    if text_content:
                        msg_vector = await self.get_text_embedding(
                            text_content, __event_emitter__
                        )
            else:
                text_content = self.extract_text_from_content(msg_content)
                if text_content:
                    msg_vector = await self.get_text_embedding(
                        text_content, __event_emitter__
                    )
            
            if msg_vector:
                similarity = self.cosine_similarity(query_vector, msg_vector)
                similarities.append((i, similarity, msg))
                self.debug_log(3, f"消息{i}相似度: {similarity:.3f}", "📊")
            else:
                # 没有向量的消息给中等分数但仍保留
                similarities.append((i, 0.4, msg))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 根据阈值过滤
        threshold = self.valves.vector_similarity_threshold
        filtered_similarities = [item for item in similarities if item[1] >= threshold]
        
        # 如果过滤后太少，保留更多消息
        if len(filtered_similarities) < len(candidate_messages) * 0.5:
            filtered_similarities = similarities[
                : max(len(similarities) // 2, self.valves.vector_top_k)
            ]
        
        # 限制数量
        top_similarities = filtered_similarities[: self.valves.vector_top_k]
        
        # 提取消息并保持原始顺序
        relevant_messages = []
        selected_indices = sorted([item[0] for item in top_similarities])
        
        for idx in selected_indices:
            relevant_messages.append(candidate_messages[idx])
        
        self.debug_log(
            1,
            f"向量检索完成: {len(candidate_messages)} -> {len(relevant_messages)}条",
            "✅",
        )
        
        await self.send_status(
            __event_emitter__,
            f"向量检索完成: {len(relevant_messages)}条相关消息",
            True,
            "✅",
        )
        
        return relevant_messages

    def extract_text_from_content(self, content) -> str:
        """从内容中提取文本"""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
        else:
            return str(content)

    # ========== 重排序功能 ==========
    async def _rerank_messages_impl(self, query_text: str, documents: List[str], __event_emitter__):
        """实际的重排序实现"""
        if not HTTPX_AVAILABLE:
            return None
        
        async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
            headers = {
                "Authorization": f"Bearer {self.valves.rerank_api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "model": self.valves.rerank_model,
                "query": query_text,
                "documents": documents,
                "top_n": min(self.valves.rerank_top_k, len(documents)),
                "return_documents": True,
            }
            
            response = await client.post(
                f"{self.valves.rerank_api_base}/v1/rerank",
                headers=headers,
                json=data,
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            # 检查响应格式
            response_text = response.text
            if not self.is_json_response(response_text):
                error_info = self.extract_error_info(response_text)
                raise Exception(f"非JSON响应: {error_info}")
            
            result = response.json()
            
            if result.get("code") != 200:
                error_msg = result.get("message", "未知错误")
                raise Exception(f"API错误: {error_msg}")
            
            return result.get("data", {}).get("results", [])

    async def rerank_messages(
        self, query_message: dict, candidate_messages: List[dict], __event_emitter__
    ) -> List[dict]:
        """重排序消息"""
        if not candidate_messages or not self.valves.enable_reranking:
            return candidate_messages
        
        self.debug_log(1, f"开始重排序: 查询1条，候选{len(candidate_messages)}条", "🔄")
        
        await self.send_status(
            __event_emitter__,
            f"重排序 {len(candidate_messages)} 条消息...",
            False,
            "🔄",
        )
        
        # 准备查询文本
        query_text = self.extract_text_from_content(query_message.get("content", ""))
        if not query_text:
            return candidate_messages
        
        # 准备文档列表
        documents = []
        for msg in candidate_messages:
            text = self.extract_text_from_content(msg.get("content", ""))
            if text:
                # 提高文档长度限制
                if len(text) > 3000:
                    text = text[:3000] + "..."
                documents.append(text)
            else:
                documents.append("空消息")
        
        if not documents:
            return candidate_messages
        
        # 调用重排序API
        rerank_results = await self.safe_api_call(
            self._rerank_messages_impl, "重排序", query_text, documents, __event_emitter__
        )
        
        if rerank_results:
            # 按重排序结果重新排列消息
            reranked_messages = []
            for item in rerank_results:
                original_index = item.get("index", 0)
                if 0 <= original_index < len(candidate_messages):
                    reranked_messages.append(candidate_messages[original_index])
                    score = item.get("relevance_score", 0)
                    self.debug_log(
                        3,
                        f"重排序结果: index={original_index}, score={score:.3f}",
                        "📊",
                    )
            
            self.debug_log(
                1,
                f"重排序完成: {len(candidate_messages)} -> {len(reranked_messages)}条",
                "✅",
            )
            
            await self.send_status(
                __event_emitter__,
                f"重排序完成: {len(reranked_messages)}条消息",
                True,
                "✅",
            )
            
            return reranked_messages
        
        return candidate_messages

    # ========== Vision处理 ==========
    def get_vision_client(self):
        if not OPENAI_AVAILABLE:
            return None
        
        if self._vision_client:
            return self._vision_client
        
        api_key = self.valves.vision_api_key
        if not api_key:
            api_key = (
                self.valves.multimodal_vector_api_key or self.valves.text_vector_api_key
            )
        
        if api_key:
            self._vision_client = AsyncOpenAI(
                base_url=self.valves.vision_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout,
            )
            self.debug_log(2, "Vision客户端已创建", "👁️")
        
        return self._vision_client

    async def _describe_image_impl(self, image_url: str, __event_emitter__):
        """实际的图片描述实现"""
        client = self.get_vision_client()
        if not client:
            return None
        
        response = await client.chat.completions.create(
            model=self.valves.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.valves.vision_prompt_template,
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=self.valves.vision_max_tokens,
            temperature=0.2,
        )
        
        if response.choices:
            return response.choices[0].message.content.strip()
        return None

    async def describe_image(self, image_url: str, __event_emitter__) -> str:
        """描述单张图片"""
        image_hash = hashlib.md5(image_url.encode()).hexdigest()
        
        if image_hash in self.vision_cache:
            self.debug_log(3, f"使用缓存的图片描述: {image_hash[:8]}", "📋")
            return self.vision_cache[image_hash]
        
        self.debug_log(2, f"开始识别图片: {image_hash[:8]}", "👁️")
        
        description = await self.safe_api_call(
            self._describe_image_impl, "图片识别", image_url, __event_emitter__
        )
        
        if description:
            # 提高描述长度限制
            if len(description) > 1200:
                description = description[:1200] + "..."
            
            self.vision_cache[image_hash] = description
            self.debug_log(2, f"图片识别完成: {len(description)}字符", "✅")
            return description
        
        return "图片处理失败：无法获取描述"

    async def process_message_images(self, message: dict, __event_emitter__) -> dict:
        """处理单条消息中的图片"""
        content = message.get("content", "")
        if not isinstance(content, list):
            return message
        
        # 检查是否包含图片
        has_images = any(item.get("type") == "image_url" for item in content)
        if not has_images:
            return message
        
        # 处理图片
        processed_content = []
        image_count = 0
        
        for item in content:
            if item.get("type") == "text":
                processed_content.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_count += 1
                image_url = item.get("image_url", {}).get("url", "")
                if image_url:
                    description = await self.describe_image(
                        image_url, __event_emitter__
                    )
                    processed_content.append(f"[图片{image_count}描述] {description}")
        
        # 创建新消息
        processed_message = message.copy()
        processed_message["content"] = (
            "\n".join(processed_content) if processed_content else ""
        )
        
        self.debug_log(2, f"消息图片处理完成: {image_count}张图片", "🖼️")
        return processed_message

    async def process_multimodal_content(
        self, messages: List[dict], model_name: str, __event_emitter__
    ) -> List[dict]:
        """处理多模态内容"""
        if not self.valves.enable_multimodal:
            return messages
        
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return messages
        
        should_process = self.should_process_images_for_model(model_name)
        is_multimodal = self.is_multimodal_model(model_name)
        
        self.debug_log(
            1,
            f"多模态处理检查: 模型={model_name}, 多模态={is_multimodal}, 需要处理={should_process}",
            "🖼️",
        )
        
        if (
            is_multimodal
            and self.valves.preserve_images_in_multimodal
            and not should_process
        ):
            self.debug_log(2, f"多模态模型 {model_name} 保留原始图片", "📸")
            return messages
        
        # 统计图片数量
        total_images = 0
        for msg in messages:
            if isinstance(msg.get("content"), list):
                total_images += len(
                    [
                        item
                        for item in msg.get("content", [])
                        if item.get("type") == "image_url"
                    ]
                )
        
        if total_images == 0:
            return messages
        
        self.debug_log(1, f"开始处理多模态内容：{total_images} 张图片", "🖼️")
        
        await self.send_status(
            __event_emitter__,
            f"处理 {total_images} 张图片...",
            False,
            "🖼️",
        )
        
        # 处理所有消息
        processed_messages = []
        processed_count = 0
        
        for message in messages:
            if self.has_images_in_content(message.get("content")):
                processed_message = await self.process_message_images(
                    message, __event_emitter__
                )
                processed_messages.append(processed_message)
                if isinstance(message.get("content"), list):
                    processed_count += len(
                        [
                            item
                            for item in message.get("content", [])
                            if item.get("type") == "image_url"
                        ]
                    )
            else:
                processed_messages.append(message)
        
        self.debug_log(1, f"多模态处理完成：{processed_count} 张图片", "✅")
        await self.send_status(__event_emitter__, "图片处理完成", True, "✅")
        
        return processed_messages

    # ========== 内容最大化保留策略 ==========
    def get_summary_client(self):
        """获取摘要客户端"""
        if not OPENAI_AVAILABLE:
            return None
        
        api_key = self.valves.summary_api_key
        if not api_key:
            api_key = (
                self.valves.multimodal_vector_api_key
                or self.valves.text_vector_api_key
                or self.valves.vision_api_key
            )
        
        if api_key:
            return AsyncOpenAI(
                base_url=self.valves.summary_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout,
            )
        
        return None

    def smart_message_selection_v2(
        self, messages: List[dict], target_tokens: int, iteration: int = 0
    ) -> Tuple[List[dict], List[dict]]:
        """
        内容最大化保留的智能消息选择策略
        核心思想：强制保留用户最后消息，最大化保留其他内容
        """
        if not messages:
            return [], []
        
        # 分离不同类型的消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        protected = []
        current_tokens = 0
        
        # 1. 强制保留用户最后消息
        last_user_message = None
        if self.valves.force_preserve_last_user_message and user_messages:
            last_user_message = user_messages[-1]
            protected.append(last_user_message)
            current_tokens += self.count_message_tokens(last_user_message)
            self.debug_log(1, f"🔒 强制保留用户最后消息: {current_tokens}tokens", "💾")
        
        # 2. 保留系统消息
        for msg in system_messages:
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                protected.append(msg)
                current_tokens += msg_tokens
        
        # 3. 动态调整保护策略（更保守的调整）
        preserve_exchanges = max(2, self.valves.preserve_recent_exchanges - iteration)
        max_preserve_tokens = int(
            target_tokens * max(0.5, self.valves.max_preserve_ratio - iteration * 0.05)
        )
        
        self.debug_log(
            1,
            f"🔄 第{iteration+1}次迭代: 保护{preserve_exchanges}轮对话, 最大{max_preserve_tokens}tokens",
            "📊",
        )
        
        # 4. 保护最近的对话轮次
        remaining_messages = [msg for msg in messages if msg not in protected]
        
        # 按时间顺序找到最近的对话轮次
        exchanges_protected = 0
        i = len(remaining_messages) - 1
        
        while (
            i >= 0
            and exchanges_protected < preserve_exchanges
            and current_tokens < max_preserve_tokens
        ):
            msg = remaining_messages[i]
            msg_tokens = self.count_message_tokens(msg)
            
            if current_tokens + msg_tokens <= max_preserve_tokens:
                if msg.get("role") == "assistant" and i > 0:
                    # 尝试保护完整的对话轮次
                    prev_msg = remaining_messages[i - 1]
                    if prev_msg.get("role") == "user":
                        prev_tokens = self.count_message_tokens(prev_msg)
                        if (
                            current_tokens + msg_tokens + prev_tokens
                            <= max_preserve_tokens
                        ):
                            protected.insert(-1, prev_msg)  # 插入到最后用户消息前
                            protected.insert(-1, msg)
                            current_tokens += msg_tokens + prev_tokens
                            exchanges_protected += 1
                            i -= 2
                            continue
                
                # 单独保护这条消息
                protected.insert(-1, msg)
                current_tokens += msg_tokens
            
            i -= 1
        
        # 5. 确定需要处理的消息
        to_process = [msg for msg in messages if msg not in protected]
        
        self.debug_log(
            1,
            f"📋 第{iteration+1}次选择: 保护{len(protected)}条({current_tokens}tokens), 处理{len(to_process)}条",
            "📝",
        )
        
        return protected, to_process

    async def _summarize_messages_impl(self, conversation_text: str, iteration: int):
        """实际的摘要实现"""
        client = self.get_summary_client()
        if not client:
            return None
        
        # 增强的摘要提示
        system_prompt = f"""你是专业的对话摘要助手。请为以下对话创建详细的结构化摘要，**必须最大化保留信息**。

摘要要求：
1. 保持对话的完整逻辑脉络和时间顺序
2. 保留所有关键信息、技术细节、参数配置、数据
3. 保留重要的问答内容和讨论要点
4. 如有图片描述，完整保留视觉信息
5. 使用清晰的结构：问题 → 回答 → 后续讨论
6. 优先级：内容完整性 > 长度限制
7. 如果内容很重要，**必须**保留，可以适当超出长度限制
8. 保留具体的配置、代码、数据、参数等技术细节

处理信息：
- 第{iteration+1}次摘要处理
- 目标：最大化信息保留，避免重要信息丢失

对话内容："""
        
        response = await client.chat.completions.create(
            model=self.valves.summary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text},
            ],
            max_tokens=self.valves.max_summary_length,
            temperature=0.05,
            timeout=self.valves.request_timeout,
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return None

    async def summarize_messages_v2(
        self, messages: List[dict], __event_emitter__, iteration: int = 0
    ) -> str:
        """增强的摘要功能 - 最大化信息保留"""
        if not messages:
            return ""
        
        # 先处理图片
        processed_messages = messages
        if self.valves.always_process_images_before_summary:
            has_images = any(
                self.has_images_in_content(msg.get("content")) for msg in messages
            )
            if has_images:
                self.debug_log(2, f"摘要前处理图片: {len(messages)}条消息", "🖼️")
                processed_messages = []
                for msg in messages:
                    if self.has_images_in_content(msg.get("content")):
                        processed_msg = await self.process_message_images(
                            msg, __event_emitter__
                        )
                        processed_messages.append(processed_msg)
                    else:
                        processed_messages.append(msg)
        
        # 按角色分组处理
        conversation_parts = []
        current_exchange = []
        
        for msg in processed_messages:
            role = msg.get("role", "unknown")
            content = self.extract_text_from_content(msg.get("content", ""))
            
            if len(content) > 6000:
                content = content[:6000] + "...(长内容已截断)"
            
            if role == "user":
                if current_exchange:
                    conversation_parts.append(self.format_exchange(current_exchange))
                    current_exchange = []
                current_exchange.append(f"👤 用户: {content}")
            elif role == "assistant":
                current_exchange.append(f"🤖 助手: {content}")
            else:
                current_exchange.append(f"[{role}]: {content}")
        
        if current_exchange:
            conversation_parts.append(self.format_exchange(current_exchange))
        
        conversation_text = "\n\n".join(conversation_parts)
        
        # 调用摘要API
        summary = await self.safe_api_call(
            self._summarize_messages_impl, "摘要生成", conversation_text, iteration
        )
        
        if summary and len(summary) >= 200:
            self.debug_log(1, f"📝 摘要生成成功: {len(summary)}字符", "📝")
            return summary
        elif summary:
            self.debug_log(1, f"⚠️ 摘要过短({len(summary)}字符)，使用原始内容", "📝")
        
        # 摘要失败或过短时，返回原始内容的截断版本
        if len(conversation_text) > 2000:
            return conversation_text[:2000] + "...(原始内容截断)"
        return conversation_text

    def format_exchange(self, exchange: List[str]) -> str:
        """格式化对话轮次"""
        return "\n".join(exchange)

    async def content_maximization_processing(
        self, messages: List[dict], target_tokens: int, __event_emitter__
    ) -> List[dict]:
        """内容最大化处理的核心逻辑"""
        current_tokens = self.count_messages_tokens(messages)
        if current_tokens <= target_tokens:
            return messages
        
        self.debug_log(
            1,
            f"🚀 开始内容最大化处理: {current_tokens} -> {target_tokens} tokens",
            "📈",
        )
        
        iteration = 0
        processed_messages = messages
        
        while iteration < self.valves.max_processing_iterations:
            current_tokens = self.count_messages_tokens(processed_messages)
            
            if current_tokens <= target_tokens:
                self.debug_log(
                    1,
                    f"✅ 内容最大化完成: 迭代{iteration+1}次, 最终{current_tokens}tokens",
                    "📈",
                )
                break
            
            await self.send_status(
                __event_emitter__,
                f"内容最大化处理 第{iteration+1}轮 ({current_tokens}→{target_tokens})",
                False,
                "📈",
            )
            
            # 智能选择消息
            protected_messages, to_process = self.smart_message_selection_v2(
                processed_messages, target_tokens, iteration
            )
            
            if not to_process:
                self.debug_log(1, f"⚠️ 没有可处理的消息，停止处理", "📝")
                break
            
            # 向量检索相关消息
            if self.valves.enable_vector_retrieval and len(to_process) > 3:
                # 使用用户最后消息作为查询
                query_msg = None
                for msg in reversed(processed_messages):
                    if msg.get("role") == "user":
                        query_msg = msg
                        break
                
                if query_msg:
                    self.debug_log(2, f"🔍 对{len(to_process)}条消息进行向量检索", "🔍")
                    relevant_messages = await self.vector_retrieve_relevant_messages(
                        query_msg, to_process, __event_emitter__
                    )
                    
                    # 重排序
                    if self.valves.enable_reranking and len(relevant_messages) > 2:
                        self.debug_log(
                            2, f"🔄 对{len(relevant_messages)}条消息进行重排序", "🔄"
                        )
                        relevant_messages = await self.rerank_messages(
                            query_msg, relevant_messages, __event_emitter__
                        )
                    
                    to_process = relevant_messages
            
            # 处理消息
            new_messages = protected_messages.copy()
            
            if to_process:
                # 按重要性分组处理
                important_messages = []
                normal_messages = []
                
                for msg in to_process:
                    msg_tokens = self.count_message_tokens(msg)
                    if msg_tokens > self.valves.max_single_message_tokens:
                        important_messages.append(msg)
                    else:
                        normal_messages.append(msg)
                
                # 处理超大消息
                for msg in important_messages:
                    summarized = await self.summarize_single_message_v2(
                        msg, __event_emitter__, iteration
                    )
                    if summarized:
                        new_messages.append(summarized)
                
                # 批量处理普通消息
                if normal_messages:
                    summary_text = await self.summarize_messages_v2(
                        normal_messages, __event_emitter__, iteration
                    )
                    if summary_text and len(summary_text) > 50:
                        summary_message = {
                            "role": "system",
                            "content": f"=== 📋 智能摘要 (第{iteration+1}轮处理) ===\n{summary_text}\n{'='*60}",
                        }
                        new_messages.append(summary_message)
                    else:
                        # 摘要失败或过短，保留更多原始消息
                        self.debug_log(1, f"❌ 摘要失败或过短，保留原始消息", "📝")
                        # 保留最重要的消息
                        keep_count = max(len(normal_messages) // 2, 3)
                        new_messages.extend(normal_messages[-keep_count:])
            
            processed_messages = new_messages
            iteration += 1
            
            # 检查进度
            new_tokens = self.count_messages_tokens(processed_messages)
            reduction = current_tokens - new_tokens
            
            self.debug_log(
                1,
                f"📊 第{iteration}轮处理: {current_tokens} -> {new_tokens} tokens (减少{reduction})",
                "📊",
            )
            
            # 更严格的停止条件
            if reduction < self.valves.min_reduction_threshold:
                self.debug_log(1, f"⚠️ 减少幅度过小({reduction}tokens)，停止处理", "📝")
                break
        
        final_tokens = self.count_messages_tokens(processed_messages)
        
        # 更保守的紧急截断
        if final_tokens > target_tokens * 1.1:  # 允许10%的超出
            self.debug_log(1, f"⚠️ 仍超出限制，启用紧急策略", "🆘")
            processed_messages = self.emergency_truncate_v2(
                processed_messages, target_tokens
            )
        
        await self.send_status(
            __event_emitter__,
            f"内容最大化完成: {final_tokens}/{target_tokens} tokens",
            True,
            "✅",
        )
        
        return processed_messages

    async def _summarize_single_message_impl(self, content: str, iteration: int):
        """实际的单条消息摘要实现"""
        client = self.get_summary_client()
        if not client:
            return None
        
        system_prompt = f"""请将以下内容进行详细摘要，**必须最大化保留关键信息**：

要求：
- 保留所有重要细节、参数、配置、数据
- 保留图片描述信息
- 保持逻辑结构完整
- 优先级：内容完整性 > 长度限制
- 如果内容很重要，可以适当超出长度限制
- 这是第{iteration+1}次处理，但仍需保留核心技术信息

目标长度：{self.valves.max_summary_length}字符（可适当超出）"""
        
        response = await client.chat.completions.create(
            model=self.valves.summary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content[:8000]},
            ],
            max_tokens=self.valves.max_summary_length,
            temperature=0.05,
            timeout=self.valves.request_timeout,
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return None

    async def summarize_single_message_v2(
        self, message: dict, __event_emitter__, iteration: int = 0
    ) -> Optional[dict]:
        """增强的单条消息摘要"""
        # 先处理图片
        processed_message = message
        if self.has_images_in_content(message.get("content")):
            processed_message = await self.process_message_images(
                message, __event_emitter__
            )
        
        content = self.extract_text_from_content(processed_message.get("content", ""))
        if not content:
            return None
        
        # 尝试API摘要
        summary = await self.safe_api_call(
            self._summarize_single_message_impl, "单条消息摘要", content, iteration
        )
        
        if summary and len(summary) > 100:
            result = processed_message.copy()
            result["content"] = f"[智能摘要] {summary}"
            return result
        
        # 失败时更保守的截断
        if len(content) > 2000:
            result = processed_message.copy()
            result["content"] = content[:2000] + "...(内容已截断)"
            return result
        
        return processed_message

    def emergency_truncate_v2(
        self, messages: List[dict], target_tokens: int
    ) -> List[dict]:
        """增强的紧急截断策略 - 更保守的处理"""
        self.debug_log(1, f"🆘 启用增强紧急截断策略", "📝")
        
        # 分类消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        result = []
        current_tokens = 0
        
        # 1. 保留系统消息
        for msg in system_messages:
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                result.append(msg)
                current_tokens += msg_tokens
        
        # 2. 强制保留用户最后消息
        if user_messages:
            last_user_msg = user_messages[-1]
            msg_tokens = self.count_message_tokens(last_user_msg)
            
            if current_tokens + msg_tokens <= target_tokens:
                result.append(last_user_msg)
                current_tokens += msg_tokens
            else:
                # 更保守的截断用户消息内容
                content = self.extract_text_from_content(
                    last_user_msg.get("content", "")
                )
                if content:
                    max_content_length = min(1000, len(content) // 2)
                    truncated_content = content[:max_content_length] + "...(紧急截断)"
                    truncated_msg = last_user_msg.copy()
                    truncated_msg["content"] = truncated_content
                    result.append(truncated_msg)
                    current_tokens += self.count_message_tokens(truncated_msg)
        
        # 3. 尽可能保留最近的assistant消息
        for msg in reversed(assistant_messages):
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                result.insert(-1, msg)  # 插入到最后用户消息前
                current_tokens += msg_tokens
            else:
                break
        
        # 4. 补充其他用户消息
        remaining_tokens = target_tokens - current_tokens
        if remaining_tokens > 200:  # 提高最小剩余token要求
            for msg in reversed(user_messages[:-1]):  # 除了最后一条
                msg_tokens = self.count_message_tokens(msg)
                if msg_tokens <= remaining_tokens:
                    result.insert(-1, msg)
                    remaining_tokens -= msg_tokens
                else:
                    break
        
        final_tokens = self.count_messages_tokens(result)
        self.debug_log(
            1, f"🆘 增强紧急截断完成: {len(result)}条消息, {final_tokens}tokens", "📝"
        )
        
        return result

    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """入口函数 - 处理请求"""
        print("\n🚀 ===== INLET CALLED =====")
        print(f"📨 收到请求: {list(body.keys())}")
        
        if not self.valves.enable_processing:
            print("❌ 处理功能已禁用")
            return body
        
        messages = body.get("messages", [])
        if not messages:
            print("❌ 无消息内容")
            return body
        
        model_name = body.get("model", "未知")
        print(f"📋 模型: {model_name}, 消息数: {len(messages)}")
        
        if self.is_model_excluded(model_name):
            print(f"🚫 模型已排除")
            return body
        
        # Token分析
        original_tokens = self.count_messages_tokens(messages)
        token_limit = self.get_model_token_limit(model_name)
        
        print(f"📊 Token: {original_tokens}/{token_limit}")
        print(f"🔧 错误重试: {self.valves.api_error_retry_times}次")
        
        try:
            # 1. 多模态处理
            processed_messages = await self.process_multimodal_content(
                messages, model_name, __event_emitter__
            )
            processed_tokens = self.count_messages_tokens(processed_messages)
            print(f"📊 多模态处理后: {processed_tokens} tokens")
            
            # 2. 内容最大化处理
            if (
                self.valves.enable_content_maximization
                and processed_tokens > token_limit
            ):
                print(f"🚀 Token超限，开始内容最大化处理...")
                final_messages = await self.content_maximization_processing(
                    processed_messages, token_limit, __event_emitter__
                )
                final_tokens = self.count_messages_tokens(final_messages)
                print(f"📊 内容最大化处理后: {final_tokens} tokens")
                
                # 计算保留比例
                retention_ratio = final_tokens / original_tokens if original_tokens > 0 else 0
                print(f"📈 内容保留比例: {retention_ratio:.2%}")
                
                if retention_ratio < 0.3:
                    print(f"⚠️ 内容保留比例过低({retention_ratio:.2%})，建议调整参数")
                
                body["messages"] = final_messages
                print("✅ 使用内容最大化处理后的消息")
            else:
                body["messages"] = processed_messages
                print("✅ 直接使用处理后的消息")
                
        except Exception as e:
            print(f"❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()
            
            await self.send_status(
                __event_emitter__,
                f"处理失败: {str(e)[:50]}",
                True,
                "❌",
            )
        
        print("🏁 ===== INLET DONE =====\n")
        return body

    async def outlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """出口函数 - 返回响应"""
        return body
