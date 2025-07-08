"""
title: 🚀 Advanced Multimodal Context Manager
author: JiangNanGenius
version: 1.1.0
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
            description="🔄 启用长上下文和多模态处理 | 主开关，关闭后插件不工作"
        )
        
        enable_multimodal: bool = Field(
            default=True,
            description="🖼️ 启用多模态功能 | 为不支持图片的模型添加视觉能力，无论是否超限都会处理图片内容"
        )
        
        enable_vision_preprocessing: bool = Field(
            default=True,
            description="👁️ 启用图片预处理 | 当使用文本向量模型时，自动将图片转换为详细的文本描述进行向量化"
        )
        
        force_multimodal_check: bool = Field(
            default=True,
            description="🔍 强制多模态检查 | 即使未超限也检查并处理多模态内容，确保非vision模型也能理解图片"
        )
        
        enable_context_processing: bool = Field(
            default=True,
            description="📚 启用上下文处理 | 当token超限时启用检索增强或摘要压缩功能"
        )

        # === 📊 调试配置 ===
        debug_level: int = Field(
            default=1,
            description="🐛 调试级别 | 0=完全关闭 1=基础信息(推荐) 2=详细过程 3=完整调试信息"
        )
        
        show_frontend_progress: bool = Field(
            default=True,
            description="📱 显示前端处理进度 | 在OpenWebUI界面显示实时处理状态，建议开启以便了解处理进度"
        )

        # === 🎯 Token管理配置 ===
        default_token_limit: int = Field(
            default=120000,
            description="⚖️ 默认token限制 | 当模型未在内置列表中时使用此限制，建议设置为目标模型实际限制的85%"
        )
        
        token_safety_ratio: float = Field(
            default=0.85,
            description="🛡️ Token安全比例 | 实际使用限制=模型上限×此比例，预留15%空间给模型响应，范围0.7-0.9"
        )
        
        preserve_last_messages: int = Field(
            default=2,
            description="💾 强制保留的最后消息数量 | 保护最新的N对用户-助手对话不被摘要，确保上下文连贯性"
        )
        
        context_preserve_ratio: float = Field(
            default=0.6,
            description="📝 上下文保留比例 | 检索模式下，60%token用于原始上下文，40%用于检索结果，可根据需要调整"
        )

        # === 👁️ Vision预处理配置 ===
        vision_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="👁️ Vision预处理API地址 | 图片描述服务的API端点，豆包默认地址，也可使用OpenAI等兼容服务"
        )
        
        vision_api_key: str = Field(
            default="",
            description="🔑 Vision预处理API密钥 | 留空时会自动尝试使用其他已配置的API密钥"
        )
        
        vision_model: str = Field(
            default="doubao-1.5-vision-pro-250328",
            description="🧠 Vision预处理模型 | 推荐：doubao-1.5-vision-pro-250328(高质量) doubao-1.5-vision-lite-250328(快速) gpt-4o(国际)"
        )
        
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。描述要准确、具体、完整，便于后续的语义检索。",
            description="👁️ Vision模型提示词 | 用于指导图片描述的生成，可根据具体需求调整描述重点"
        )

        # === 🌐 向量化服务配置 - 多模态模型 ===
        enable_multimodal_vector: bool = Field(
            default=True,
            description="🖼️ 启用多模态向量模型 | 可直接处理图片+文本的向量模型，质量更高但速度较慢"
        )
        
        multimodal_vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 多模态向量API地址 | 豆包默认地址，也可使用其他支持多模态的向量服务"
        )
        
        multimodal_vector_api_key: str = Field(
            default="",
            description="🔑 多模态向量API密钥 | 豆包embedding服务密钥，通常与主服务密钥相同"
        )
        
        multimodal_vector_model: str = Field(
            default="doubao-embedding-vision-250615",
            description="🧠 多模态向量模型名称 | 推荐：doubao-embedding-vision-250615(最新) text-embedding-3-large(OpenAI) 等"
        )

        # === 🌐 向量化服务配置 - 文本模型 ===
        enable_text_vector: bool = Field(
            default=True,
            description="📝 启用文本向量模型 | 处理纯文本的向量模型，速度快，适合大量文本处理"
        )
        
        text_vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 文本向量API地址 | 建议与多模态API使用同一服务以简化配置"
        )
        
        text_vector_api_key: str = Field(
            default="",
            description="🔑 文本向量API密钥 | 可与多模态API共用密钥"
        )
        
        text_vector_model: str = Field(
            default="doubao-embedding-large-text-250515",
            description="🧠 文本向量模型名称 | 推荐：doubao-embedding-large-text-250515(大模型) text-embedding-3-large(OpenAI) bge-large-zh(本地)"
        )

        # === 🎯 向量化策略配置 ===
        vector_strategy: VectorStrategy = Field(
            default=VectorStrategy.AUTO,
            description="🎯 向量化策略 | auto=智能选择 multimodal_first=优先多模态 text_first=优先文本 mixed=双模型 fallback=失败重试"
        )
        
        vector_similarity_threshold: float = Field(
            default=0.5,
            description="🎯 基础相似度阈值 | 0-1之间，值越高筛选越严格，建议0.4-0.6，可根据实际效果调整"
        )
        
        multimodal_similarity_threshold: float = Field(
            default=0.45,
            description="🖼️ 多模态内容相似度阈值 | 图片内容检索阈值，通常比文本略低，因为视觉语义匹配较宽泛"
        )
        
        text_similarity_threshold: float = Field(
            default=0.55,
            description="📝 纯文本相似度阈值 | 文本检索阈值，可以设置较高以保证相关性，建议0.5-0.7"
        )

        # === 🔄 重排序配置 ===
        enable_reranking: bool = Field(
            default=True,
            description="🔄 启用语义重排序 | 使用专门的重排序模型进一步优化检索结果顺序，显著提高相关性"
        )
        
        rerank_api_base: str = Field(
            default="https://api.bochaai.com",
            description="🔄 重排序API地址 | 博查AI等重排序服务地址，也可使用其他兼容服务"
        )
        
        rerank_api_key: str = Field(
            default="",
            description="🔑 重排序API密钥 | 专门的重排序服务密钥，不填则跳过重排序步骤"
        )
        
        rerank_model: str = Field(
            default="gte-rerank",
            description="🧠 重排序模型名称 | 推荐：gte-rerank(通用) bocha-semantic-reranker-cn(中文) bge-reranker-large(开源)"
        )
        
        rerank_top_k: int = Field(
            default=10,
            description="🔝 重排序返回数量 | 最终返回给模型的检索结果数量，建议5-15个，太多会影响效果"
        )

        # === 📑 摘要配置 ===
        summary_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="📝 摘要服务API地址 | 用于生成对话摘要的服务端点"
        )
        
        summary_api_key: str = Field(
            default="",
            description="🔑 摘要服务API密钥 | 可与向量服务共用密钥"
        )
        
        summary_model: str = Field(
            default="doubao-1.5-thinking-pro-250415",
            description="🧠 摘要模型名称 | 推荐：doubao-1.5-thinking-pro(高质量摘要) gpt-4o(国际) claude-3-5-sonnet(长文本)"
        )
        
        max_summary_length: int = Field(
            default=3000,
            description="📏 单次摘要最大长度 | 每轮摘要生成的最大字符数，建议2000-5000，太短会丢失信息"
        )
        
        max_recursion_depth: int = Field(
            default=3,
            description="🔄 最大递归摘要深度 | 防止无限递归，通常3层已足够处理极长对话"
        )

        # === ⚡ 性能配置 ===
        max_concurrent_requests: int = Field(
            default=3,
            description="⚡ 最大并发请求数 | API并发限制，过高可能触发限流，建议2-5个"
        )
        
        request_timeout: int = Field(
            default=60,
            description="⏱️ API请求超时时间(秒) | 单个API调用的最大等待时间，建议30-120秒"
        )
        
        chunk_size: int = Field(
            default=1000,
            description="📄 文本分片大小(tokens) | 历史消息分片的token数量，影响检索粒度，建议500-2000"
        )
        
        overlap_size: int = Field(
            default=100,
            description="🔗 分片重叠大小(tokens) | 相邻分片的重叠部分，保证上下文连续性，建议chunk_size的10-20%"
        )

        @classmethod
        def model_validate(cls, v):
            """验证和修正配置"""
            if isinstance(v, dict):
                # 确保至少启用一个向量模型
                if not v.get('enable_multimodal_vector', True) and not v.get('enable_text_vector', True):
                    v['enable_text_vector'] = True
                    print("⚠️ 警告：至少需要启用一个向量模型，已自动启用文本向量模型")
                
                # 自动配置vision API
                if v.get('enable_vision_preprocessing', True) and not v.get('vision_api_key'):
                    if v.get('multimodal_vector_api_key'):
                        v['vision_api_key'] = v['multimodal_vector_api_key']
                        v['vision_api_base'] = v.get('multimodal_vector_api_base', v['vision_api_base'])
                        print("💡 提示：已自动使用多模态向量API配置vision服务")
                    elif v.get('text_vector_api_key'):
                        v['vision_api_key'] = v['text_vector_api_key']
                        v['vision_api_base'] = v.get('text_vector_api_base', v['vision_api_base'])
                        print("💡 提示：已自动使用文本向量API配置vision服务")
                
                # 检查重叠大小合理性
                chunk_size = v.get('chunk_size', 1000)
                overlap_size = v.get('overlap_size', 100)
                if overlap_size >= chunk_size * 0.5:
                    v['overlap_size'] = int(chunk_size * 0.2)
                    print(f"⚠️ 警告：重叠大小过大，已调整为 {v['overlap_size']}")
                
                # 检查相似度阈值合理性
                for threshold_key in ['vector_similarity_threshold', 'multimodal_similarity_threshold', 'text_similarity_threshold']:
                    threshold = v.get(threshold_key, 0.5)
                    if threshold < 0 or threshold > 1:
                        v[threshold_key] = max(0, min(1, threshold))
                        print(f"⚠️ 警告：{threshold_key} 已调整到合理范围 [0,1]")
                
                # 检查token安全比例
                safety_ratio = v.get('token_safety_ratio', 0.85)
                if safety_ratio < 0.5 or safety_ratio > 0.95:
                    v['token_safety_ratio'] = max(0.5, min(0.95, safety_ratio))
                    print(f"⚠️ 警告：token安全比例已调整为 {v['token_safety_ratio']}")
            
            return super().model_validate(v)

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAzdjE4bTktOWwtOS05LTkgOSIgLz4KPC9zdmc+"""
        
        # 初始化状态和缓存
        self._multimodal_vector_client = None
        self._text_vector_client = None
        self._summary_client = None
        self._vision_client = None
        self._encoding = None
        self.processing_cache = {}  # 处理结果缓存
        self.vision_cache = {}      # 图片描述缓存
        self.vector_cache = {}      # 向量计算缓存

    # === 🛠️ 核心工具函数 ===
    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        """分级调试日志输出"""
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            print(f"{prefix} {emoji} {message}")

    def get_encoding(self):
        """获取tiktoken编码器，用于精确计算token数量"""
        if not TIKTOKEN_AVAILABLE:
            return None
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                self.debug_log(1, f"获取tiktoken编码器失败: {e}", "⚠️")
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """精确计算文本的token数量"""
        if not text:
            return 0
        
        encoding = self.get_encoding()
        if encoding is None:
            return len(text) // 4
        
        try:
            return len(encoding.encode(text))
        except Exception as e:
            self.debug_log(2, f"Token计算失败，使用估算: {e}", "⚠️")
            return len(text) // 4

    def count_message_tokens(self, message: dict) -> int:
        """计算单个消息的token数，包括角色标识和格式开销"""
        content = message.get("content", "")
        role = message.get("role", "")
        total_tokens = 0
        
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += self.count_tokens(text)
                elif item.get("type") == "image_url":
                    total_tokens += 1000  # 图片按经验值计算
        elif isinstance(content, str):
            total_tokens = self.count_tokens(content)
        
        total_tokens += self.count_tokens(role) + 4
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数"""
        return sum(self.count_message_tokens(msg) for msg in messages)

    def get_model_token_limit(self, model_name: str) -> int:
        """获取模型的实际可用token限制"""
        limit = MODEL_TOKEN_LIMITS.get(model_name.lower())
        if limit:
            return int(limit * self.valves.token_safety_ratio)
        
        # 模糊匹配
        for model_key, model_limit in MODEL_TOKEN_LIMITS.items():
            if model_key in model_name.lower():
                self.debug_log(2, f"模型 {model_name} 匹配到 {model_key}, 限制: {model_limit}", "🎯")
                return int(model_limit * self.valves.token_safety_ratio)
        
        self.debug_log(1, f"未知模型 {model_name}, 使用默认限制: {self.valves.default_token_limit}", "⚠️")
        return int(self.valves.default_token_limit * self.valves.token_safety_ratio)

    def is_multimodal_model(self, model_name: str) -> bool:
        """检查模型是否原生支持多模态"""
        return any(mm_model in model_name.lower() for mm_model in MULTIMODAL_MODELS)

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        """检查消息列表中是否包含图片"""
        for message in messages:
            if self.has_images_in_content(message.get("content", "")):
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
        """从内容中提取所有图片URL"""
        images = []
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        images.append(image_url)
        return images

    def should_process_multimodal(self, messages: List[dict], model_name: str) -> bool:
        """判断是否需要进行多模态处理"""
        if not self.valves.enable_multimodal:
            return False
        
        # 检查是否有图片内容
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return False
        
        # 检查目标模型是否支持多模态
        model_supports_multimodal = self.is_multimodal_model(model_name)
        
        # 如果模型不支持多模态，或者强制检查开启，则需要处理
        if not model_supports_multimodal or self.valves.force_multimodal_check:
            self.debug_log(2, f"需要多模态处理: 模型支持={model_supports_multimodal}, 强制检查={self.valves.force_multimodal_check}", "🖼️")
            return True
        
        return False

    async def send_status(self, __event_emitter__, message: str, done: bool = True, emoji: str = "🔄"):
        """向前端发送处理状态更新"""
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
                self.debug_log(1, f"状态发送失败: {e}", "❌")

    # === 👁️ Vision预处理功能 ===
    def get_vision_client(self):
        """获取或创建Vision API客户端"""
        if not OPENAI_AVAILABLE:
            self.debug_log(1, "OpenAI库不可用，无法使用Vision功能", "❌")
            return None
        
        if self._vision_client is None:
            api_key = self.valves.vision_api_key
            if not api_key:
                self.debug_log(1, "Vision API密钥未配置", "⚠️")
                return None
            
            self._vision_client = AsyncOpenAI(
                base_url=self.valves.vision_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout
            )
            self.debug_log(2, f"Vision客户端已创建: {self.valves.vision_api_base}", "👁️")
        
        return self._vision_client

    async def describe_image_with_vision(self, image_url: str, __event_emitter__) -> str:
        """使用Vision模型生成图片的详细文本描述"""
        image_hash = hashlib.md5(image_url.encode()).hexdigest()
        if image_hash in self.vision_cache:
            self.debug_log(2, f"使用缓存的图片描述: {image_hash[:8]}...", "💾")
            return self.vision_cache[image_hash]
        
        client = self.get_vision_client()
        if not client:
            return "无法处理图片：Vision服务未配置"
        
        try:
            await self.send_status(__event_emitter__, f"正在分析图片内容...", False, "👁️")
            
            response = await client.chat.completions.create(
                model=self.valves.vision_model,
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
                temperature=0.2,
                timeout=self.valves.request_timeout
            )
            
            if response.choices and response.choices[0].message.content:
                description = response.choices[0].message.content.strip()
                self.vision_cache[image_hash] = description
                self.debug_log(2, f"图片描述生成成功 ({len(description)}字符): {description[:50]}...", "✅")
                return description
            else:
                error_msg = "Vision模型返回空响应"
                self.debug_log(1, error_msg, "❌")
                return f"图片描述失败: {error_msg}"
        
        except Exception as e:
            error_msg = f"Vision API调用失败: {str(e)[:100]}"
            self.debug_log(1, error_msg, "❌")
            return f"图片处理错误: {error_msg}"

    async def process_multimodal_content(self, messages: List[dict], __event_emitter__) -> List[dict]:
        """处理多模态内容：为不支持图片的模型添加视觉能力"""
        total_images = sum(len(self.extract_images_from_content(msg.get("content", ""))) 
                          for msg in messages)
        
        if total_images == 0:
            return messages
        
        await self.send_status(__event_emitter__, 
            f"检测到 {total_images} 张图片，开始多模态处理...", False, "🖼️")
        
        processed_messages = []
        processed_count = 0
        
        for i, message in enumerate(messages):
            content = message.get("content", "")
            
            if isinstance(content, list):
                text_parts = []
                image_descriptions = []
                
                for item in content:
                    if item.get("type") == "text":
                        text_content = item.get("text", "").strip()
                        if text_content:
                            text_parts.append(text_content)
                            
                    elif item.get("type") == "image_url":
                        processed_count += 1
                        image_url = item.get("image_url", {}).get("url", "")
                        
                        if image_url:
                            try:
                                await self.send_status(__event_emitter__, 
                                    f"处理第 {processed_count}/{total_images} 张图片...", False, "👁️")
                                
                                description = await self.describe_image_with_vision(image_url, __event_emitter__)
                                image_descriptions.append(f"[图片{processed_count}] {description}")
                                
                            except Exception as e:
                                self.debug_log(1, f"图片{processed_count}处理失败: {e}", "❌")
                                image_descriptions.append(f"[图片{processed_count}] 处理失败: {str(e)[:50]}")
                
                all_content = text_parts + image_descriptions
                combined_content = " ".join(all_content) if all_content else ""
                
                processed_message = message.copy()
                processed_message["content"] = combined_content
                processed_messages.append(processed_message)
                
            else:
                processed_messages.append(message)
        
        await self.send_status(__event_emitter__, 
            f"多模态处理完成：{processed_count} 张图片已转换为文本", True, "✅")
        
        return processed_messages

    # === 🔗 向量化和检索功能 ===
    def choose_vector_model(self, content_type: str = "text", has_images: bool = False) -> Tuple[str, str, str, str]:
        """根据策略和内容类型智能选择向量模型"""
        strategy = self.valves.vector_strategy
        
        if has_images and not self.valves.enable_multimodal_vector and self.valves.enable_text_vector:
            strategy = VectorStrategy.VISION_TO_TEXT
            self.debug_log(2, "检测到图片但多模态向量不可用，自动切换到VISION_TO_TEXT策略", "🔄")
        
        if strategy == VectorStrategy.AUTO:
            if has_images and self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    self.valves.multimodal_vector_model,
                    "multimodal"
                )
            elif self.valves.enable_text_vector:
                return (
                    self.valves.text_vector_api_base,
                    self.valves.text_vector_api_key,
                    self.valves.text_vector_model,
                    "text"
                )
            elif self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    self.valves.multimodal_vector_model,
                    "multimodal"
                )
        
        elif strategy == VectorStrategy.MULTIMODAL_FIRST:
            if self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    self.valves.multimodal_vector_model,
                    "multimodal"
                )
            elif self.valves.enable_text_vector:
                return (
                    self.valves.text_vector_api_base,
                    self.valves.text_vector_api_key,
                    self.valves.text_vector_model,
                    "text"
                )
        
        elif strategy in [VectorStrategy.TEXT_FIRST, VectorStrategy.VISION_TO_TEXT]:
            if self.valves.enable_text_vector:
                return (
                    self.valves.text_vector_api_base,
                    self.valves.text_vector_api_key,
                    self.valves.text_vector_model,
                    "text"
                )
            elif self.valves.enable_multimodal_vector:
                return (
                    self.valves.multimodal_vector_api_base,
                    self.valves.multimodal_vector_api_key,
                    self.valves.multimodal_vector_model,
                    "multimodal"
                )
        
        # 默认降级方案
        if self.valves.enable_text_vector:
            return (
                self.valves.text_vector_api_base,
                self.valves.text_vector_api_key,
                self.valves.text_vector_model,
                "text"
            )
        elif self.valves.enable_multimodal_vector:
            return (
                self.valves.multimodal_vector_api_base,
                self.valves.multimodal_vector_api_key,
                self.valves.multimodal_vector_model,
                "multimodal"
            )
        
        raise ValueError("没有可用的向量模型配置")

    async def preprocess_content_for_text_vector(self, content, __event_emitter__) -> str:
        """为文本向量模型预处理内容：将图片转换为文本描述"""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            processed_parts = []
            image_count = 0
            
            for item in content:
                if item.get("type") == "text":
                    text_content = item.get("text", "").strip()
                    if text_content:
                        processed_parts.append(text_content)
                        
                elif item.get("type") == "image_url":
                    image_count += 1
                    image_url = item.get("image_url", {}).get("url", "")
                    
                    if image_url and self.valves.enable_vision_preprocessing:
                        description = await self.describe_image_with_vision(image_url, __event_emitter__)
                        processed_parts.append(f"[图片{image_count}描述] {description}")
                    else:
                        processed_parts.append(f"[图片{image_count}] 无法处理或功能已禁用")
            
            if image_count > 0:
                self.debug_log(2, f"多模态内容已转换: {image_count}张图片 -> 文本描述", "🔄")
            
            return " ".join(processed_parts)
        
        return str(content)

    async def vectorize_content(self, content, __event_emitter__, content_type: str = "text", has_images: bool = False) -> Optional[List[float]]:
        """智能向量化内容，自动处理图片转文本"""
        if not HTTPX_AVAILABLE:
            self.debug_log(1, "httpx库不可用，无法进行向量化", "❌")
            return None
        
        # 生成缓存key
        content_str = str(content)[:100]
        cache_key = hashlib.md5(f"{content_str}_{content_type}_{has_images}".encode()).hexdigest()
        
        if cache_key in self.vector_cache:
            self.debug_log(3, f"使用向量缓存: {cache_key[:8]}...", "💾")
            return self.vector_cache[cache_key]
        
        try:
            api_base, api_key, model_name, model_type = self.choose_vector_model(content_type, has_images)
        except ValueError as e:
            self.debug_log(1, f"向量模型选择失败: {e}", "❌")
            return None
        
        if not api_key:
            self.debug_log(1, f"向量模型 {model_type} 缺少API密钥", "❌")
            return None
        
        # 预处理内容
        text_content = content
        preprocessing_info = ""
        
        if model_type == "text" and (has_images or self.has_images_in_content(content)):
            text_content = await self.preprocess_content_for_text_vector(content, __event_emitter__)
            preprocessing_info = " (图片→文本)"
            self.debug_log(2, f"多模态内容已转换为文本: {text_content[:100]}...", "🔄")
            
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            text_content = " ".join(text_parts)
            
        elif not isinstance(content, str):
            text_content = str(content)
        
        url = f"{api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_name,
            "input": text_content,
            "encoding_format": "float"
        }
        
        strategy_info = f"{self.valves.vector_strategy.value}{preprocessing_info}"
        self.debug_log(2, f"向量化: {model_type}模型 {model_name} | 策略: {strategy_info}", "🧠")
        
        try:
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                if "data" in result and result["data"] and result["data"][0].get("embedding"):
                    embedding = result["data"][0]["embedding"]
                    self.vector_cache[cache_key] = embedding
                    self.debug_log(3, f"向量化成功，维度: {len(embedding)}", "✅")
                    return embedding
                else:
                    self.debug_log(1, "向量化响应格式错误", "❌")
                    return None
        
        except Exception as e:
            self.debug_log(1, f"向量化失败 ({model_type}): {e}", "❌")
            if self.valves.vector_strategy == VectorStrategy.FALLBACK:
                return await self.try_fallback_vectorization(content, __event_emitter__, model_type, has_images)
            return None

    async def try_fallback_vectorization(self, content, __event_emitter__, failed_model_type: str, has_images: bool = False) -> Optional[List[float]]:
        """尝试使用备用向量化模型"""
        self.debug_log(2, f"尝试备用向量化，主模型({failed_model_type})失败", "🔄")
        
        try:
            if failed_model_type == "multimodal" and self.valves.enable_text_vector:
                api_base = self.valves.text_vector_api_base
                api_key = self.valves.text_vector_api_key
                model_name = self.valves.text_vector_model
                backup_type = "text"
                text_content = await self.preprocess_content_for_text_vector(content, __event_emitter__)
                
            elif failed_model_type == "text" and self.valves.enable_multimodal_vector:
                api_base = self.valves.multimodal_vector_api_base
                api_key = self.valves.multimodal_vector_api_key
                model_name = self.valves.multimodal_vector_model
                backup_type = "multimodal"
                
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    text_content = " ".join(text_parts)
                else:
                    text_content = str(content)
            else:
                self.debug_log(2, "没有可用的备用向量模型", "⚠️")
                return None
            
            url = f"{api_base}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": model_name,
                "input": text_content,
                "encoding_format": "float"
            }
            
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                if "data" in result and result["data"] and result["data"][0].get("embedding"):
                    self.debug_log(2, f"备用向量模型 {backup_type} 成功", "✅")
                    return result["data"][0]["embedding"]
                else:
                    return None
        
        except Exception as e:
            self.debug_log(1, f"备用向量化也失败: {e}", "❌")
            return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return max(-1.0, min(1.0, similarity))
            
        except Exception as e:
            self.debug_log(2, f"相似度计算失败: {e}", "❌")
            return 0.0

    def get_similarity_threshold(self, has_images: bool = False) -> float:
        """根据内容类型获取合适的相似度阈值"""
        if has_images:
            return self.valves.multimodal_similarity_threshold
        else:
            return self.valves.text_similarity_threshold

    # === 📚 上下文处理功能 ===
    async def chunk_messages_intelligently(self, messages: List[dict]) -> List[Dict]:
        """智能分片历史消息，保持语义完整性"""
        chunks = []
        current_chunk_content = ""
        current_chunk_tokens = 0
        current_chunk_messages = []
        
        self.debug_log(2, f"开始智能分片 {len(messages)} 条消息", "📄")
        
        for i, message in enumerate(messages):
            content = message.get("content", "")
            has_images = self.has_images_in_content(content)
            
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        text_parts.append("[图片内容]")
                content_text = " ".join(text_parts)
            else:
                content_text = str(content)
            
            message_tokens = self.count_tokens(content_text)
            
            if (current_chunk_tokens + message_tokens > self.valves.chunk_size and 
                current_chunk_content):
                
                sentences = self.split_by_sentences(current_chunk_content)
                
                if len(sentences) > 1:
                    overlap_sentences = sentences[-max(1, len(sentences) // 5):]
                    overlap_content = " ".join(overlap_sentences)
                    overlap_tokens = self.count_tokens(overlap_content)
                    
                    chunks.append({
                        "content": current_chunk_content,
                        "messages": current_chunk_messages.copy(),
                        "index": len(chunks),
                        "tokens": current_chunk_tokens,
                        "has_images": any(self.has_images_in_content(msg.get("content")) 
                                        for msg in current_chunk_messages),
                        "sentence_count": len(sentences)
                    })
                    
                    if overlap_tokens <= self.valves.overlap_size:
                        current_chunk_content = overlap_content + " " + content_text
                        current_chunk_tokens = overlap_tokens + message_tokens
                    else:
                        current_chunk_content = content_text
                        current_chunk_tokens = message_tokens
                    
                    current_chunk_messages = [message]
                    
                else:
                    chunks.append({
                        "content": current_chunk_content,
                        "messages": current_chunk_messages.copy(),
                        "index": len(chunks),
                        "tokens": current_chunk_tokens,
                        "has_images": any(self.has_images_in_content(msg.get("content")) 
                                        for msg in current_chunk_messages),
                        "sentence_count": 1
                    })
                    
                    current_chunk_content = content_text
                    current_chunk_tokens = message_tokens
                    current_chunk_messages = [message]
            else:
                if current_chunk_content:
                    current_chunk_content += " " + content_text
                else:
                    current_chunk_content = content_text
                
                current_chunk_tokens += message_tokens
                current_chunk_messages.append(message)
        
        if current_chunk_content:
            chunks.append({
                "content": current_chunk_content,
                "messages": current_chunk_messages,
                "index": len(chunks),
                "tokens": current_chunk_tokens,
                "has_images": any(self.has_images_in_content(msg.get("content")) 
                                for msg in current_chunk_messages),
                "sentence_count": len(self.split_by_sentences(current_chunk_content))
            })
        
        self.debug_log(2, f"智能分片完成: {len(chunks)}个分片, 平均{sum(c['tokens'] for c in chunks)//len(chunks) if chunks else 0}tokens/片", "✅")
        
        return chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """按句子边界分割文本，支持中英文"""
        if not text:
            return []
        
        sentences = re.split(r'[.!?。！？；;]+\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    async def semantic_search_and_rerank(self, query, chunks: List[Dict], __event_emitter__) -> List[Dict]:
        """语义搜索和重排序：从历史分片中找到最相关的内容"""
        if not chunks:
            self.debug_log(1, "没有可搜索的分片", "⚠️")
            return []
        
        await self.send_status(__event_emitter__, 
            f"开始语义检索 {len(chunks)} 个分片...", False, "🔍")
        
        # 预处理查询内容
        query_text = query
        query_has_images = False
        
        if isinstance(query, list):
            text_parts = []
            for item in query:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    query_has_images = True
            query_text = " ".join(text_parts)
        else:
            query_has_images = self.has_images_in_content(query)
        
        self.debug_log(2, f"查询内容: {query_text[:100]}... (包含图片: {query_has_images})", "🔍")
        
        # 向量化查询
        query_vector = await self.vectorize_content(query, __event_emitter__, "query", query_has_images)
        if not query_vector:
            self.debug_log(1, "查询向量化失败，跳过语义检索", "⚠️")
            return chunks[:self.valves.rerank_top_k]
        
        # 计算相似度并筛选
        scored_chunks = []
        processed_count = 0
        
        for chunk in chunks:
            processed_count += 1
            if processed_count % 5 == 0:
                await self.send_status(__event_emitter__, 
                    f"计算相似度... ({processed_count}/{len(chunks)})", False, "🧮")
            
            chunk_has_images = chunk.get("has_images", False)
            chunk_threshold = self.get_similarity_threshold(chunk_has_images)
            
            try:
                chunk_vector = await self.vectorize_content(
                    chunk["content"], __event_emitter__, "chunk", chunk_has_images
                )
                if chunk_vector:
                    similarity = self.cosine_similarity(query_vector, chunk_vector)
                    if similarity >= chunk_threshold:
                        chunk_copy = chunk.copy()
                        chunk_copy["similarity_score"] = similarity
                        scored_chunks.append(chunk_copy)
                        
                        self.debug_log(3, f"分片{chunk['index']}: 相似度{similarity:.3f} (阈值{chunk_threshold:.3f})", "📊")
            except Exception as e:
                self.debug_log(3, f"分片{chunk['index']}向量化失败: {e}", "⚠️")
                continue
        
        # 按相似度排序
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        await self.send_status(__event_emitter__, 
            f"语义检索完成：{len(scored_chunks)}/{len(chunks)} 个分片通过筛选", True, "✅")
        
        # 重排序优化
        if (self.valves.enable_reranking and 
            self.valves.rerank_api_key and 
            len(scored_chunks) > 1):
            
            top_chunks = scored_chunks[:20]
            reranked_chunks = await self.rerank_chunks(query_text, top_chunks, __event_emitter__)
            return reranked_chunks[:self.valves.rerank_top_k]
        else:
            return scored_chunks[:self.valves.rerank_top_k]

    async def rerank_chunks(self, query: str, chunks: List[Dict], __event_emitter__) -> List[Dict]:
        """使用专门的重排序服务进一步优化检索结果"""
        if not HTTPX_AVAILABLE or not self.valves.rerank_api_key:
            self.debug_log(2, "重排序服务不可用，跳过此步骤", "⚠️")
            return chunks
        
        await self.send_status(__event_emitter__, 
            f"正在重排序 {len(chunks)} 个片段...", False, "🔄")
        
        url = f"{self.valves.rerank_api_base}/v1/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.rerank_api_key}"
        }
        
        documents = [chunk["content"] for chunk in chunks]
        
        data = {
            "model": self.valves.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(self.valves.rerank_top_k, len(documents)),
            "return_documents": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                if "data" in result and "results" in result["data"]:
                    reranked_chunks = []
                    for item in result["data"]["results"]:
                        original_index = item["index"]
                        chunk = chunks[original_index].copy()
                        chunk["rerank_score"] = item.get("relevance_score", item.get("score", 0))
                        reranked_chunks.append(chunk)
                    
                    await self.send_status(__event_emitter__, 
                        f"重排序完成：优化了 {len(reranked_chunks)} 个结果", True, "✅")
                    
                    self.debug_log(2, f"重排序成功，返回{len(reranked_chunks)}个结果", "🔄")
                    return reranked_chunks
                else:
                    self.debug_log(1, "重排序响应格式错误，使用原始排序", "⚠️")
                    return chunks
        
        except Exception as e:
            self.debug_log(1, f"重排序失败: {e}，使用原始排序", "❌")
            return chunks

    # === 📝 摘要处理功能 ===
    async def recursive_summarize(self, messages: List[dict], target_tokens: int, __event_emitter__, depth: int = 0) -> List[dict]:
        """递归摘要处理：当内容仍然超限时进行多轮摘要"""
        if depth >= self.valves.max_recursion_depth:
            self.debug_log(1, f"达到最大递归深度 {depth}，强制截断", "🔄")
            preserved = self.preserve_essential_messages(messages)
            return preserved
        
        current_tokens = self.count_messages_tokens(messages)
        if current_tokens <= target_tokens:
            self.debug_log(2, f"递归摘要depth={depth}: 已满足token限制", "✅")
            return messages
        
        await self.send_status(__event_emitter__, 
            f"第{depth+1}轮递归摘要 ({current_tokens}→{target_tokens} tokens)", False, "📝")
        
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        protected_count = self.valves.preserve_last_messages
        protected_messages = other_messages[-protected_count:] if len(other_messages) > protected_count else other_messages
        to_summarize = other_messages[:-protected_count] if len(other_messages) > protected_count else []
        
        if not to_summarize:
            await self.send_status(__event_emitter__, 
                "无法继续摘要，保留核心内容", True, "⚠️")
            return system_messages + protected_messages
        
        self.debug_log(2, f"摘要 {len(to_summarize)} 条消息，保护 {len(protected_messages)} 条", "📝")
        summary_text = await self.summarize_messages(to_summarize, __event_emitter__, depth)
        
        summary_message = {
            "role": "system",
            "content": f"=== 📋 历史对话摘要 (第{depth+1}轮) ===\n{summary_text}\n{'='*50}"
        }
        
        new_messages = system_messages + [summary_message] + protected_messages
        new_tokens = self.count_messages_tokens(new_messages)
        
        if new_tokens > target_tokens:
            self.debug_log(2, f"递归摘要后仍超限 ({new_tokens}>{target_tokens})，继续下一轮", "🔄")
            return await self.recursive_summarize(new_messages, target_tokens, __event_emitter__, depth + 1)
        else:
            await self.send_status(__event_emitter__, 
                f"递归摘要完成 ({current_tokens}→{new_tokens} tokens)", True, "✅")
            self.debug_log(1, f"递归摘要成功: depth={depth}, {current_tokens}→{new_tokens} tokens", "📝")
            return new_messages

    def preserve_essential_messages(self, messages: List[dict]) -> List[dict]:
        """保留最核心的消息：系统消息+最后一对对话"""
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        essential_others = other_messages[-1:] if other_messages else []
        
        return system_messages + essential_others

    async def summarize_messages(self, messages: List[dict], __event_emitter__, depth: int = 0) -> str:
        """使用AI模型生成对话摘要"""
        if not OPENAI_AVAILABLE:
            return "无法生成摘要：OpenAI库不可用"
        
        api_key = self.valves.summary_api_key
        if not api_key:
            return "无法生成摘要：API密钥未配置"
        
        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = []
                image_count = 0
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        image_count += 1
                        text_parts.append(f"[包含图片{image_count}]")
                content = " ".join(text_parts)
            
            conversation_text += f"\n[{i+1}] {role}: {content}\n"
        
        client = AsyncOpenAI(
            base_url=self.valves.summary_api_base,
            api_key=api_key,
            timeout=self.valves.request_timeout
        )
        
        system_prompt = f"""你是专业的对话摘要专家。请为以下对话创建简洁而完整的摘要。

摘要要求：
1. **保留关键信息**：重要决定、技术细节、数据、代码片段、链接等
2. **保持逻辑顺序**：按时间顺序组织，标明重要转折点
3. **突出核心主题**：识别主要讨论话题和子话题
4. **保留上下文**：维持对话的因果关系和背景信息
5. **控制长度**：摘要长度不超过{self.valves.max_summary_length}字符
6. **结构化表达**：使用标题、要点等方式组织内容

当前递归深度：{depth}
原始对话条数：{len(messages)}

对话内容："""
        
        try:
            await self.send_status(__event_emitter__, 
                f"生成对话摘要... ({len(conversation_text)}字符)", False, "🤖")
            
            response = await client.chat.completions.create(
                model=self.valves.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=self.valves.max_summary_length,
                temperature=0.2,
                stream=False
            )
            
            if response.choices and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
                self.debug_log(2, f"摘要生成成功 ({len(summary)}字符): {summary[:100]}...", "📝")
                return summary
            else:
                return "摘要生成失败：模型返回空响应"
        
        except Exception as e:
            error_msg = f"摘要生成失败: {str(e)[:200]}"
            self.debug_log(1, error_msg, "❌")
            
            fallback_summary = f"""对话摘要（降级版本）：
- 总消息数：{len(messages)}
- 时间跨度：从第1条到第{len(messages)}条消息
- 主要参与者：{', '.join(set(msg.get('role', 'unknown') for msg in messages))}
- 错误信息：{str(e)[:100]}
注：由于API调用失败，此为自动生成的简化摘要。"""
            
            return fallback_summary

    # === 🎯 核心处理逻辑 ===
    async def process_context_with_retrieval(self, messages: List[dict], model_name: str, __event_emitter__) -> List[dict]:
        """使用检索增强的智能上下文处理"""
        # 验证配置
        if not self.valves.enable_context_processing:
            self.debug_log(1, "上下文处理已禁用", "⚠️")
            return messages
        
        # 只有当需要向量检索时才验证向量配置
        if not self.valves.enable_multimodal_vector and not self.valves.enable_text_vector:
            self.debug_log(1, "向量模型均未启用，使用纯摘要模式", "⚠️")
            token_limit = self.get_model_token_limit(model_name)
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        token_limit = self.get_model_token_limit(model_name)
        current_tokens = self.count_messages_tokens(messages)
        
        self.debug_log(1, f"上下文处理: {len(messages)}条消息, {current_tokens}/{token_limit} tokens, 模型:{model_name}", "🎯")
        
        if current_tokens <= token_limit:
            self.debug_log(1, "内容未超限，无需上下文处理", "✅")
            return messages
        
        await self.send_status(__event_emitter__, 
            f"内容超限 ({current_tokens}/{token_limit})，启动上下文处理...", False, "🚀")
        
        # 提取用户查询
        user_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        
        if not user_query:
            self.debug_log(1, "未找到用户查询，使用递归摘要", "⚠️")
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        # 分片历史消息
        protected_count = self.valves.preserve_last_messages * 2
        protected_messages = messages[-protected_count:] if len(messages) > protected_count else messages
        history_messages = messages[:-protected_count] if len(messages) > protected_count else []
        
        if not history_messages:
            self.debug_log(1, "没有历史消息可处理，使用递归摘要", "⚠️")
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        # 智能分片
        chunks = await self.chunk_messages_intelligently(history_messages)
        self.debug_log(2, f"创建 {len(chunks)} 个智能分片", "📄")
        
        # 语义检索和重排序
        relevant_chunks = await self.semantic_search_and_rerank(user_query, chunks, __event_emitter__)
        
        # 构建增强上下文
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        enhanced_context = []
        
        if relevant_chunks:
            references = []
            context_content = ""
            used_tokens = 0
            available_tokens = int(token_limit * self.valves.context_preserve_ratio)
            
            for i, chunk in enumerate(relevant_chunks):
                chunk_tokens = chunk["tokens"]
                if used_tokens + chunk_tokens <= available_tokens:
                    model_info = ""
                    if chunk.get("similarity_score"):
                        model_info = f"[相似度:{chunk['similarity_score']:.3f}] "
                    
                    context_content += f"\n### 📎 相关上下文 {i+1} {model_info}\n{chunk['content']}\n"
                    references.append(f"[REF-{i+1}]")
                    used_tokens += chunk_tokens
                    
                    chunk["reference_id"] = f"REF-{i+1}"
                else:
                    break
            
            if context_content:
                ref_list = ", ".join(references)
                strategy_info = f"策略: {self.valves.vector_strategy.value}"
                enhanced_message = {
                    "role": "system",
                    "content": f"=== 🔍 检索增强上下文 ===\n基于查询检索到的相关内容 ({ref_list}) | {strategy_info}:\n{context_content}\n\n💡 请基于上述上下文和对话历史回答用户问题。如果引用了上下文内容，请标注相应的引用标记。"
                }
                enhanced_context.append(enhanced_message)
        
        # 组合最终消息
        final_messages = system_messages + enhanced_context + protected_messages
        final_tokens = self.count_messages_tokens(final_messages)
        
        if final_tokens > token_limit:
            await self.send_status(__event_emitter__, "增强上下文仍超限，启动递归摘要...", False, "🔄")
            return await self.recursive_summarize(final_messages, token_limit, __event_emitter__)
        else:
            await self.send_status(__event_emitter__, f"上下文处理完成 ({current_tokens}→{final_tokens} tokens)", True, "🎉")
            self.debug_log(1, f"最终结果: {len(final_messages)} 条消息, {final_tokens} tokens", "🎉")
            return final_messages

    # === 🚀 主要入口函数 ===
    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """主处理入口 - 新的处理流程"""
        # 检查总开关
        if not self.toggle or not self.valves.enable_processing:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        model_name = body.get("model", "")
        self.debug_log(1, f"🚀 开始处理: 模型={model_name}, 消息数={len(messages)}", "🚀")

        try:
            original_messages = messages.copy()
            processed_messages = messages
            
            # === 第一步：多模态处理（无论是否超限都要检查） ===
            if self.should_process_multimodal(processed_messages, model_name):
                await self.send_status(__event_emitter__, "开始多模态内容处理...", False, "🖼️")
                processed_messages = await self.process_multimodal_content(processed_messages, __event_emitter__)
                self.debug_log(1, f"多模态处理完成: {len(messages)}→{len(processed_messages)} 条消息", "🖼️")
            
            # === 第二步：检查token限制，决定是否需要上下文处理 ===
            token_limit = self.get_model_token_limit(model_name)
            current_tokens = self.count_messages_tokens(processed_messages)
            
            self.debug_log(1, f"Token检查: {current_tokens}/{token_limit} ({current_tokens/token_limit*100:.1f}%)", "📊")
            
            if current_tokens > token_limit and self.valves.enable_context_processing:
                await self.send_status(__event_emitter__, f"内容超限，启动上下文处理...", False, "📚")
                processed_messages = await self.process_context_with_retrieval(
                    processed_messages, model_name, __event_emitter__
                )
                self.debug_log(1, f"上下文处理完成: {current_tokens}→{self.count_messages_tokens(processed_messages)} tokens", "📚")
            
            # === 第三步：更新body并返回结果 ===
            final_tokens = self.count_messages_tokens(processed_messages)
            
            # 统计处理效果
            original_tokens = self.count_messages_tokens(original_messages)
            has_images = self.has_images_in_messages(original_messages)
            
            self.debug_log(1, f"✅ 处理完成: {original_tokens}→{final_tokens} tokens, 图片={has_images}, 超限处理={'是' if current_tokens > token_limit else '否'}", "🎉")
            
            # 向前端发送最终状态
            if original_tokens != final_tokens or has_images:
                processing_info = []
                if has_images:
                    processing_info.append("多模态转换")
                if original_tokens != final_tokens:
                    processing_info.append(f"上下文优化({original_tokens}→{final_tokens})")
                
                await self.send_status(__event_emitter__, 
                    f"处理完成：{', '.join(processing_info)}", True, "🎉")
            
            body["messages"] = processed_messages
            return body
            
        except Exception as e:
            await self.send_status(__event_emitter__, f"❌ 处理失败: {str(e)}", True, "❌")
            self.debug_log(1, f"处理出错: {e}", "❌")
            if self.valves.debug_level >= 2:
                import traceback
                traceback.print_exc()
            
            # 发生错误时返回原始消息
            return body

    async def outlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """输出后处理"""
        # 这里可以添加输出后处理逻辑，比如引用标记的美化等
        return body
