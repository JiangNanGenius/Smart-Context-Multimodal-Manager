"""
title: 🚀 Advanced Multimodal Context Manager  
author: JiangNanGenius
version: 1.4.4
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

MULTIMODAL_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview",
    "doubao-1.5-vision-pro", "doubao-1.5-vision-lite",
    "claude-3", "gemini-pro-vision", "qwen-vl"
}

MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-4": 8192, "gpt-3.5-turbo": 16385,
    "doubao-1.5-thinking-pro": 128000, "doubao-1.5-vision-pro": 128000,
    "doubao-seed": 50000, "doubao": 50000,
    "claude-3": 200000, "gemini-pro": 128000,
}

class VectorStrategy(str, Enum):
    AUTO = "auto"
    MULTIMODAL_FIRST = "multimodal_first"
    TEXT_FIRST = "text_first"
    MIXED = "mixed"
    FALLBACK = "fallback"
    VISION_TO_TEXT = "vision_to_text"

class MultimodalStrategy(str, Enum):
    ALL_MODELS = "all_models"  # 所有模型都进行图片处理
    NON_MULTIMODAL_ONLY = "non_multimodal_only"  # 只对非多模态模型处理
    CUSTOM_LIST = "custom_list"  # 自定义模型列表
    SMART_ADAPTIVE = "smart_adaptive"  # 智能自适应

class Filter:
    class Valves(BaseModel):
        # 基础控制
        enable_processing: bool = Field(default=True, description="🔄 启用所有处理功能")
        excluded_models: str = Field(default="", description="🚫 排除模型列表(逗号分隔)")
        
        # 多模态处理策略
        multimodal_processing_strategy: MultimodalStrategy = Field(
            default=MultimodalStrategy.SMART_ADAPTIVE, 
            description="🖼️ 多模态处理策略"
        )
        force_vision_processing_models: str = Field(
            default="gpt-4,gpt-3.5-turbo,doubao-1.5-thinking-pro", 
            description="🔍 强制进行视觉处理的模型列表(逗号分隔)"
        )
        preserve_images_in_multimodal: bool = Field(
            default=True, 
            description="📸 多模态模型是否保留原始图片"
        )
        always_process_images_before_summary: bool = Field(
            default=True, 
            description="📝 摘要前总是先处理图片"
        )
        
        # 功能开关
        enable_multimodal: bool = Field(default=True, description="🖼️ 启用多模态处理")
        enable_vision_preprocessing: bool = Field(default=True, description="👁️ 启用图片预处理")
        enable_smart_truncation: bool = Field(default=True, description="✂️ 启用智能截断")
        enable_vector_retrieval: bool = Field(default=True, description="🔍 启用向量检索")
        
        # 调试
        debug_level: int = Field(default=2, description="🐛 调试级别 0-3")
        show_frontend_progress: bool = Field(default=True, description="📱 显示处理进度")
        
        # Token管理
        default_token_limit: int = Field(default=100000, description="⚖️ 默认token限制")
        token_safety_ratio: float = Field(default=0.75, description="🛡️ Token安全比例")
        
        # 保护策略
        preserve_current_query: bool = Field(default=True, description="💾 始终保护当前用户查询")
        preserve_recent_exchanges: int = Field(default=1, description="💬 保护最近完整对话轮次")
        max_preserve_ratio: float = Field(default=0.4, description="🔒 保护消息最大token比例")
        max_single_message_tokens: int = Field(default=8000, description="📝 单条消息最大token(超过则摘要)")
        context_preserve_ratio: float = Field(default=0.6, description="📝 上下文保留比例")
        
        # Vision配置
        vision_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="👁️ Vision API地址")
        vision_api_key: str = Field(default="", description="🔑 Vision API密钥")
        vision_model: str = Field(default="doubao-1.5-vision-pro-250328", description="🧠 Vision模型")
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。保持客观准确，重点突出关键信息。", 
            description="👁️ Vision提示词"
        )
        vision_max_tokens: int = Field(default=800, description="👁️ Vision最大输出tokens")
        
        # 多模态向量
        enable_multimodal_vector: bool = Field(default=True, description="🖼️ 启用多模态向量")
        multimodal_vector_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="🔗 多模态向量API")
        multimodal_vector_api_key: str = Field(default="", description="🔑 多模态向量密钥")
        multimodal_vector_model: str = Field(default="doubao-embedding-vision-250615", description="🧠 多模态向量模型")
        
        # 文本向量
        enable_text_vector: bool = Field(default=True, description="📝 启用文本向量")
        text_vector_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="🔗 文本向量API")
        text_vector_api_key: str = Field(default="", description="🔑 文本向量密钥")
        text_vector_model: str = Field(default="doubao-embedding-large-text-250515", description="🧠 文本向量模型")
        
        # 向量策略
        vector_strategy: VectorStrategy = Field(default=VectorStrategy.AUTO, description="🎯 向量化策略")
        vector_similarity_threshold: float = Field(default=0.5, description="🎯 基础相似度阈值")
        multimodal_similarity_threshold: float = Field(default=0.45, description="🖼️ 多模态相似度阈值")
        text_similarity_threshold: float = Field(default=0.55, description="📝 文本相似度阈值")
        
        # 重排序
        enable_reranking: bool = Field(default=False, description="🔄 启用重排序")
        rerank_api_base: str = Field(default="https://api.bochaai.com", description="🔄 重排序API")
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=10, description="🔝 重排序返回数量")
        
        # 摘要配置
        summary_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="📝 摘要API")
        summary_api_key: str = Field(default="", description="🔑 摘要密钥")
        summary_model: str = Field(default="doubao-1.5-thinking-pro-250415", description="🧠 摘要模型")
        max_summary_length: int = Field(default=1500, description="📏 摘要最大长度")
        max_recursion_depth: int = Field(default=3, description="🔄 最大递归深度")
        
        # 性能配置
        max_concurrent_requests: int = Field(default=3, description="⚡ 最大并发数")
        request_timeout: int = Field(default=60, description="⏱️ 请求超时(秒)")
        chunk_size: int = Field(default=800, description="📄 分片大小")
        overlap_size: int = Field(default=80, description="🔗 重叠大小")

    def __init__(self):
        print("\n" + "="*60)
        print("🚀 Advanced Multimodal Context Manager v1.4.4")
        print("📍 插件正在初始化...")
        
        self.valves = self.Valves()
        self._vision_client = None
        self._encoding = None
        self.vision_cache = {}
        self.vector_cache = {}
        self.processing_cache = {}
        
        print(f"✅ 插件初始化完成")
        print(f"🔧 处理功能: {self.valves.enable_processing}")
        print(f"🔧 多模态策略: {self.valves.multimodal_processing_strategy}")
        print(f"🔧 保护策略: 当前查询+{self.valves.preserve_recent_exchanges}轮对话")
        print(f"🔧 保护上限: {self.valves.max_preserve_ratio*100}%")
        print(f"🔧 调试级别: {self.valves.debug_level}")
        print("="*60 + "\n")

    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            print(f"{prefix} {emoji} {message}")

    def is_model_excluded(self, model_name: str) -> bool:
        if not self.valves.excluded_models or not model_name:
            return False
        
        excluded_list = [model.strip().lower() for model in self.valves.excluded_models.split(",") if model.strip()]
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
        
        return max(len(text) // 3, len(text.encode('utf-8')) // 4)

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
        for model_key, limit in MODEL_TOKEN_LIMITS.items():
            if model_key in model_name.lower():
                safe_limit = int(limit * self.valves.token_safety_ratio)
                self.debug_log(2, f"模型 {model_name} 限制: {limit} -> {safe_limit}", "⚖️")
                return safe_limit
        
        safe_limit = int(self.valves.default_token_limit * self.valves.token_safety_ratio)
        self.debug_log(1, f"未知模型 {model_name}, 使用默认限制: {safe_limit}", "⚠️")
        return safe_limit

    def is_multimodal_model(self, model_name: str) -> bool:
        """检查模型是否原生支持多模态"""
        return any(mm in model_name.lower() for mm in MULTIMODAL_MODELS)

    def should_process_images_for_model(self, model_name: str) -> bool:
        """根据策略判断是否应该为此模型处理图片"""
        if not self.valves.enable_multimodal:
            return False
        
        model_lower = model_name.lower()
        
        # 检查强制处理列表
        force_list = [m.strip().lower() for m in self.valves.force_vision_processing_models.split(",") if m.strip()]
        if any(force_model in model_lower for force_model in force_list):
            self.debug_log(2, f"模型 {model_name} 在强制处理列表中", "🔍")
            return True
        
        # 根据策略判断
        is_multimodal = self.is_multimodal_model(model_name)
        
        if self.valves.multimodal_processing_strategy == MultimodalStrategy.ALL_MODELS:
            return True
        elif self.valves.multimodal_processing_strategy == MultimodalStrategy.NON_MULTIMODAL_ONLY:
            return not is_multimodal
        elif self.valves.multimodal_processing_strategy == MultimodalStrategy.SMART_ADAPTIVE:
            # 智能自适应：非多模态模型总是处理，多模态模型在需要摘要时处理
            return True  # 让后续逻辑决定具体处理方式
        else:
            return not is_multimodal

    def has_images_in_content(self, content) -> bool:
        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)
        return False

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        return any(self.has_images_in_content(msg.get("content")) for msg in messages)

    async def send_status(self, __event_emitter__, message: str, done: bool = True, emoji: str = "🔄"):
        self.debug_log(2, f"状态: {message}", emoji)
        if __event_emitter__ and self.valves.show_frontend_progress:
            try:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"{emoji} {message}", "done": done}
                })
            except:
                pass

    def get_vision_client(self):
        if not OPENAI_AVAILABLE:
            return None
        
        if self._vision_client:
            return self._vision_client
        
        api_key = self.valves.vision_api_key
        if not api_key:
            if self.valves.multimodal_vector_api_key:
                api_key = self.valves.multimodal_vector_api_key
            elif self.valves.text_vector_api_key:
                api_key = self.valves.text_vector_api_key
        
        if api_key:
            self._vision_client = AsyncOpenAI(
                base_url=self.valves.vision_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout
            )
            self.debug_log(2, "Vision客户端已创建", "👁️")
        
        return self._vision_client

    async def describe_image(self, image_url: str, __event_emitter__) -> str:
        """描述单张图片"""
        image_hash = hashlib.md5(image_url.encode()).hexdigest()
        
        if image_hash in self.vision_cache:
            self.debug_log(3, f"使用缓存的图片描述: {image_hash[:8]}", "📋")
            return self.vision_cache[image_hash]
        
        client = self.get_vision_client()
        if not client:
            return "无法处理图片：Vision服务未配置"
        
        try:
            self.debug_log(2, f"开始识别图片: {image_hash[:8]}", "👁️")
            
            response = await client.chat.completions.create(
                model=self.valves.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.valves.vision_prompt_template},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                max_tokens=self.valves.vision_max_tokens,
                temperature=0.2
            )
            
            if response.choices:
                description = response.choices[0].message.content.strip()
                
                # 限制描述长度
                if len(description) > 600:
                    description = description[:600] + "..."
                
                self.vision_cache[image_hash] = description
                self.debug_log(2, f"图片识别完成: {len(description)}字符", "✅")
                return description
            
            return "图片描述生成失败"
            
        except Exception as e:
            error_msg = f"图片处理错误: {str(e)[:100]}"
            self.debug_log(1, error_msg, "❌")
            return error_msg

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
                    description = await self.describe_image(image_url, __event_emitter__)
                    processed_content.append(f"[图片{image_count}描述] {description}")
        
        # 创建新消息
        processed_message = message.copy()
        processed_message["content"] = "\n".join(processed_content) if processed_content else ""
        
        self.debug_log(2, f"消息图片处理完成: {image_count}张图片", "🖼️")
        return processed_message

    async def process_multimodal_content(self, messages: List[dict], model_name: str, __event_emitter__) -> List[dict]:
        """处理多模态内容的主要逻辑"""
        if not self.valves.enable_multimodal:
            return messages
        
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return messages
        
        # 检查是否需要处理图片
        should_process = self.should_process_images_for_model(model_name)
        is_multimodal = self.is_multimodal_model(model_name)
        
        self.debug_log(1, f"多模态处理检查: 模型={model_name}, 多模态={is_multimodal}, 需要处理={should_process}", "🖼️")
        
        # 如果是多模态模型且设置保留原始图片，则不处理
        if is_multimodal and self.valves.preserve_images_in_multimodal and not should_process:
            self.debug_log(2, f"多模态模型 {model_name} 保留原始图片", "📸")
            return messages
        
        # 统计图片数量
        total_images = 0
        for msg in messages:
            if isinstance(msg.get("content"), list):
                total_images += len([item for item in msg.get("content", []) if item.get("type") == "image_url"])
        
        self.debug_log(1, f"开始处理多模态内容：{total_images} 张图片", "🖼️")
        await self.send_status(__event_emitter__, f"处理 {total_images} 张图片...", False, "🖼️")
        
        # 处理所有消息
        processed_messages = []
        processed_count = 0
        
        for message in messages:
            if self.has_images_in_content(message.get("content")):
                processed_message = await self.process_message_images(message, __event_emitter__)
                processed_messages.append(processed_message)
                # 统计处理的图片数量
                if isinstance(message.get("content"), list):
                    processed_count += len([item for item in message.get("content", []) if item.get("type") == "image_url"])
            else:
                processed_messages.append(message)
        
        self.debug_log(1, f"多模态处理完成：{processed_count} 张图片", "✅")
        await self.send_status(__event_emitter__, "图片处理完成", True, "✅")
        
        return processed_messages

    def get_summary_client(self):
        """获取摘要客户端"""
        if not OPENAI_AVAILABLE:
            return None
        
        api_key = self.valves.summary_api_key
        if not api_key:
            if self.valves.multimodal_vector_api_key:
                api_key = self.valves.multimodal_vector_api_key
            elif self.valves.text_vector_api_key:
                api_key = self.valves.text_vector_api_key
        
        if api_key:
            return AsyncOpenAI(
                base_url=self.valves.summary_api_base,
                api_key=api_key,
                timeout=self.valves.request_timeout
            )
        return None

    def smart_message_selection(self, messages: List[dict], target_tokens: int) -> Tuple[List[dict], List[dict]]:
        """
        智能选择需要保护的消息和需要摘要的消息
        """
        if not messages:
            return [], []
        
        # 分离系统消息和其他消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        if not other_messages:
            return system_messages, []
        
        # 计算最大保护token数
        max_protect_tokens = int(target_tokens * self.valves.max_preserve_ratio)
        system_tokens = self.count_messages_tokens(system_messages)
        available_protect_tokens = max_protect_tokens - system_tokens
        
        self.debug_log(1, f"🔒 保护策略: 最大{max_protect_tokens}tokens, 系统消息{system_tokens}tokens, 可用{available_protect_tokens}tokens", "📊")
        
        protected = []
        current_protect_tokens = 0
        
        # 1. 始终保护最后一条用户消息（当前查询）
        current_user_msg = None
        if self.valves.preserve_current_query:
            for msg in reversed(other_messages):
                if msg.get("role") == "user":
                    current_user_msg = msg
                    msg_tokens = self.count_message_tokens(msg)
                    
                    # 如果当前用户消息过大，需要处理
                    if msg_tokens > self.valves.max_single_message_tokens:
                        self.debug_log(1, f"⚠️ 当前用户消息过大({msg_tokens}tokens)，需要截断", "📝")
                        # 截断用户消息内容
                        content = msg.get("content", "")
                        if isinstance(content, str) and len(content) > 1000:
                            truncated_content = content[:1000] + "...(用户消息已截断)"
                            current_user_msg = msg.copy()
                            current_user_msg["content"] = truncated_content
                            msg_tokens = self.count_message_tokens(current_user_msg)
                    
                    if msg_tokens <= available_protect_tokens:
                        protected.append(current_user_msg)
                        current_protect_tokens += msg_tokens
                        available_protect_tokens -= msg_tokens
                        self.debug_log(1, f"🔒 保护当前查询: {msg_tokens}tokens", "💾")
                    else:
                        self.debug_log(1, f"⚠️ 当前查询太大({msg_tokens}tokens)，无法完全保护", "⚠️")
                    break
        
        # 2. 保护最近的完整对话轮次
        remaining_messages = [msg for msg in other_messages if msg != (current_user_msg or other_messages[-1])]
        exchanges_protected = 0
        
        # 从后往前寻找完整的对话轮次
        i = len(remaining_messages) - 1
        while i >= 0 and exchanges_protected < self.valves.preserve_recent_exchanges and available_protect_tokens > 0:
            if remaining_messages[i].get("role") == "assistant" and i > 0:
                # 找到assistant回复，查找对应的user消息
                assistant_msg = remaining_messages[i]
                user_msg = remaining_messages[i-1] if remaining_messages[i-1].get("role") == "user" else None
                
                if user_msg:
                    user_tokens = self.count_message_tokens(user_msg)
                    assistant_tokens = self.count_message_tokens(assistant_msg)
                    
                    # 检查assistant消息是否过大
                    if assistant_tokens > self.valves.max_single_message_tokens:
                        self.debug_log(1, f"⚠️ Assistant消息过大({assistant_tokens}tokens)，跳过保护", "📝")
                        break
                    
                    pair_tokens = user_tokens + assistant_tokens
                    if pair_tokens <= available_protect_tokens:
                        protected.insert(0, user_msg)
                        protected.insert(1, assistant_msg)
                        current_protect_tokens += pair_tokens
                        available_protect_tokens -= pair_tokens
                        exchanges_protected += 1
                        self.debug_log(1, f"🔒 保护对话轮次{exchanges_protected}: {pair_tokens}tokens", "💾")
                        i -= 2
                    else:
                        self.debug_log(1, f"⚠️ 剩余保护token不足({available_protect_tokens}tokens)，停止保护", "📝")
                        break
                else:
                    i -= 1
            else:
                i -= 1
        
        # 3. 确定需要摘要的消息
        to_summarize = [msg for msg in other_messages if msg not in protected]
        
        total_protect_tokens = system_tokens + current_protect_tokens
        protect_ratio = total_protect_tokens / target_tokens if target_tokens > 0 else 0
        
        self.debug_log(1, f"📋 消息分配: 系统{len(system_messages)}条, 保护{len(protected)}条({current_protect_tokens}tokens), 摘要{len(to_summarize)}条", "📝")
        self.debug_log(1, f"📊 保护比例: {protect_ratio:.2%} (限制: {self.valves.max_preserve_ratio:.2%})", "📊")
        
        return system_messages + protected, to_summarize

    async def summarize_messages(self, messages: List[dict], __event_emitter__, depth: int = 0) -> str:
        """批量摘要消息 - 先处理图片再摘要"""
        if not messages:
            return ""
        
        # 如果需要，先处理图片
        processed_messages = messages
        if self.valves.always_process_images_before_summary:
            has_images = any(self.has_images_in_content(msg.get("content")) for msg in messages)
            if has_images:
                self.debug_log(2, f"摘要前处理图片: {len(messages)}条消息", "🖼️")
                processed_messages = []
                for msg in messages:
                    if self.has_images_in_content(msg.get("content")):
                        processed_msg = await self.process_message_images(msg, __event_emitter__)
                        processed_messages.append(processed_msg)
                    else:
                        processed_messages.append(msg)
        
        self.debug_log(1, f"🤖 开始调用摘要API，消息数: {len(processed_messages)}", "📝")
        
        client = self.get_summary_client()
        if not client:
            return ""
        
        # 转换消息为文本
        conversation_parts = []
        total_chars = 0
        
        for i, msg in enumerate(processed_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)
            
            content = str(content)
            
            # 适当截断过长的单条消息
            if len(content) > 3000:
                content = content[:3000] + "...(内容已截断)"
            
            # 格式化消息
            formatted_msg = f"## {role.title()}\n{content}\n"
            conversation_parts.append(formatted_msg)
            total_chars += len(formatted_msg)
            
            # 防止总长度过长
            if total_chars > 12000:
                conversation_parts.append("\n...(更多消息已省略)")
                break
        
        conversation_text = "\n".join(conversation_parts)
        self.debug_log(2, f"📝 对话文本长度: {len(conversation_text)}字符", "📝")
        
        # 改进的摘要提示
        system_prompt = f"""你是专业的对话摘要助手。请为以下对话创建一个结构化的摘要。

要求：
1. 按对话顺序整理关键信息
2. 保留重要的问题、回答和讨论要点
3. 如有技术内容，保留具体的参数、配置或方法
4. 如有图片描述，保留关键视觉信息
5. 保持逻辑性和连贯性
6. 控制在{self.valves.max_summary_length}字符以内
7. 使用清晰的结构，如：用户问题 -> 助手回答 -> 进一步讨论

原始消息数量：{len(processed_messages)}
递归深度：{depth}
请开始摘要："""
        
        try:
            response = await client.chat.completions.create(
                model=self.valves.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=self.valves.max_summary_length // 2,
                temperature=0.1,
                timeout=self.valves.request_timeout
            )
            
            if response.choices and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
                summary_length = len(summary)
                
                # 检查摘要质量
                if summary_length < 50:
                    self.debug_log(1, f"⚠️ 摘要过短({summary_length}字符)，可能质量不佳", "📝")
                    return ""
                
                self.debug_log(1, f"📝 摘要生成成功: {summary_length}字符", "📝")
                
                # 确保摘要不超长
                if summary_length > self.valves.max_summary_length:
                    summary = summary[:self.valves.max_summary_length] + "..."
                
                return summary
            else:
                self.debug_log(1, f"❌ 摘要API返回空响应", "📝")
                return ""
                
        except Exception as e:
            error_msg = f"摘要API调用失败: {str(e)[:200]}"
            self.debug_log(1, error_msg, "❌")
            return ""

    async def summarize_single_message(self, message: dict, __event_emitter__) -> dict:
        """摘要单条超长消息 - 先处理图片"""
        # 如果有图片，先处理图片
        processed_message = message
        if self.has_images_in_content(message.get("content")):
            processed_message = await self.process_message_images(message, __event_emitter__)
        
        content = processed_message.get("content", "")
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            content = " ".join(text_parts)
        
        content = str(content)
        if not content:
            return processed_message
        
        self.debug_log(1, f"🔄 摘要单条消息，长度: {len(content)}字符", "📝")
        
        # 如果内容太长，先截断
        if len(content) > 5000:
            content = content[:5000] + "...(内容已截断)"
        
        try:
            client = self.get_summary_client()
            if not client:
                # 简单截断
                truncated_content = content[:800] + "..."
                result = processed_message.copy()
                result["content"] = truncated_content
                return result
            
            system_prompt = f"""请将以下内容摘要为简洁版本，保留关键信息和重要细节（包括图片描述信息），控制在{self.valves.max_summary_length//2}字符以内："""
            
            response = await client.chat.completions.create(
                model=self.valves.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=self.valves.max_summary_length // 3,
                temperature=0.1,
                timeout=self.valves.request_timeout
            )
            
            if response.choices and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
                if len(summary) > self.valves.max_summary_length // 2:
                    summary = summary[:self.valves.max_summary_length // 2] + "..."
                
                result = processed_message.copy()
                result["content"] = f"[摘要] {summary}"
                self.debug_log(1, f"✅ 单条消息摘要完成: {len(content)} -> {len(summary)}字符", "📝")
                return result
            else:
                # 摘要失败，截断处理
                truncated_content = content[:800] + "..."
                result = processed_message.copy()
                result["content"] = truncated_content
                return result
                
        except Exception as e:
            self.debug_log(1, f"❌ 单条消息摘要失败: {e}", "📝")
            truncated_content = content[:800] + "..."
            result = processed_message.copy()
            result["content"] = truncated_content
            return result

    async def recursive_summarize(self, messages: List[dict], target_tokens: int, __event_emitter__, depth: int = 0) -> List[dict]:
        """递归摘要处理"""
        self.debug_log(1, f"🔄 开始第{depth+1}轮递归摘要", "📝")
        
        if depth >= self.valves.max_recursion_depth:
            self.debug_log(1, f"❌ 达到最大递归深度，使用紧急保护", "🔄")
            return self.emergency_truncate(messages, target_tokens)
        
        current_tokens = self.count_messages_tokens(messages)
        self.debug_log(1, f"📊 当前token: {current_tokens}, 目标: {target_tokens}", "📝")
        
        if current_tokens <= target_tokens:
            self.debug_log(1, f"✅ Token已满足要求", "📝")
            return messages
        
        await self.send_status(__event_emitter__, f"第{depth+1}轮摘要 ({current_tokens}→{target_tokens})", False, "📝")
        
        # 智能选择消息
        protected_messages, to_summarize = self.smart_message_selection(messages, target_tokens)
        
        if not to_summarize:
            self.debug_log(1, f"⚠️ 没有可摘要的消息，检查是否需要强制处理", "📝")
            # 如果保护的消息仍然超限，需要强制处理
            protected_tokens = self.count_messages_tokens(protected_messages)
            if protected_tokens > target_tokens:
                self.debug_log(1, f"⚠️ 保护消息超限({protected_tokens}>{target_tokens})，强制处理", "📝")
                return self.emergency_truncate(protected_messages, target_tokens)
            else:
                return protected_messages
        
        # 处理需要摘要的消息
        processed_messages = protected_messages.copy()
        
        # 检查是否有超大消息需要单独处理
        large_messages = []
        normal_messages = []
        
        for msg in to_summarize:
            msg_tokens = self.count_message_tokens(msg)
            if msg_tokens > self.valves.max_single_message_tokens:
                large_messages.append(msg)
            else:
                normal_messages.append(msg)
        
        # 处理超大消息
        if large_messages:
            self.debug_log(1, f"🔄 处理{len(large_messages)}条超大消息", "📝")
            for large_msg in large_messages:
                summarized_msg = await self.summarize_single_message(large_msg, __event_emitter__)
                processed_messages.append(summarized_msg)
        
        # 处理正常消息
        if normal_messages:
            self.debug_log(1, f"🔄 批量摘要{len(normal_messages)}条消息", "📝")
            summary_text = await self.summarize_messages(normal_messages, __event_emitter__, depth)
            
            if summary_text and len(summary_text) > 50:
                summary_message = {
                    "role": "system",
                    "content": f"=== 📋 对话摘要 (第{depth+1}轮) ===\n{summary_text}\n{'='*50}"
                }
                processed_messages.append(summary_message)
            else:
                # 摘要失败，保留部分重要消息
                self.debug_log(1, f"❌ 摘要质量不佳，保留重要消息", "📝")
                important_messages = normal_messages[-1:] if normal_messages else []
                processed_messages.extend(important_messages)
        
        new_tokens = self.count_messages_tokens(processed_messages)
        self.debug_log(1, f"📊 处理后token: {new_tokens}, 减少{current_tokens - new_tokens}", "📝")
        
        # 检查是否需要继续递归
        if new_tokens > target_tokens:
            self.debug_log(1, f"🔄 仍超限，继续递归", "📝")
            return await self.recursive_summarize(processed_messages, target_tokens, __event_emitter__, depth + 1)
        else:
            self.debug_log(1, f"✅ 摘要成功: {current_tokens}→{new_tokens}tokens", "📝")
            await self.send_status(__event_emitter__, f"摘要完成 ({new_tokens}/{target_tokens})", True, "✅")
            return processed_messages

    def emergency_truncate(self, messages: List[dict], target_tokens: int) -> List[dict]:
        """紧急截断策略：保留最核心的消息"""
        self.debug_log(1, f"🆘 启用紧急截断策略", "📝")
        
        # 分离系统消息和其他消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        result = system_messages.copy()
        current_tokens = self.count_messages_tokens(result)
        
        # 从后往前保留消息，优先保留用户消息
        for msg in reversed(other_messages):
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                result.append(msg)
                current_tokens += msg_tokens
            elif msg.get("role") == "user":
                # 用户消息优先保留，即使需要截断
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    truncated_msg = msg.copy()
                    truncated_msg["content"] = content[:200] + "...(紧急截断)"
                    truncated_tokens = self.count_message_tokens(truncated_msg)
                    if current_tokens + truncated_tokens <= target_tokens:
                        result.append(truncated_msg)
                        current_tokens += truncated_tokens
                break
        
        # 重新排序（保持系统消息在前）
        other_result = [msg for msg in result if msg.get("role") != "system"]
        other_result.reverse()
        final_result = system_messages + other_result
        
        final_tokens = self.count_messages_tokens(final_result)
        self.debug_log(1, f"🆘 紧急截断完成: {len(final_result)}条消息, {final_tokens}tokens", "📝")
        
        return final_result

    async def smart_truncate_messages(self, messages: List[dict], target_tokens: int, __event_emitter__) -> List[dict]:
        """智能截断消息"""
        current_tokens = self.count_messages_tokens(messages)
        
        if current_tokens <= target_tokens:
            return messages
        
        self.debug_log(1, f"开始智能截断: {current_tokens} -> {target_tokens} tokens", "✂️")
        return await self.recursive_summarize(messages, target_tokens, __event_emitter__)

    async def inlet(self, body: dict, user: Optional[dict] = None, __event_emitter__: Optional[Callable] = None) -> dict:
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
        
        try:
            # 1. 多模态处理
            should_process_images = self.should_process_images_for_model(model_name)
            is_multimodal = self.is_multimodal_model(model_name)
            
            print(f"🖼️ 图片处理策略: 应该处理={should_process_images}, 多模态模型={is_multimodal}")
            
            processed_messages = await self.process_multimodal_content(messages, model_name, __event_emitter__)
            processed_tokens = self.count_messages_tokens(processed_messages)
            print(f"📊 多模态处理后: {processed_tokens} tokens")
            
            # 2. 智能截断
            if self.valves.enable_smart_truncation and processed_tokens > token_limit:
                print(f"⚠️ Token超限，开始智能截断...")
                final_messages = await self.smart_truncate_messages(processed_messages, token_limit, __event_emitter__)
                final_tokens = self.count_messages_tokens(final_messages)
                print(f"📊 截断后: {final_tokens} tokens")
                
                # 更严格的检查
                if final_tokens <= token_limit:
                    body["messages"] = final_messages
                    print("✅ 使用截断后的消息")
                    await self.send_status(__event_emitter__, f"处理完成 ({final_tokens}/{token_limit})", True, "✅")
                else:
                    print(f"⚠️ 截断效果不佳，启用紧急策略")
                    emergency_messages = self.emergency_truncate(final_messages, token_limit)
                    emergency_tokens = self.count_messages_tokens(emergency_messages)
                    body["messages"] = emergency_messages
                    print(f"🆘 紧急处理: {emergency_tokens} tokens")
                    await self.send_status(__event_emitter__, f"紧急处理完成 ({emergency_tokens}/{token_limit})", True, "🆘")
            else:
                body["messages"] = processed_messages
                print("✅ 直接使用处理后的消息")
                
        except Exception as e:
            print(f"❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()
            await self.send_status(__event_emitter__, f"处理失败: {str(e)[:50]}", True, "❌")
        
        print("🏁 ===== INLET DONE =====\n")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None, __event_emitter__: Optional[Callable] = None) -> dict:
        """出口函数 - 返回响应"""
        return body
