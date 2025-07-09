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
    "claude-3", "gemini-pro-vision"
}

MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-4": 8192, "gpt-3.5-turbo": 16385,
    "doubao-1.5-thinking-pro": 128000, "doubao-1.5-vision-pro": 128000,
    "claude-3": 200000, "gemini-pro": 128000,
}

class VectorStrategy(str, Enum):
    AUTO = "auto"
    MULTIMODAL_FIRST = "multimodal_first"
    TEXT_FIRST = "text_first"
    MIXED = "mixed"
    FALLBACK = "fallback"
    VISION_TO_TEXT = "vision_to_text"

class Filter:
    class Valves(BaseModel):
        # 基础配置
        enable_processing: bool = Field(default=True, description="🔄 启用插件功能")
        enable_multimodal: bool = Field(default=True, description="🖼️ 启用多模态处理")
        enable_vision_preprocessing: bool = Field(default=True, description="👁️ 启用图片预处理")
        force_truncate_first: bool = Field(default=True, description="✂️ 强制先检查截断")
        
        # 调试配置
        debug_level: int = Field(default=1, description="🐛 调试级别 0-3")
        show_frontend_progress: bool = Field(default=True, description="📱 显示处理进度")
        
        # Token管理
        default_token_limit: int = Field(default=120000, description="⚖️ 默认token限制")
        token_safety_ratio: float = Field(default=0.85, description="🛡️ Token安全比例")
        preserve_last_messages: int = Field(default=2, description="💾 保留最后消息数")
        context_preserve_ratio: float = Field(default=0.6, description="📝 上下文保留比例")
        
        # Vision配置
        vision_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="👁️ Vision API地址")
        vision_api_key: str = Field(default="", description="🔑 Vision API密钥")
        vision_model: str = Field(default="doubao-1.5-vision-pro-250328", description="🧠 Vision模型")
        vision_prompt_template: str = Field(default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。描述要准确、具体、完整，便于后续的语义检索。", description="👁️ Vision提示词")
        
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
        enable_reranking: bool = Field(default=True, description="🔄 启用重排序")
        rerank_api_base: str = Field(default="https://api.bochaai.com", description="🔄 重排序API")
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=10, description="🔝 重排序返回数量")
        
        # 摘要配置
        summary_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="📝 摘要API")
        summary_api_key: str = Field(default="", description="🔑 摘要密钥")
        summary_model: str = Field(default="doubao-1.5-thinking-pro-250415", description="🧠 摘要模型")
        max_summary_length: int = Field(default=3000, description="📏 摘要最大长度")
        max_recursion_depth: int = Field(default=3, description="🔄 最大递归深度")
        
        # 性能配置
        max_concurrent_requests: int = Field(default=3, description="⚡ 最大并发数")
        request_timeout: int = Field(default=60, description="⏱️ 请求超时(秒)")
        chunk_size: int = Field(default=1000, description="📄 分片大小")
        overlap_size: int = Field(default=100, description="🔗 重叠大小")

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAzdjE4bTktOWwtOS05LTkgOSIgLz4KPC9zdmc+"""
        
        self._vision_client = None
        self._encoding = None
        self.vision_cache = {}
        self.vector_cache = {}

    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        if self.valves.debug_level >= level:
            prefix = ["", "🐛", "🔍", "📋"][min(level, 3)]
            print(f"{prefix} {emoji} {message}")

    def get_encoding(self):
        if not TIKTOKEN_AVAILABLE or self._encoding:
            return self._encoding
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except:
            pass
        return self._encoding

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        encoding = self.get_encoding()
        if encoding:
            try:
                return len(encoding.encode(text))
            except:
                pass
        return len(text) // 4

    def count_message_tokens(self, message: dict) -> int:
        content = message.get("content", "")
        total_tokens = 0
        
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total_tokens += self.count_tokens(item.get("text", ""))
                elif item.get("type") == "image_url":
                    total_tokens += 1000
        else:
            total_tokens = self.count_tokens(str(content))
        
        return total_tokens + self.count_tokens(message.get("role", "")) + 4

    def count_messages_tokens(self, messages: List[dict]) -> int:
        return sum(self.count_message_tokens(msg) for msg in messages)

    def get_model_token_limit(self, model_name: str) -> int:
        for model_key, limit in MODEL_TOKEN_LIMITS.items():
            if model_key in model_name.lower():
                return int(limit * self.valves.token_safety_ratio)
        return int(self.valves.default_token_limit * self.valves.token_safety_ratio)

    def is_multimodal_model(self, model_name: str) -> bool:
        return any(mm in model_name.lower() for mm in MULTIMODAL_MODELS)

    def has_images_in_content(self, content) -> bool:
        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)
        return False

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        return any(self.has_images_in_content(msg.get("content")) for msg in messages)

    async def send_status(self, __event_emitter__, message: str, done: bool = True, emoji: str = "🔄"):
        if __event_emitter__ and self.valves.show_frontend_progress:
            try:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"{emoji} {message}", "done": done}
                })
            except:
                pass

    # Vision处理
    def get_vision_client(self):
        if not OPENAI_AVAILABLE or self._vision_client:
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
        return self._vision_client

    async def describe_image(self, image_url: str, __event_emitter__) -> str:
        image_hash = hashlib.md5(image_url.encode()).hexdigest()
        if image_hash in self.vision_cache:
            return self.vision_cache[image_hash]
        
        client = self.get_vision_client()
        if not client:
            return "无法处理图片：Vision服务未配置"
        
        try:
            response = await client.chat.completions.create(
                model=self.valves.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.valves.vision_prompt_template},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                max_tokens=800,
                temperature=0.2
            )
            
            if response.choices:
                description = response.choices[0].message.content.strip()
                self.vision_cache[image_hash] = description
                return description
            return "图片描述生成失败"
        except Exception as e:
            error_msg = f"图片处理错误: {str(e)[:100]}"
            self.debug_log(1, error_msg, "❌")
            return error_msg

    async def process_multimodal_content(self, messages: List[dict], model_name: str, __event_emitter__) -> List[dict]:
        """处理多模态内容：为不支持图片的模型添加视觉能力"""
        if not self.valves.enable_multimodal:
            return messages
        
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return messages
        
        is_target_multimodal = self.is_multimodal_model(model_name)
        if is_target_multimodal:
            self.debug_log(2, f"目标模型 {model_name} 支持多模态，跳过预处理", "✅")
            return messages
        
        total_images = sum(len([item for item in msg.get("content", []) if isinstance(msg.get("content"), list) and item.get("type") == "image_url"]) for msg in messages)
        
        await self.send_status(__event_emitter__, f"检测到 {total_images} 张图片，开始处理...", False, "🖼️")
        
        processed_messages = []
        processed_count = 0
        
        for message in messages:
            content = message.get("content", "")
            
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        processed_count += 1
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            description = await self.describe_image(image_url, __event_emitter__)
                            text_parts.append(f"[图片{processed_count}] {description}")
                
                processed_message = message.copy()
                processed_message["content"] = " ".join(text_parts) if text_parts else ""
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        
        await self.send_status(__event_emitter__, f"多模态处理完成：{processed_count} 张图片", True, "✅")
        return processed_messages

    # 向量化功能
    def choose_vector_model(self, has_images: bool = False) -> Tuple[str, str, str, str]:
        strategy = self.valves.vector_strategy
        
        if has_images and not self.valves.enable_multimodal_vector and self.valves.enable_text_vector:
            strategy = VectorStrategy.VISION_TO_TEXT
        
        if strategy == VectorStrategy.AUTO:
            if has_images and self.valves.enable_multimodal_vector:
                return (self.valves.multimodal_vector_api_base, self.valves.multimodal_vector_api_key, 
                       self.valves.multimodal_vector_model, "multimodal")
            elif self.valves.enable_text_vector:
                return (self.valves.text_vector_api_base, self.valves.text_vector_api_key, 
                       self.valves.text_vector_model, "text")
        
        elif strategy in [VectorStrategy.TEXT_FIRST, VectorStrategy.VISION_TO_TEXT]:
            if self.valves.enable_text_vector:
                return (self.valves.text_vector_api_base, self.valves.text_vector_api_key, 
                       self.valves.text_vector_model, "text")
        
        elif strategy == VectorStrategy.MULTIMODAL_FIRST:
            if self.valves.enable_multimodal_vector:
                return (self.valves.multimodal_vector_api_base, self.valves.multimodal_vector_api_key, 
                       self.valves.multimodal_vector_model, "multimodal")
        
        if self.valves.enable_text_vector:
            return (self.valves.text_vector_api_base, self.valves.text_vector_api_key, 
                   self.valves.text_vector_model, "text")
        elif self.valves.enable_multimodal_vector:
            return (self.valves.multimodal_vector_api_base, self.valves.multimodal_vector_api_key, 
                   self.valves.multimodal_vector_model, "multimodal")
        
        raise ValueError("没有可用的向量模型")

    async def preprocess_for_text_vector(self, content, __event_emitter__) -> str:
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url and self.valves.enable_vision_preprocessing:
                        description = await self.describe_image(image_url, __event_emitter__)
                        parts.append(f"[图片描述] {description}")
            return " ".join(parts)
        
        return str(content)

    async def vectorize_content(self, content, __event_emitter__, has_images: bool = False) -> Optional[List[float]]:
        if not HTTPX_AVAILABLE:
            return None
        
        content_str = str(content)[:50]
        cache_key = hashlib.md5(f"{content_str}_{has_images}".encode()).hexdigest()
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        try:
            api_base, api_key, model_name, model_type = self.choose_vector_model(has_images)
        except:
            return None
        
        if not api_key:
            return None
        
        text_content = content
        if model_type == "text" and (has_images or self.has_images_in_content(content)):
            text_content = await self.preprocess_for_text_vector(content, __event_emitter__)
        elif isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            text_content = " ".join(text_parts)
        
        try:
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(f"{api_base}/embeddings", 
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    json={"model": model_name, "input": str(text_content), "encoding_format": "float"})
                
                response.raise_for_status()
                result = response.json()
                
                if "data" in result and result["data"]:
                    embedding = result["data"][0]["embedding"]
                    self.vector_cache[cache_key] = embedding
                    return embedding
        except Exception as e:
            self.debug_log(1, f"向量化失败: {e}", "❌")
        
        return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        try:
            import math
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
        except:
            return 0

    # 分片和检索
    async def chunk_messages(self, messages: List[dict]) -> List[Dict]:
        chunks = []
        current_content, current_tokens, current_messages = "", 0, []
        
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)
            
            message_tokens = self.count_tokens(str(content))
            
            if current_tokens + message_tokens > self.valves.chunk_size and current_content:
                chunks.append({
                    "content": current_content,
                    "messages": current_messages.copy(),
                    "tokens": current_tokens,
                    "has_images": any(self.has_images_in_content(msg.get("content")) for msg in current_messages)
                })
                
                sentences = re.split(r'[.!?。！？]+', current_content)
                if len(sentences) > 1:
                    overlap = " ".join(sentences[-2:])
                    current_content = overlap + " " + content
                    current_tokens = self.count_tokens(current_content)
                else:
                    current_content = content
                    current_tokens = message_tokens
                current_messages = [message]
            else:
                current_content += " " + content if current_content else content
                current_tokens += message_tokens
                current_messages.append(message)
        
        if current_content:
            chunks.append({
                "content": current_content,
                "messages": current_messages,
                "tokens": current_tokens,
                "has_images": any(self.has_images_in_content(msg.get("content")) for msg in current_messages)
            })
        
        return chunks

    async def semantic_search(self, query, chunks: List[Dict], __event_emitter__) -> List[Dict]:
        if not chunks:
            return []
        
        await self.send_status(__event_emitter__, f"语义检索 {len(chunks)} 个分片...", False, "🔍")
        
        query_text = query
        query_has_images = False
        if isinstance(query, list):
            text_parts = [item.get("text", "") for item in query if item.get("type") == "text"]
            query_text = " ".join(text_parts)
            query_has_images = any(item.get("type") == "image_url" for item in query)
        
        query_vector = await self.vectorize_content(query, __event_emitter__, query_has_images)
        if not query_vector:
            return chunks[:self.valves.rerank_top_k]
        
        scored_chunks = []
        for chunk in chunks:
            chunk_vector = await self.vectorize_content(chunk["content"], __event_emitter__, chunk.get("has_images", False))
            if chunk_vector:
                similarity = self.cosine_similarity(query_vector, chunk_vector)
                threshold = self.valves.multimodal_similarity_threshold if chunk.get("has_images") else self.valves.text_similarity_threshold
                if similarity >= threshold:
                    chunk["similarity_score"] = similarity
                    scored_chunks.append(chunk)
        
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        if self.valves.enable_reranking and self.valves.rerank_api_key:
            return await self.rerank_chunks(query_text, scored_chunks[:20], __event_emitter__)
        
        return scored_chunks[:self.valves.rerank_top_k]

    async def rerank_chunks(self, query: str, chunks: List[Dict], __event_emitter__) -> List[Dict]:
        if not HTTPX_AVAILABLE:
            return chunks
        
        try:
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(f"{self.valves.rerank_api_base}/v1/rerank",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.valves.rerank_api_key}"},
                    json={
                        "model": self.valves.rerank_model,
                        "query": query,
                        "documents": [chunk["content"] for chunk in chunks],
                        "top_n": min(self.valves.rerank_top_k, len(chunks))
                    })
                
                result = response.json()
                if "data" in result and "results" in result["data"]:
                    reranked = []
                    for item in result["data"]["results"]:
                        chunk = chunks[item["index"]].copy()
                        chunk["rerank_score"] = item.get("relevance_score", 0)
                        reranked.append(chunk)
                    return reranked
        except Exception as e:
            self.debug_log(1, f"重排序失败: {e}", "❌")
        
        return chunks

    # 摘要功能
    async def recursive_summarize(self, messages: List[dict], target_tokens: int, __event_emitter__, depth: int = 0) -> List[dict]:
        if depth >= self.valves.max_recursion_depth:
            return self.preserve_essential_messages(messages)
        
        current_tokens = self.count_messages_tokens(messages)
        if current_tokens <= target_tokens:
            return messages
        
        await self.send_status(__event_emitter__, f"第{depth+1}轮摘要 ({current_tokens}→{target_tokens})", False, "📝")
        
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        protected_count = self.valves.preserve_last_messages
        protected = other_messages[-protected_count:] if len(other_messages) > protected_count else other_messages
        to_summarize = other_messages[:-protected_count] if len(other_messages) > protected_count else []
        
        if not to_summarize:
            return system_messages + protected
        
        summary_text = await self.summarize_messages(to_summarize, depth)
        summary_message = {"role": "system", "content": f"=== 历史摘要 (第{depth+1}轮) ===\n{summary_text}"}
        
        new_messages = system_messages + [summary_message] + protected
        new_tokens = self.count_messages_tokens(new_messages)
        
        if new_tokens > target_tokens:
            return await self.recursive_summarize(new_messages, target_tokens, __event_emitter__, depth + 1)
        
        return new_messages

    def preserve_essential_messages(self, messages: List[dict]) -> List[dict]:
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        return system_messages + (other_messages[-1:] if other_messages else [])

    async def summarize_messages(self, messages: List[dict], depth: int = 0) -> str:
        if not OPENAI_AVAILABLE or not self.valves.summary_api_key:
            return f"摘要失败：API不可用 | 消息数: {len(messages)}"
        
        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)
            conversation_text += f"[{i+1}] {role}: {content}\n"
        
        try:
            client = AsyncOpenAI(base_url=self.valves.summary_api_base, api_key=self.valves.summary_api_key, timeout=self.valves.request_timeout)
            
            response = await client.chat.completions.create(
                model=self.valves.summary_model,
                messages=[
                    {"role": "system", "content": f"请简洁摘要以下对话，保留关键信息。长度限制{self.valves.max_summary_length}字符。深度:{depth}"},
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=self.valves.max_summary_length,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip() if response.choices else "摘要生成失败"
        except Exception as e:
            return f"摘要失败: {str(e)[:100]} | 消息数: {len(messages)}"

    # 核心处理逻辑
    async def process_context_with_retrieval(self, messages: List[dict], model_name: str, __event_emitter__) -> List[dict]:
        token_limit = self.get_model_token_limit(model_name)
        current_tokens = self.count_messages_tokens(messages)
        
        if current_tokens <= token_limit:
            return messages
        
        # 检查向量配置
        if not self.valves.enable_multimodal_vector and not self.valves.enable_text_vector:
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        # 提取查询
        user_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        
        if not user_query:
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        # 分片和检索
        protected_count = self.valves.preserve_last_messages * 2
        protected_messages = messages[-protected_count:] if len(messages) > protected_count else messages
        history_messages = messages[:-protected_count] if len(messages) > protected_count else []
        
        if not history_messages:
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        chunks = await self.chunk_messages(history_messages)
        relevant_chunks = await self.semantic_search(user_query, chunks, __event_emitter__)
        
        # 构建增强上下文
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        enhanced_context = []
        
        if relevant_chunks:
            context_content = ""
            used_tokens = 0
            available_tokens = int(token_limit * self.valves.context_preserve_ratio)
            
            for i, chunk in enumerate(relevant_chunks):
                if used_tokens + chunk["tokens"] <= available_tokens:
                    context_content += f"\n### 相关上下文 {i+1}\n{chunk['content']}\n"
                    used_tokens += chunk["tokens"]
                else:
                    break
            
            if context_content:
                enhanced_message = {
                    "role": "system",
                    "content": f"=== 检索增强上下文 ===\n{context_content}\n\n请基于上述上下文回答用户问题。"
                }
                enhanced_context.append(enhanced_message)
        
        final_messages = system_messages + enhanced_context + protected_messages
        final_tokens = self.count_messages_tokens(final_messages)
        
        if final_tokens > token_limit:
            return await self.recursive_summarize(final_messages, token_limit, __event_emitter__)
        
        return final_messages

    # 主入口
    async def inlet(self, body: dict, user: Optional[dict] = None, __event_emitter__: Optional[Callable] = None) -> dict:
        if not self.toggle or not self.valves.enable_processing:
            return body
        
        messages = body.get("messages", [])
        if not messages:
            return body
        
        model_name = body.get("model", "")
        self.debug_log(1, f"处理开始: {len(messages)}条消息, 模型: {model_name}", "🚀")
        
        try:
            # 1. 多模态处理 (无论是否超限都进行)
            processed_messages = await self.process_multimodal_content(messages, model_name, __event_emitter__)
            
            # 2. 超限检查和处理
            if self.valves.force_truncate_first:
                final_messages = await self.process_context_with_retrieval(processed_messages, model_name, __event_emitter__)
                body["messages"] = final_messages
                
                final_tokens = self.count_messages_tokens(final_messages)
                token_limit = self.get_model_token_limit(model_name)
                
                if final_tokens <= token_limit:
                    await self.send_status(__event_emitter__, f"处理完成 ({final_tokens}/{token_limit} tokens)", True, "✅")
                else:
                    await self.send_status(__event_emitter__, "处理后仍超限，使用原始消息", True, "⚠️")
                    body["messages"] = processed_messages
            else:
                body["messages"] = processed_messages
        
        except Exception as e:
            self.debug_log(1, f"处理失败: {e}", "❌")
            await self.send_status(__event_emitter__, f"处理失败: {str(e)[:50]}", True, "❌")
        
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None, __event_emitter__: Optional[Callable] = None) -> dict:
        return body