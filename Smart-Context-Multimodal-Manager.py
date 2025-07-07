"""
title: 🚀 Advanced Multimodal Context Manager
author: JiangNanGenius
version: 1.0.0
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
        
        # === 🌐 向量化服务配置 ===
        vector_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🔗 向量化API基础URL"
        )
        
        vector_api_key: str = Field(
            default="",
            description="🔑 向量化API密钥"
        )
        
        vector_model: str = Field(
            default="doubao-embedding-vision-250615",
            description="🧠 向量模型名称",
            json_schema_extra={"enum": [
                "doubao-embedding-vision-250615",
                "doubao-embedding-large-text-250515",
                "doubao-embedding-large-text-240915"
            ]}
        )
        
        vector_similarity_threshold: float = Field(
            default=0.5,
            description="🎯 向量相似度阈值"
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
            default="doubao-1-5-thinking-pro-250415",
            description="🧠 摘要模型名称"
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

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAzdjE4bTktOWwtOS05LTkgOSIgLz4KPC9zdmc+"""
        
        # 初始化状态
        self._vector_client = None
        self._summary_client = None
        self._encoding = None
        self.processing_cache = {}
        
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
                        # 向量化图片
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            try:
                                description = await self.describe_image(image_url, __event_emitter__)
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

    async def describe_image(self, image_url: str, __event_emitter__) -> str:
        """使用多模态模型描述图片"""
        if not OPENAI_AVAILABLE or not self.valves.vector_api_key:
            return "无法处理图片：缺少API配置"
            
        client = AsyncOpenAI(
            base_url=self.valves.vector_api_base,
            api_key=self.valves.vector_api_key,
            timeout=self.valves.request_timeout
        )
        
        try:
            response = await client.chat.completions.create(
                model="doubao-1.5-vision-pro-250328",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请详细描述这张图片的内容，包括主要对象、场景、文字等信息。"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            else:
                return "图片描述生成失败"
                
        except Exception as e:
            self.debug_log(1, f"图片描述API调用失败: {e}", "❌")
            return f"图片处理错误: {str(e)[:100]}"

    # === 🔗 向量化和检索 ===
    async def vectorize_content(self, text: str, __event_emitter__) -> Optional[List[float]]:
        """向量化文本内容"""
        if not HTTPX_AVAILABLE or not self.valves.vector_api_key:
            return None
            
        url = f"{self.valves.vector_api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.vector_api_key}"
        }
        
        data = {
            "model": self.valves.vector_model,
            "input": text,
            "encoding_format": "float"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                if "data" in result and result["data"]:
                    return result["data"][0]["embedding"]
                else:
                    return None
                    
        except Exception as e:
            self.debug_log(1, f"向量化失败: {e}", "❌")
            return None

    async def chunk_messages_intelligently(self, messages: List[dict]) -> List[Dict]:
        """智能分片消息"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_messages = []
        
        for i, message in enumerate(messages):
            content = message.get("content", "")
            if isinstance(content, list):
                # 处理多模态内容
                content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
            
            message_tokens = self.count_tokens(content)
            
            # 检查是否需要开始新分片
            if current_tokens + message_tokens > self.valves.chunk_size and current_chunk:
                # 尝试在句子边界分割
                sentences = self.split_by_sentences(current_chunk)
                if len(sentences) > 1:
                    # 保留部分重叠
                    overlap_content = " ".join(sentences[-2:])
                    chunks.append({
                        "content": current_chunk,
                        "messages": current_messages.copy(),
                        "index": len(chunks),
                        "tokens": current_tokens
                    })
                    
                    current_chunk = overlap_content + " " + content
                    current_messages = [message]
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    # 直接分割
                    chunks.append({
                        "content": current_chunk,
                        "messages": current_messages.copy(),
                        "index": len(chunks),
                        "tokens": current_tokens
                    })
                    
                    current_chunk = content
                    current_messages = [message]
                    current_tokens = message_tokens
            else:
                current_chunk += " " + content if current_chunk else content
                current_messages.append(message)
                current_tokens += message_tokens
        
        # 添加最后一个分片
        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "messages": current_messages,
                "index": len(chunks),
                "tokens": current_tokens
            })
        
        return chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 使用正则表达式按句号、问号、感叹号分割
        sentences = re.split(r'[.!?。！？]+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def semantic_search_and_rerank(self, query: str, chunks: List[Dict], __event_emitter__) -> List[Dict]:
        """语义搜索和重排序"""
        if not chunks:
            return []
            
        await self.send_status(__event_emitter__, f"开始语义检索 {len(chunks)} 个片段...", False, "🔍")
        
        # 1. 向量化查询
        query_vector = await self.vectorize_content(query, __event_emitter__)
        if not query_vector:
            self.debug_log(1, "查询向量化失败，跳过检索", "⚠️")
            return chunks  # 返回原始chunks
        
        # 2. 计算相似度（简化版，实际应该用专门的向量数据库）
        scored_chunks = []
        for chunk in chunks:
            chunk_vector = await self.vectorize_content(chunk["content"], __event_emitter__)
            if chunk_vector:
                # 计算余弦相似度
                similarity = self.cosine_similarity(query_vector, chunk_vector)
                if similarity >= self.valves.vector_similarity_threshold:
                    chunk["similarity_score"] = similarity
                    scored_chunks.append(chunk)
        
        # 3. 按相似度排序
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 4. 重排序
        if self.valves.enable_reranking and self.valves.rerank_api_key:
            reranked_chunks = await self.rerank_chunks(query, scored_chunks[:20], __event_emitter__)
            return reranked_chunks[:self.valves.rerank_top_k]
        else:
            return scored_chunks[:self.valves.rerank_top_k]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0

    async def rerank_chunks(self, query: str, chunks: List[Dict], __event_emitter__) -> List[Dict]:
        """使用重排序服务对chunks进行重排序"""
        if not HTTPX_AVAILABLE or not self.valves.rerank_api_key:
            return chunks
            
        await self.send_status(__event_emitter__, f"正在重排序 {len(chunks)} 个片段...", False, "🔄")
        
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
                        chunk["rerank_score"] = item["relevance_score"]
                        reranked_chunks.append(chunk)
                    
                    await self.send_status(__event_emitter__, "重排序完成", True, "✅")
                    return reranked_chunks
                else:
                    self.debug_log(1, "重排序响应格式错误", "⚠️")
                    return chunks
                    
        except Exception as e:
            self.debug_log(1, f"重排序失败: {e}", "❌")
            return chunks

    # === 📝 摘要处理 ===
    async def recursive_summarize(self, messages: List[dict], target_tokens: int, __event_emitter__, depth: int = 0) -> List[dict]:
        """递归摘要处理"""
        if depth >= self.valves.max_recursion_depth:
            self.debug_log(1, f"达到最大递归深度 {depth}", "🔄")
            return messages[:self.valves.preserve_last_messages]
        
        current_tokens = self.count_messages_tokens(messages)
        if current_tokens <= target_tokens:
            return messages
            
        await self.send_status(__event_emitter__, f"第{depth+1}轮递归摘要 ({current_tokens}→{target_tokens} tokens)", False, "📝")
        
        # 分离系统消息、历史消息和最新消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # 保护最后的消息
        protected_count = self.valves.preserve_last_messages
        protected_messages = other_messages[-protected_count:] if len(other_messages) > protected_count else other_messages
        to_summarize = other_messages[:-protected_count] if len(other_messages) > protected_count else []
        
        if not to_summarize:
            # 没有可摘要的内容，只能压缩受保护的消息
            await self.send_status(__event_emitter__, "无法继续摘要，返回基础内容", True, "⚠️")
            return system_messages + protected_messages
        
        # 摘要历史消息
        summary_text = await self.summarize_messages(to_summarize, __event_emitter__, depth)
        
        # 构建新的消息列表
        summary_message = {
            "role": "system",
            "content": f"=== 历史对话摘要 (第{depth+1}轮) ===\n{summary_text}"
        }
        
        new_messages = system_messages + [summary_message] + protected_messages
        
        # 检查是否还需要继续摘要
        new_tokens = self.count_messages_tokens(new_messages)
        if new_tokens > target_tokens:
            return await self.recursive_summarize(new_messages, target_tokens, __event_emitter__, depth + 1)
        else:
            await self.send_status(__event_emitter__, f"递归摘要完成 ({current_tokens}→{new_tokens} tokens)", True, "✅")
            return new_messages

    async def summarize_messages(self, messages: List[dict], __event_emitter__, depth: int = 0) -> str:
        """摘要消息列表"""
        if not OPENAI_AVAILABLE or not self.valves.summary_api_key:
            return "无法摘要：缺少API配置"
            
        # 将消息转换为文本
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
            conversation_text += f"{role}: {content}\n\n"
        
        client = AsyncOpenAI(
            base_url=self.valves.summary_api_base,
            api_key=self.valves.summary_api_key,
            timeout=self.valves.request_timeout
        )
        
        system_prompt = f"""你是专业的对话摘要专家。请为以下对话创建简洁但完整的摘要（递归深度: {depth}）。

摘要要求：
1. 保留所有重要信息、关键决定和讨论要点
2. 保持对话的逻辑流程和因果关系
3. 如果涉及技术内容、数据或代码，务必保留核心信息
4. 摘要长度控制在{self.valves.max_summary_length}字以内
5. 使用简洁准确的语言，保持可读性
6. 按时间顺序组织内容，标明重要节点

对话内容："""

        try:
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
            
            if response.choices:
                return response.choices[0].message.content.strip()
            else:
                return "摘要生成失败：无响应"
                
        except Exception as e:
            self.debug_log(1, f"摘要生成失败: {e}", "❌")
            return f"摘要生成错误: {str(e)[:200]}"

    # === 🎯 核心处理逻辑 ===
    async def process_context_with_retrieval(self, messages: List[dict], model_name: str, __event_emitter__) -> List[dict]:
        """使用检索增强的上下文处理"""
        token_limit = self.get_model_token_limit(model_name)
        current_tokens = self.count_messages_tokens(messages)
        
        self.debug_log(1, f"开始处理 {len(messages)} 条消息 ({current_tokens}/{token_limit} tokens)", "🎯")
        
        if current_tokens <= token_limit:
            self.debug_log(1, "内容未超限，无需处理", "✅")
            return messages
        
        await self.send_status(__event_emitter__, f"内容超限 ({current_tokens}/{token_limit})，启动智能处理...", False, "🚀")
        
        # 1. 提取查询（最后一条用户消息）
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    query = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                else:
                    query = content
                break
        
        if not query:
            self.debug_log(1, "未找到用户查询，使用递归摘要", "⚠️")
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        # 2. 分片历史消息
        # 保护最后几条消息
        protected_count = self.valves.preserve_last_messages * 2  # user+assistant对
        protected_messages = messages[-protected_count:] if len(messages) > protected_count else messages
        history_messages = messages[:-protected_count] if len(messages) > protected_count else []
        
        if not history_messages:
            self.debug_log(1, "没有历史消息可处理，使用递归摘要", "⚠️")
            return await self.recursive_summarize(messages, token_limit, __event_emitter__)
        
        # 3. 智能分片
        chunks = await self.chunk_messages_intelligently(history_messages)
        self.debug_log(2, f"创建 {len(chunks)} 个智能分片", "📄")
        
        # 4. 语义检索和重排序
        relevant_chunks = await self.semantic_search_and_rerank(query, chunks, __event_emitter__)
        
        # 5. 构建增强上下文
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        enhanced_context = []
        
        if relevant_chunks:
            # 添加检索到的相关内容
            references = []
            context_content = ""
            used_tokens = 0
            available_tokens = int(token_limit * self.valves.context_preserve_ratio)
            
            for i, chunk in enumerate(relevant_chunks):
                chunk_tokens = chunk["tokens"]
                if used_tokens + chunk_tokens <= available_tokens:
                    context_content += f"\n### 📎 相关上下文 {i+1}\n{chunk['content']}\n"
                    references.append(f"[REF-{i+1}]")
                    used_tokens += chunk_tokens
                    
                    # 添加引用标记
                    chunk["reference_id"] = f"REF-{i+1}"
                else:
                    break
            
            if context_content:
                ref_list = ", ".join(references)
                enhanced_message = {
                    "role": "system",
                    "content": f"=== 🔍 检索增强上下文 ===\n基于查询检索到的相关内容 ({ref_list}):\n{context_content}\n\n💡 请基于上述上下文和对话历史回答用户问题。如果引用了上下文内容，请标注相应的引用标记。"
                }
                enhanced_context.append(enhanced_message)
        
        # 6. 组合最终消息
        final_messages = system_messages + enhanced_context + protected_messages
        final_tokens = self.count_messages_tokens(final_messages)
        
        if final_tokens > token_limit:
            # 如果还是超限，进行递归摘要
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
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """主处理入口"""
        # 检查开关
        if not self.toggle or not self.valves.enable_processing:
            return body
            
        messages = body.get("messages", [])
        if not messages:
            return body
            
        model_name = body.get("model", "")
        
        self.debug_log(1, f"开始处理模型 {model_name} 的 {len(messages)} 条消息", "🚀")
        
        try:
            # 1. 强制截断检查
            if self.valves.force_truncate_first:
                token_limit = self.get_model_token_limit(model_name)
                current_tokens = self.count_messages_tokens(messages)
                
                if current_tokens > token_limit:
                    await self.send_status(__event_emitter__, "内容超限，开始智能处理...", False, "✂️")
                    
                    # 2. 多模态处理
                    is_multimodal = self.is_multimodal_model(model_name)
                    if not is_multimodal and self.valves.enable_multimodal:
                        messages = await self.process_multimodal_content(messages, __event_emitter__)
                    
                    # 3. 上下文处理
                    processed_messages = await self.process_context_with_retrieval(
                        messages, model_name, __event_emitter__
                    )
                    
                    body["messages"] = processed_messages
                    
                    # 判断是否使用处理后的结果
                    original_tokens = self.count_messages_tokens(messages)
                    processed_tokens = self.count_messages_tokens(processed_messages)
                    
                    if processed_tokens <= token_limit:
                        await self.send_status(__event_emitter__, 
                            f"✅ 使用处理后的消息 ({original_tokens}→{processed_tokens} tokens)", True, "🎯")
                        self.debug_log(1, f"使用处理后的消息: {len(processed_messages)} 条", "✅")
                    else:
                        await self.send_status(__event_emitter__, 
                            "⚠️ 处理后仍超限，使用原始消息", True, "⚠️")
                        body["messages"] = messages
                else:
                    self.debug_log(1, "内容未超限，无需处理", "✅")
            
        except Exception as e:
            await self.send_status(__event_emitter__, f"❌ 处理失败: {str(e)}", True, "❌")
            self.debug_log(1, f"处理出错: {e}", "❌")
            if self.valves.debug_level >= 2:
                import traceback
                traceback.print_exc()
        
        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """输出后处理"""
        # 这里可以添加输出后处理逻辑，比如引用标记的美化等
        return body

    async def stream(self, event: dict) -> dict:
        """流式响应处理"""
        # 这里可以添加流式响应的实时处理逻辑
        return event
