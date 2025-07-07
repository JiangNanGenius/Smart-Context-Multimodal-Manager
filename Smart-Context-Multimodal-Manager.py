"""
title: 🧠 Smart Context & Multimodal Manager
author: Advanced AI Assistant
version: 1.0.0
license: MIT
required_open_webui_version: 0.6.0
description: 智能长上下文和多模态内容处理器，支持向量化检索、语义重排序和多模态理解
"""

import json
import hashlib
import asyncio
import re
import base64
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime
import urllib.parse

# 导入所需库
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


class Filter:
    class Valves(BaseModel):
        # 🔧 基础配置
        enable_context_management: bool = Field(
            default=True, 
            description="🧠 启用智能上下文管理"
        )
        
        enable_multimodal_processing: bool = Field(
            default=True, 
            description="🖼️ 启用多模态内容处理"
        )
        
        # 📊 Token管理配置
        default_token_limit: int = Field(
            default=32000, 
            description="🎯 默认token限制（建议设置为模型上下文的70-80%）"
        )
        
        model_token_limits: str = Field(
            default='{"gpt-4": 32000, "gpt-3.5-turbo": 16000, "claude-3": 200000, "doubao": 32000}',
            description="📝 模型特定token限制配置（JSON格式）"
        )
        
        preserve_last_messages_count: int = Field(
            default=2,
            description="🔒 强制保留最后N条消息的完整性"
        )
        
        context_preserve_ratio: float = Field(
            default=0.6,
            description="📊 上下文保留比例（0.6表示保留60%原文，40%向量化检索）"
        )
        
        # 🔍 向量化配置
        enable_vectorization: bool = Field(
            default=True,
            description="🔍 启用向量化检索功能"
        )
        
        vector_api_base_url: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🌐 向量化API基础URL"
        )
        
        vector_api_key: str = Field(
            default="",
            description="🔑 向量化API密钥"
        )
        
        vector_model: str = Field(
            default="doubao-embedding-large-text-250515",
            description="🤖 向量化模型名称"
        )
        
        multimodal_vector_model: str = Field(
            default="doubao-embedding-vision-250615",
            description="🎨 多模态向量化模型名称"
        )
        
        # 🎯 检索配置
        retrieval_top_k: int = Field(
            default=10,
            description="🎯 检索Top-K候选数量"
        )
        
        chunk_size: int = Field(
            default=1000,
            description="📦 文本分片大小（token数）"
        )
        
        chunk_overlap: int = Field(
            default=200,
            description="🔄 分片重叠大小（token数）"
        )
        
        # 🔄 重排序配置
        enable_rerank: bool = Field(
            default=True,
            description="🔄 启用语义重排序"
        )
        
        rerank_api_url: str = Field(
            default="https://api.bochaai.com/v1/rerank",
            description="🔄 重排序API地址"
        )
        
        rerank_api_key: str = Field(
            default="",
            description="🔑 重排序API密钥"
        )
        
        rerank_model: str = Field(
            default="gte-rerank",
            description="🤖 重排序模型名称"
        )
        
        rerank_top_n: int = Field(
            default=5,
            description="🎯 重排序返回数量"
        )
        
        # 🖼️ 多模态配置
        vision_model: str = Field(
            default="doubao-1.5-vision-pro-250328",
            description="👁️ 视觉理解模型"
        )
        
        vision_api_base_url: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🌐 视觉模型API基础URL"
        )
        
        vision_api_key: str = Field(
            default="",
            description="🔑 视觉模型API密钥"
        )
        
        models_with_native_vision: str = Field(
            default='["gpt-4-vision", "gpt-4o", "claude-3", "doubao-1.5-vision"]',
            description="👁️ 原生支持视觉的模型列表（JSON格式）"
        )
        
        # 🐛 调试配置
        debug_level: int = Field(
            default=1,
            description="🐛 调试级别：0=关闭，1=基础，2=详细，3=完整"
        )
        
        show_processing_status: bool = Field(
            default=True,
            description="📊 显示处理状态信息"
        )
        
        # ⚙️ 高级配置
        max_concurrent_requests: int = Field(
            default=3,
            description="🚀 最大并发请求数"
        )
        
        request_timeout: int = Field(
            default=60,
            description="⏱️ API请求超时时间（秒）"
        )
        
        intelligent_chunking: bool = Field(
            default=True,
            description="🧠 启用智能分片（按句子、段落分片）"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True  # UI开关
        self.icon = """xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1OC0xLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+"""
        
        # 缓存和状态
        self._encoding = None
        self._vector_client = None
        self._vision_client = None
        self._vector_store = {}  # 简单内存向量存储
        self._processing_stats = {}

    def debug_log(self, level: int, message: str, emoji: str = "🔍"):
        """分级debug日志"""
        if self.valves.debug_level >= level:
            prefix = ["", f"{emoji}[DEBUG]", f"{emoji}[DETAIL]", f"{emoji}[VERBOSE]"][min(level, 3)]
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{prefix} [{timestamp}] {message}")

    def get_encoding(self):
        """获取tiktoken编码器"""
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
            return len(text) // 4  # 粗略估算
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
                    total_tokens += 1000  # 图像大概估算
        elif isinstance(content, str):
            total_tokens = self.count_tokens(content)

        total_tokens += self.count_tokens(role) + 4
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数"""
        return sum(self.count_message_tokens(msg) for msg in messages)

    def get_model_token_limit(self, model_name: str) -> int:
        """获取模型特定的token限制"""
        try:
            model_limits = json.loads(self.valves.model_token_limits)
            for model_key in model_limits:
                if model_key.lower() in model_name.lower():
                    return model_limits[model_key]
        except:
            pass
        return self.valves.default_token_limit

    def has_native_vision(self, model_name: str) -> bool:
        """检查模型是否原生支持视觉"""
        try:
            native_models = json.loads(self.valves.models_with_native_vision)
            return any(model.lower() in model_name.lower() for model in native_models)
        except:
            return False

    def extract_images_from_messages(self, messages: List[dict]) -> List[Tuple[int, dict]]:
        """提取消息中的图像"""
        images = []
        for msg_idx, message in enumerate(messages):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        images.append((msg_idx, item))
        return images

    def smart_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """智能文本分片"""
        if not self.valves.intelligent_chunking:
            # 简单分片
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks
        
        # 智能分片 - 按句子和段落
        chunks = []
        sentences = re.split(r'[.!?。！？]\s+', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # 当前块已满，保存并开始新块
                chunks.append(current_chunk.strip())
                # 保留重叠
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-overlap//4:])  # 粗略重叠
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    async def get_vector_client(self):
        """获取向量化客户端"""
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI库未安装")
        if not self.valves.vector_api_key:
            raise Exception("向量化API密钥未配置")
        
        if self._vector_client is None:
            self._vector_client = AsyncOpenAI(
                base_url=self.valves.vector_api_base_url,
                api_key=self.valves.vector_api_key,
                timeout=self.valves.request_timeout,
            )
        return self._vector_client

    async def get_vision_client(self):
        """获取视觉理解客户端"""
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI库未安装")
        if not self.valves.vision_api_key:
            raise Exception("视觉API密钥未配置")
        
        if self._vision_client is None:
            self._vision_client = AsyncOpenAI(
                base_url=self.valves.vision_api_base_url,
                api_key=self.valves.vision_api_key,
                timeout=self.valves.request_timeout,
            )
        return self._vision_client

    async def vectorize_text(self, text: str, is_multimodal: bool = False) -> List[float]:
        """文本向量化"""
        try:
            client = await self.get_vector_client()
            model = self.valves.multimodal_vector_model if is_multimodal else self.valves.vector_model
            
            response = await client.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
        except Exception as e:
            self.debug_log(1, f"❌ 向量化失败: {str(e)}")
            return []

    async def vectorize_multimodal_content(self, content_list: List[dict]) -> List[float]:
        """多模态内容向量化"""
        try:
            client = await self.get_vector_client()
            
            # 构建多模态输入
            multimodal_input = []
            for item in content_list:
                if item.get("type") == "text":
                    multimodal_input.append({
                        "type": "text",
                        "text": item.get("text", "")
                    })
                elif item.get("type") == "image_url":
                    multimodal_input.append({
                        "type": "image_url",
                        "image_url": item["image_url"]
                    })
            
            response = await client.embeddings.create(
                model=self.valves.multimodal_vector_model,
                input=multimodal_input,
                encoding_format="float"
            )
            
            return response.data[0].embedding
        except Exception as e:
            self.debug_log(1, f"❌ 多模态向量化失败: {str(e)}")
            return []

    async def describe_image(self, image_item: dict) -> str:
        """图像描述生成"""
        try:
            client = await self.get_vision_client()
            
            response = await client.chat.completions.create(
                model=self.valves.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、风格等重要信息。"
                            },
                            {
                                "type": "image_url",
                                "image_url": image_item["image_url"]
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.debug_log(1, f"❌ 图像描述生成失败: {str(e)}")
            return f"图像描述生成失败: {str(e)[:100]}"

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    async def retrieve_similar_chunks(self, query_vector: List[float], chat_id: str) -> List[Tuple[str, float]]:
        """检索相似文本块"""
        if not query_vector or chat_id not in self._vector_store:
            return []
        
        similarities = []
        chunks_data = self._vector_store[chat_id]
        
        for chunk_text, chunk_vector in chunks_data:
            similarity = self.cosine_similarity(query_vector, chunk_vector)
            similarities.append((chunk_text, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.valves.retrieval_top_k]

    async def rerank_documents(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """语义重排序"""
        if not self.valves.enable_rerank or not self.valves.rerank_api_key:
            return [(doc, 1.0) for doc in documents]
        
        if not HTTPX_AVAILABLE:
            self.debug_log(1, "❌ httpx库未安装，跳过重排序")
            return [(doc, 1.0) for doc in documents]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.rerank_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.valves.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": min(self.valves.rerank_top_n, len(documents)),
                "return_documents": True
            }
            
            async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
                response = await client.post(
                    self.valves.rerank_api_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 200:
                        reranked = []
                        for item in result["data"]["results"]:
                            doc_text = item["document"]["text"]
                            score = item["relevance_score"]
                            reranked.append((doc_text, score))
                        return reranked
                    else:
                        self.debug_log(1, f"❌ 重排序API错误: {result.get('msg', 'Unknown error')}")
                else:
                    self.debug_log(1, f"❌ 重排序HTTP错误: {response.status_code}")
        
        except Exception as e:
            self.debug_log(1, f"❌ 重排序失败: {str(e)}")
        
        return [(doc, 1.0) for doc in documents]

    async def process_multimodal_content(self, messages: List[dict], model_name: str) -> List[dict]:
        """处理多模态内容"""
        if not self.valves.enable_multimodal_processing:
            return messages
        
        # 检查是否需要处理
        if self.has_native_vision(model_name):
            self.debug_log(2, f"👁️ 模型 {model_name} 原生支持视觉，跳过多模态处理")
            return messages
        
        # 提取图像
        images = self.extract_images_from_messages(messages)
        if not images:
            return messages
        
        self.debug_log(1, f"🖼️ 检测到 {len(images)} 张图像，开始处理...")
        
        # 处理每张图像
        processed_messages = []
        for i, message in enumerate(messages):
            processed_message = message.copy()
            content = message.get("content", [])
            
            if isinstance(content, list):
                new_content = []
                has_images = False
                
                for item in content:
                    if item.get("type") == "image_url":
                        has_images = True
                        # 生成图像描述
                        try:
                            description = await self.describe_image(item)
                            new_content.append({
                                "type": "text",
                                "text": f"[图像描述]: {description}"
                            })
                            self.debug_log(2, f"🎨 图像描述生成成功: {description[:100]}...")
                        except Exception as e:
                            self.debug_log(1, f"❌ 图像处理失败: {str(e)}")
                            new_content.append({
                                "type": "text", 
                                "text": f"[图像处理失败]: {str(e)[:100]}"
                            })
                    else:
                        new_content.append(item)
                
                if has_images:
                    processed_message["content"] = new_content
            
            processed_messages.append(processed_message)
        
        return processed_messages

    async def store_conversation_chunks(self, messages: List[dict], chat_id: str):
        """存储对话块到向量库"""
        if not self.valves.enable_vectorization:
            return
        
        chunks_data = []
        
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                # 分片
                chunks = self.smart_chunk_text(
                    content,
                    self.valves.chunk_size,
                    self.valves.chunk_overlap
                )
                
                # 向量化每个分片
                for chunk in chunks:
                    if chunk.strip():
                        try:
                            vector = await self.vectorize_text(chunk)
                            if vector:
                                chunks_data.append((chunk, vector))
                        except Exception as e:
                            self.debug_log(2, f"❌ 分片向量化失败: {str(e)}")
        
        # 存储到向量库
        if chunks_data:
            if chat_id not in self._vector_store:
                self._vector_store[chat_id] = []
            self._vector_store[chat_id].extend(chunks_data)
            
            # 限制存储大小（保留最新的1000个块）
            if len(self._vector_store[chat_id]) > 1000:
                self._vector_store[chat_id] = self._vector_store[chat_id][-1000:]
            
            self.debug_log(1, f"📦 存储了 {len(chunks_data)} 个文本块到向量库")

    async def smart_context_compression(self, messages: List[dict], token_limit: int, chat_id: str) -> List[dict]:
        """智能上下文压缩"""
        current_tokens = self.count_messages_tokens(messages)
        
        if current_tokens <= token_limit:
            self.debug_log(2, f"✅ 当前token数 {current_tokens} 未超限制 {token_limit}")
            return messages
        
        self.debug_log(1, f"⚠️ 需要压缩: {current_tokens} -> {token_limit} tokens")
        
        # 分离系统消息和对话消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # 保留最后几条消息
        preserve_count = self.valves.preserve_last_messages_count
        if len(conversation_messages) <= preserve_count:
            return messages
        
        preserved_messages = conversation_messages[-preserve_count:]
        to_compress_messages = conversation_messages[:-preserve_count]
        
        # 计算保留空间
        system_tokens = self.count_messages_tokens(system_messages)
        preserved_tokens = self.count_messages_tokens(preserved_messages)
        available_tokens = token_limit - system_tokens - preserved_tokens
        
        if available_tokens <= 0:
            self.debug_log(1, "⚠️ 保留消息已超限，仅返回必要内容")
            return system_messages + preserved_messages
        
        # 计算压缩策略
        preserve_ratio = self.valves.context_preserve_ratio
        preserve_tokens = int(available_tokens * preserve_ratio)
        retrieval_tokens = available_tokens - preserve_tokens
        
        # 1. 保留部分原始消息
        preserved_original = []
        current_preserve_tokens = 0
        
        for msg in reversed(to_compress_messages):
            msg_tokens = self.count_message_tokens(msg)
            if current_preserve_tokens + msg_tokens <= preserve_tokens:
                preserved_original.insert(0, msg)
                current_preserve_tokens += msg_tokens
            else:
                break
        
        # 2. 为剩余消息进行向量检索
        remaining_messages = to_compress_messages[:-len(preserved_original)] if preserved_original else to_compress_messages
        
        if remaining_messages and self.valves.enable_vectorization:
            # 存储到向量库
            await self.store_conversation_chunks(remaining_messages, chat_id)
            
            # 构建查询（使用最后一条用户消息）
            query_text = ""
            for msg in reversed(conversation_messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        query_text = content
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                query_text = item.get("text", "")
                                break
                    break
            
            if query_text:
                # 向量检索
                query_vector = await self.vectorize_text(query_text)
                if query_vector:
                    similar_chunks = await self.retrieve_similar_chunks(query_vector, chat_id)
                    
                    if similar_chunks:
                        # 重排序
                        documents = [chunk for chunk, _ in similar_chunks]
                        reranked = await self.rerank_documents(query_text, documents)
                        
                        # 构建检索内容
                        retrieval_content = ""
                        current_retrieval_tokens = 0
                        
                        for doc, score in reranked:
                            doc_tokens = self.count_tokens(doc)
                            if current_retrieval_tokens + doc_tokens <= retrieval_tokens:
                                retrieval_content += f"\n[相关内容 - 相似度: {score:.3f}]\n{doc}\n"
                                current_retrieval_tokens += doc_tokens
                            else:
                                break
                        
                        if retrieval_content:
                            # 添加检索摘要到系统消息
                            retrieval_summary = {
                                "role": "system",
                                "content": f"=== 📚 智能检索内容 ===\n基于当前查询检索到的相关历史对话内容：\n{retrieval_content}"
                            }
                            system_messages.append(retrieval_summary)
                            
                            self.debug_log(1, f"🔍 添加了 {current_retrieval_tokens} tokens 的检索内容")
        
        # 组装最终结果
        final_messages = system_messages + preserved_original + preserved_messages
        final_tokens = self.count_messages_tokens(final_messages)
        
        self.debug_log(1, f"✅ 压缩完成: {current_tokens} -> {final_tokens} tokens")
        return final_messages

    async def send_status(self, __event_emitter__, message: str, done: bool = True):
        """发送状态消息"""
        if __event_emitter__ and self.valves.show_processing_status:
            try:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": done,
                    },
                })
            except Exception as e:
                self.debug_log(2, f"❌ 状态发送失败: {str(e)}")

    def get_chat_id(self, __event_emitter__) -> str:
        """提取聊天ID"""
        try:
            if hasattr(__event_emitter__, "__closure__") and __event_emitter__.__closure__:
                info = __event_emitter__.__closure__[0].cell_contents
                chat_id = info.get("chat_id")
                if chat_id:
                    return chat_id
        except:
            pass
        
        # 生成fallback ID
        return f"chat_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """主要处理逻辑"""
        if not self.toggle:
            return body
        
        messages = body.get("messages", [])
        model = body.get("model", "")
        
        if not messages:
            return body
        
        chat_id = self.get_chat_id(__event_emitter__)
        start_time = datetime.now()
        
        try:
            # 获取模型token限制
            token_limit = self.get_model_token_limit(model)
            self.debug_log(1, f"🎯 模型 {model} token限制: {token_limit}")
            
            await self.send_status(__event_emitter__, f"🧠 启动智能上下文管理 (模型: {model})", False)
            
            # 1. 多模态处理
            if self.valves.enable_multimodal_processing:
                await self.send_status(__event_emitter__, "🖼️ 处理多模态内容...", False)
                messages = await self.process_multimodal_content(messages, model)
                body["messages"] = messages
            
            # 2. 上下文管理
            if self.valves.enable_context_management:
                await self.send_status(__event_emitter__, "📊 智能上下文压缩...", False)
                messages = await self.smart_context_compression(messages, token_limit, chat_id)
                body["messages"] = messages
            
            # 处理统计
            processing_time = (datetime.now() - start_time).total_seconds()
            final_tokens = self.count_messages_tokens(messages)
            
            self._processing_stats[chat_id] = {
                "last_processing_time": processing_time,
                "final_tokens": final_tokens,
                "timestamp": datetime.now()
            }
            
            await self.send_status(
                __event_emitter__, 
                f"✅ 处理完成 | 🎯 {final_tokens} tokens | ⏱️ {processing_time:.2f}s"
            )
            
            self.debug_log(1, f"✅ 处理完成: {len(messages)} 条消息, {final_tokens} tokens, 耗时 {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            await self.send_status(__event_emitter__, error_msg)
            self.debug_log(1, error_msg)
            
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
        if not self.toggle:
            return body
        
        # 可以在这里添加输出处理逻辑，比如添加溯源标记等
        return body
