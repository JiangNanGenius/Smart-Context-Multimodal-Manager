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
