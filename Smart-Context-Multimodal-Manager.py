"""
title: 🚀 Advanced Multimodal Context Manager
author: JiangNanGenius
version: 1.5.8
license: MIT
required_open_webui_version: 0.5.17
description: 智能长上下文和多模态内容处理器，支持向量化检索、语义重排序、智能分片等功能 - 修复新消息判断和前端显示
"""

import json
import hashlib
import asyncio
import re
import base64
import math
import time
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

class ProcessingStrategy(str, Enum):
    ITERATIVE = "iterative"  # 迭代处理（推荐）
    CHUNK_FIRST = "chunk_first"  # 分片优先
    SUMMARY_FIRST = "summary_first"  # 摘要优先
    MIXED = "mixed"  # 混合策略

class ProcessingStats:
    """处理统计信息"""
    def __init__(self):
        self.original_tokens = 0
        self.original_messages = 0
        self.final_tokens = 0
        self.final_messages = 0
        self.token_limit = 0
        self.iterations = 0
        self.chunked_messages = 0
        self.summarized_messages = 0
        self.vector_retrievals = 0
        self.rerank_operations = 0
        self.multimodal_processed = 0
        self.processing_time = 0.0
        self.current_user_message_preserved = False
        
    def calculate_retention_ratio(self) -> float:
        """计算内容保留比例"""
        if self.original_tokens == 0:
            return 0.0
        return self.final_tokens / self.original_tokens
    
    def calculate_window_usage_ratio(self) -> float:
        """计算对话窗口使用率"""
        if self.token_limit == 0:
            return 0.0
        return self.final_tokens / self.token_limit
    
    def calculate_compression_ratio(self) -> float:
        """计算压缩比例"""
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.final_tokens) / self.original_tokens
    
    def calculate_processing_efficiency(self) -> float:
        """计算处理效率 (保留的有用信息 / 处理时间)"""
        if self.processing_time == 0:
            return 0.0
        return self.final_tokens / self.processing_time
    
    def get_summary(self) -> str:
        """获取统计摘要"""
        retention = self.calculate_retention_ratio()
        window_usage = self.calculate_window_usage_ratio()
        compression = self.calculate_compression_ratio()
        efficiency = self.calculate_processing_efficiency()
        
        return f"""
📊 处理统计报告:
├─ 输入: {self.original_messages}条消息, {self.original_tokens:,}tokens
├─ 输出: {self.final_messages}条消息, {self.final_tokens:,}tokens
├─ 模型限制: {self.token_limit:,}tokens
├─ 📈 内容保留率: {retention:.2%}
├─ 🪟 窗口使用率: {window_usage:.2%}
├─ 📉 压缩比例: {compression:.2%}
├─ ⚡ 处理效率: {efficiency:.0f}tokens/s
├─ 🔄 迭代次数: {self.iterations}
├─ 🧩 分片消息: {self.chunked_messages}条
├─ 📝 摘要消息: {self.summarized_messages}条
├─ 🔍 向量检索: {self.vector_retrievals}次
├─ 🔄 重排序: {self.rerank_operations}次
├─ 🖼️ 多模态处理: {self.multimodal_processed}张图片
├─ 💬 当前用户消息: {'已保留' if self.current_user_message_preserved else '未保留'}
└─ ⏱️ 处理时间: {self.processing_time:.2f}秒"""

class ProgressTracker:
    """进度追踪器"""
    def __init__(self, __event_emitter__):
        self.event_emitter = __event_emitter__
        self.current_step = 0
        self.total_steps = 0
        self.current_phase = ""
        self.phase_progress = 0
        self.phase_total = 0
        
    async def start_phase(self, phase_name: str, total_items: int = 0):
        """开始新阶段"""
        self.current_phase = phase_name
        self.phase_progress = 0
        self.phase_total = total_items
        await self.update_status(f"开始 {phase_name}")
        
    async def update_progress(self, completed: int, total: int = None, detail: str = ""):
        """更新进度"""
        if total is None:
            total = self.phase_total
        
        self.phase_progress = completed
        if total > 0:
            percentage = (completed / total) * 100
            progress_bar = "█" * int(percentage // 10) + "░" * (10 - int(percentage // 10))
            status = f"{self.current_phase} [{progress_bar}] {percentage:.1f}% ({completed}/{total})"
            if detail:
                status += f" - {detail}"
        else:
            status = f"{self.current_phase} - {detail}" if detail else self.current_phase
            
        await self.update_status(status, False)
        
    async def complete_phase(self, message: str = ""):
        """完成当前阶段"""
        final_message = f"{self.current_phase} 完成"
        if message:
            final_message += f" - {message}"
        await self.update_status(final_message, True)
        
    async def update_status(self, message: str, done: bool = False):
        """更新状态"""
        if self.event_emitter:
            try:
                await self.event_emitter({
                    "type": "status",
                    "data": {"description": f"🔄 {message}", "done": done},
                })
            except:
                pass

class Filter:
    class Valves(BaseModel):
        # 基础控制
        enable_processing: bool = Field(default=True, description="🔄 启用所有处理功能")
        excluded_models: str = Field(
            default="", description="🚫 排除模型列表(逗号分隔)"
        )
        
        # 核心处理策略
        processing_strategy: str = Field(
            default="iterative", 
            description="🎯 处理策略 (iterative|chunk_first|summary_first|mixed) - 推荐iterative"
        )
        
        # 多模态模型配置
        multimodal_models: str = Field(
            default="gpt-4o,gpt-4o-mini,gpt-4-vision-preview,doubao-1.5-vision-pro,doubao-1.5-vision-lite,claude-3,gemini-pro-vision,qwen-vl",
            description="🖼️ 多模态模型列表(逗号分隔)",
        )
        
        # 模型Token限制配置
        model_token_limits: str = Field(
            default="gpt-4o:128000,gpt-4o-mini:128000,gpt-4:8192,gpt-3.5-turbo:16385,doubao-1.5-thinking-pro:128000,doubao-1.5-vision-pro:128000,doubao-seed:50000,doubao:50000,claude-3:200000,gemini-pro:128000,doubao-1-5-pro-256k:200000,doubao-seed-1-6-250615:50000",
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
        enable_intelligent_chunking: bool = Field(
            default=True, description="🧩 启用智能分片"
        )
        
        # 统计和调试
        enable_detailed_stats: bool = Field(
            default=True, description="📊 启用详细统计"
        )
        enable_detailed_progress: bool = Field(
            default=True, description="📱 启用详细进度显示"
        )
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
        
        # Token管理 - 动态调整策略
        default_token_limit: int = Field(default=100000, description="⚖️ 默认token限制")
        token_safety_ratio: float = Field(
            default=0.88, description="🛡️ Token安全比例"
        )
        target_window_usage: float = Field(
            default=0.85, description="🪟 目标窗口使用率(85%)"
        )
        min_window_usage: float = Field(
            default=0.70, description="🪟 最小窗口使用率(70%)"
        )
        max_processing_iterations: int = Field(
            default=8, description="🔄 最大处理迭代次数"
        )
        min_reduction_threshold: int = Field(
            default=2000, description="📉 最小减少阈值"
        )
        
        # 动态内容丢失比例 - 根据压缩率和窗口使用率调整
        base_content_loss_ratio: float = Field(
            default=0.20, description="📉 基础内容丢失比例(20%)"
        )
        high_compression_loss_ratio: float = Field(
            default=0.08, description="📉 高压缩场景内容丢失比例(8%)"
        )
        compression_threshold: float = Field(
            default=0.15, description="📉 高压缩场景阈值(目标/原始<15%时认为是高压缩)"
        )
        
        # 保护策略 - 修复当前消息判断
        force_preserve_current_user_message: bool = Field(
            default=True, description="🔒 强制保留当前用户消息(最新的用户输入)"
        )
        preserve_recent_exchanges: int = Field(
            default=4, description="💬 保护最近完整对话轮次"
        )
        max_preserve_ratio: float = Field(
            default=0.3, description="🔒 保护消息最大token比例"
        )
        max_single_message_tokens: int = Field(
            default=20000, description="📝 单条消息最大token"
        )
        
        # 智能分片配置
        chunk_target_tokens: int = Field(
            default=4000, description="🧩 分片目标token数"
        )
        chunk_overlap_tokens: int = Field(
            default=400, description="🔗 分片重叠token数"
        )
        chunk_min_tokens: int = Field(
            default=1000, description="📏 分片最小token数"
        )
        chunk_max_tokens: int = Field(
            default=8000, description="📏 分片最大token数"
        )
        preserve_paragraph_integrity: bool = Field(
            default=True, description="📝 保持段落完整性"
        )
        preserve_sentence_integrity: bool = Field(
            default=True, description="📝 保持句子完整性"
        )
        preserve_code_blocks: bool = Field(
            default=True, description="💻 保持代码块完整性"
        )
        
        # 迭代处理配置
        chunk_selection_ratio: float = Field(
            default=0.8, description="🧩 分片选择比例(80%)"
        )
        enable_chunk_vector_retrieval: bool = Field(
            default=True, description="🔍 启用分片向量检索"
        )
        enable_chunk_reranking: bool = Field(
            default=True, description="🔄 启用分片重排序"
        )
        enable_progressive_summarization: bool = Field(
            default=True, description="📝 启用渐进式摘要"
        )
        
        # 内容优先级设置
        high_priority_content: str = Field(
            default="代码,配置,参数,数据,错误,解决方案,步骤,方法,技术细节,API,函数,类,变量,问题,bug,修复,实现,算法,架构",
            description="🎯 高优先级内容关键词(逗号分隔)"
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
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。特别注意代码、配置、数据等技术信息。保持客观准确，重点突出关键信息。",
            description="👁️ Vision提示词",
        )
        vision_max_tokens: int = Field(
            default=2000, description="👁️ Vision最大输出tokens"
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
            default=0.25, description="🎯 基础相似度阈值"
        )
        multimodal_similarity_threshold: float = Field(
            default=0.2, description="🖼️ 多模态相似度阈值"
        )
        text_similarity_threshold: float = Field(
            default=0.3, description="📝 文本相似度阈值"
        )
        vector_top_k: int = Field(default=50, description="🔝 向量检索Top-K数量")
        
        # 重排序
        enable_reranking: bool = Field(default=True, description="🔄 启用重排序")
        rerank_api_base: str = Field(
            default="https://api.bochaai.com", description="🔄 重排序API"
        )
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=40, description="🔝 重排序返回数量")
        
        # 摘要配置 - 动态计算最小长度
        summary_api_base: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3", description="📝 摘要API"
        )
        summary_api_key: str = Field(default="", description="🔑 摘要密钥")
        summary_model: str = Field(
            default="doubao-1.5-thinking-pro-250415", description="🧠 摘要模型"
        )
        max_summary_length: int = Field(
            default=10000, description="📏 摘要最大长度"
        )
        min_summary_ratio: float = Field(
            default=0.3, description="📏 摘要最小长度比例(相对于chunk大小的30%)"
        )
        summary_compression_ratio: float = Field(
            default=0.5, description="📊 摘要压缩比例"
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
        print("🚀 Advanced Multimodal Context Manager v1.5.8")
        print("📍 插件正在初始化...")
        print("🔧 修复新消息判断逻辑，优化前端显示...")
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
        
        # 处理统计
        self.stats = ProcessingStats()
        
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
        
        # 解析高优先级内容关键词
        self.high_priority_keywords = set()
        if self.valves.high_priority_content:
            self.high_priority_keywords = {
                keyword.strip().lower()
                for keyword in self.valves.high_priority_content.split(",")
                if keyword.strip()
            }
        
        print(f"✅ 插件初始化完成")
        print(f"🎯 处理策略: {self.valves.processing_strategy}")
        print(f"📊 详细统计: {self.valves.enable_detailed_stats}")
        print(f"📱 详细进度: {self.valves.enable_detailed_progress}")
        print(f"🔒 保留当前用户消息: {self.valves.force_preserve_current_user_message}")
        print(f"🪟 目标窗口使用率: {self.valves.target_window_usage:.1%}")
        print("=" * 60 + "\n")

    def calculate_min_summary_length(self) -> int:
        """动态计算最小摘要长度"""
        return max(500, int(self.valves.chunk_target_tokens * self.valves.min_summary_ratio))

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

    def find_current_user_message(self, messages: List[dict]) -> Optional[dict]:
        """查找当前用户消息（最新的用户输入）"""
        if not messages:
            return None
        
        # 从最后一条消息开始查找，找到最新的用户消息
        # 这应该是用户刚刚发送的消息
        for msg in reversed(messages):
            if msg.get("role") == "user":
                self.debug_log(2, f"找到当前用户消息: {len(self.extract_text_from_content(msg.get('content', '')))}字符", "💬")
                return msg
        
        return None

    def separate_current_and_history_messages(self, messages: List[dict]) -> Tuple[Optional[dict], List[dict]]:
        """分离当前用户消息和历史消息"""
        if not messages:
            return None, []
        
        # 找到当前用户消息（最新的用户输入）
        current_user_message = self.find_current_user_message(messages)
        
        if not current_user_message:
            return None, messages
        
        # 分离历史消息（除了当前用户消息之外的所有消息）
        history_messages = []
        current_found = False
        
        for msg in reversed(messages):
            if msg is current_user_message and not current_found:
                # 跳过当前用户消息
                current_found = True
                continue
            else:
                history_messages.insert(0, msg)
        
        self.debug_log(
            1, 
            f"消息分离: 当前用户消息1条({self.count_message_tokens(current_user_message)}tokens), 历史消息{len(history_messages)}条", 
            "📋"
        )
        
        return current_user_message, history_messages

    def calculate_dynamic_loss_ratio(self, original_tokens: int, target_tokens: int) -> float:
        """动态计算内容丢失比例阈值"""
        if original_tokens <= 0:
            return self.valves.base_content_loss_ratio
        
        # 计算压缩率
        compression_ratio = target_tokens / original_tokens
        
        # 如果是高压缩场景
        if compression_ratio < self.valves.compression_threshold:
            max_loss_ratio = self.valves.high_compression_loss_ratio
            self.debug_log(
                2, 
                f"📉 高压缩场景(压缩率{compression_ratio:.2%})，允许最大丢失{max_loss_ratio:.1%}", 
                "📉"
            )
        else:
            max_loss_ratio = self.valves.base_content_loss_ratio
            self.debug_log(
                2, 
                f"📉 正常压缩场景(压缩率{compression_ratio:.2%})，允许最大丢失{max_loss_ratio:.1%}", 
                "📉"
            )
        
        return max_loss_ratio

    def should_continue_processing(self, current_tokens: int, target_tokens: int, original_tokens: int) -> bool:
        """判断是否应该继续处理"""
        # 1. 首要条件：是否达到目标
        if current_tokens <= target_tokens:
            return False
        
        # 2. 计算当前窗口使用率
        current_usage = current_tokens / target_tokens if target_tokens > 0 else 0
        
        # 3. 如果使用率过高，必须继续处理
        if current_usage > 1.0:
            return True
        
        # 4. 检查内容保留率
        retention_ratio = current_tokens / original_tokens if original_tokens > 0 else 0
        max_loss_ratio = self.calculate_dynamic_loss_ratio(original_tokens, target_tokens)
        
        # 5. 如果内容丢失过多，谨慎处理
        if retention_ratio < (1 - max_loss_ratio):
            self.debug_log(
                2,
                f"📊 内容保留率({retention_ratio:.2%}) < 阈值({1-max_loss_ratio:.2%})，谨慎处理",
                "⚠️"
            )
            # 但如果窗口使用率仍然过高，还是要继续
            if current_usage > 1.1:  # 超过110%必须处理
                return True
            else:
                return False
        
        return True

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

    # ========== 智能分片功能 ==========
    def find_code_blocks(self, text: str) -> List[Tuple[int, int]]:
        """查找代码块边界"""
        code_blocks = []
        # 查找```代码块
        pattern = r'```[\s\S]*?```'
        for match in re.finditer(pattern, text):
            code_blocks.append((match.start(), match.end()))
        
        # 查找`代码`
        pattern = r'`[^`\n]+`'
        for match in re.finditer(pattern, text):
            code_blocks.append((match.start(), match.end()))
        
        return code_blocks

    def find_sentence_boundaries(self, text: str) -> List[int]:
        """查找句子边界"""
        boundaries = []
        # 中文句子结束标点
        chinese_endings = ['。', '！', '？', '；', '…']
        # 英文句子结束标点
        english_endings = ['.', '!', '?', ';']
        
        for i, char in enumerate(text):
            if char in chinese_endings or char in english_endings:
                # 检查是否是真正的句子结束（避免小数点、缩写等）
                if i + 1 < len(text):
                    next_char = text[i + 1]
                    if next_char in [' ', '\n', '\t'] or next_char.isupper():
                        boundaries.append(i + 1)
                else:
                    boundaries.append(i + 1)
        return boundaries

    def find_paragraph_boundaries(self, text: str) -> List[int]:
        """查找段落边界"""
        boundaries = []
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            current_pos += len(line) + 1  # +1 for '\n'
            if line.strip() == '':  # 空行表示段落结束
                boundaries.append(current_pos)
        
        return boundaries

    def is_high_priority_content(self, text: str) -> bool:
        """判断是否为高优先级内容"""
        if not text or not self.high_priority_keywords:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.high_priority_keywords)

    def intelligent_chunk_text_v3(self, text: str, target_tokens: int) -> List[str]:
        """智能分片文本 v3 - 不丢弃内容"""
        if not text:
            return []
        
        current_tokens = self.count_tokens(text)
        if current_tokens <= target_tokens:
            return [text]
        
        self.debug_log(2, f"🧩 开始智能分片: {current_tokens}tokens -> 目标{target_tokens}tokens", "🧩")
        
        chunks = []
        
        # 1. 保护代码块
        if self.valves.preserve_code_blocks:
            code_blocks = self.find_code_blocks(text)
            if code_blocks:
                # 按代码块分割
                last_end = 0
                for start, end in code_blocks:
                    # 添加代码块前的内容
                    if start > last_end:
                        before_text = text[last_end:start].strip()
                        if before_text:
                            chunks.extend(self.chunk_text_by_paragraphs(before_text, target_tokens))
                    
                    # 添加代码块（保持完整）
                    code_text = text[start:end]
                    code_tokens = self.count_tokens(code_text)
                    if code_tokens <= self.valves.chunk_max_tokens:
                        chunks.append(code_text)
                    else:
                        # 代码块太大，谨慎分割
                        chunks.extend(self.chunk_large_code_block(code_text, target_tokens))
                    
                    last_end = end
                
                # 添加最后的内容
                if last_end < len(text):
                    after_text = text[last_end:].strip()
                    if after_text:
                        chunks.extend(self.chunk_text_by_paragraphs(after_text, target_tokens))
                
                self.debug_log(2, f"🧩 代码块分片完成: {len(chunks)}片", "🧩")
                return chunks
        
        # 2. 按段落分片
        if self.valves.preserve_paragraph_integrity:
            chunks = self.chunk_text_by_paragraphs(text, target_tokens)
        # 3. 按句子分片
        elif self.valves.preserve_sentence_integrity:
            chunks = self.chunk_text_by_sentences(text, target_tokens)
        # 4. 简单分片
        else:
            chunks = self.simple_chunk_text(text, target_tokens)
        
        self.debug_log(2, f"🧩 智能分片完成: {current_tokens}tokens -> {len(chunks)}片", "🧩")
        return chunks

    def chunk_text_by_paragraphs(self, text: str, target_tokens: int) -> List[str]:
        """按段落分片"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_tokens = self.count_tokens(paragraph)
            current_chunk_tokens = self.count_tokens(current_chunk)
            
            # 单个段落就超过目标
            if para_tokens > target_tokens:
                # 保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 分割长段落
                if para_tokens > self.valves.chunk_max_tokens:
                    chunks.extend(self.chunk_long_paragraph(paragraph, target_tokens))
                else:
                    chunks.append(paragraph)
            
            # 加入这个段落会超过目标
            elif current_chunk_tokens + para_tokens > target_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 保存最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def chunk_text_by_sentences(self, text: str, target_tokens: int) -> List[str]:
        """按句子分片"""
        sentences = []
        sentence_boundaries = self.find_sentence_boundaries(text)
        
        start = 0
        for boundary in sentence_boundaries:
            sentence = text[start:boundary].strip()
            if sentence:
                sentences.append(sentence)
            start = boundary
        
        # 剩余部分
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                sentences.append(remaining)
        
        # 组合句子成chunk
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            current_chunk_tokens = self.count_tokens(current_chunk)
            
            if sentence_tokens > target_tokens:
                # 单个句子太长，强制分割
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(self.simple_chunk_text(sentence, target_tokens))
            elif current_chunk_tokens + sentence_tokens > target_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def chunk_long_paragraph(self, paragraph: str, target_tokens: int) -> List[str]:
        """分割长段落"""
        if self.valves.preserve_sentence_integrity:
            return self.chunk_text_by_sentences(paragraph, target_tokens)
        else:
            return self.simple_chunk_text(paragraph, target_tokens)

    def chunk_large_code_block(self, code_text: str, target_tokens: int) -> List[str]:
        """分割大代码块"""
        lines = code_text.split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            test_chunk = current_chunk + "\n" + line if current_chunk else line
            if self.count_tokens(test_chunk) > target_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def simple_chunk_text(self, text: str, target_tokens: int) -> List[str]:
        """简单按长度分割文本"""
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            if self.count_tokens(test_chunk) > target_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def chunk_message_content_v3(self, message: dict, target_tokens: int) -> List[dict]:
        """增强的消息内容分片 v3 - 不丢弃内容"""
        content = self.extract_text_from_content(message.get("content", ""))
        if not content:
            return [message]
        
        current_tokens = self.count_tokens(content)
        if current_tokens <= target_tokens:
            return [message]
        
        # 判断是否为高优先级内容
        is_high_priority = self.is_high_priority_content(content)
        
        # 使用智能分片
        chunks = self.intelligent_chunk_text_v3(content, target_tokens)
        if not chunks:
            return [message]
        
        # 创建分片消息
        chunked_messages = []
        for i, chunk in enumerate(chunks):
            chunked_message = message.copy()
            
            # 添加分片标识和优先级标识
            priority_mark = "🎯[高优先级]" if is_high_priority else ""
            chunked_message["content"] = f"{priority_mark}[分片{i+1}/{len(chunks)}] {chunk}"
            chunked_messages.append(chunked_message)
        
        self.debug_log(2, f"🧩 消息分片: {current_tokens}tokens -> {len(chunks)}片 {'(高优先级)' if is_high_priority else ''}", "🧩")
        
        # 更新统计
        self.stats.chunked_messages += 1
        
        return chunked_messages

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
            if time.time() - cache_time < 300:  # 5分钟缓存
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
                    self.api_error_cache[error_key] = (time.time(), error_msg)
                    return None
                
                if attempt < self.valves.api_error_retry_times:
                    self.debug_log(
                        1, f"{call_name} 第{attempt+1}次尝试失败，{self.valves.api_error_retry_delay}秒后重试: {error_msg}", "🔄"
                    )
                    await asyncio.sleep(self.valves.api_error_retry_delay)
                else:
                    self.debug_log(1, f"{call_name} 最终失败: {error_msg}", "❌")
                    # 记录到错误缓存
                    self.api_error_cache[error_key] = (time.time(), error_msg)
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
        self, query_message: dict, candidate_messages: List[dict], progress: ProgressTracker
    ) -> List[dict]:
        """基于向量相似度检索相关消息"""
        if not candidate_messages or not self.valves.enable_vector_retrieval:
            return candidate_messages
        
        await progress.start_phase("向量检索", len(candidate_messages))
        
        self.debug_log(
            1, f"开始向量检索: 查询1条，候选{len(candidate_messages)}条", "🔍"
        )
        
        # 更新统计
        self.stats.vector_retrievals += 1
        
        # 获取查询向量
        query_content = query_message.get("content", "")
        query_vector = None
        strategy = self.valves.vector_strategy.lower()
        
        await progress.update_progress(0, len(candidate_messages), "获取查询向量")
        
        # 根据策略选择向量化方法
        if self.has_images_in_content(query_content):
            if strategy in ["auto", "multimodal_first"]:
                query_vector = await self.get_multimodal_embedding(
                    query_content, progress.event_emitter
                )
            if not query_vector and strategy in ["auto", "fallback"]:
                # 转换为文本再向量化
                text_content = self.extract_text_from_content(query_content)
                if text_content:
                    query_vector = await self.get_text_embedding(
                        text_content, progress.event_emitter
                    )
        else:
            text_content = self.extract_text_from_content(query_content)
            if text_content:
                query_vector = await self.get_text_embedding(
                    text_content, progress.event_emitter
                )
        
        if not query_vector:
            self.debug_log(1, "查询向量获取失败，返回原始消息", "⚠️")
            await progress.complete_phase("查询向量获取失败")
            return candidate_messages
        
        # 计算候选消息的相似度
        similarities = []
        for i, msg in enumerate(candidate_messages):
            await progress.update_progress(i + 1, len(candidate_messages), f"计算相似度 {i+1}/{len(candidate_messages)}")
            
            msg_content = msg.get("content", "")
            msg_vector = None
            
            # 为候选消息获取向量
            if self.has_images_in_content(msg_content):
                msg_vector = await self.get_multimodal_embedding(
                    msg_content, progress.event_emitter
                )
                if not msg_vector:
                    text_content = self.extract_text_from_content(msg_content)
                    if text_content:
                        msg_vector = await self.get_text_embedding(
                            text_content, progress.event_emitter
                        )
            else:
                text_content = self.extract_text_from_content(msg_content)
                if text_content:
                    msg_vector = await self.get_text_embedding(
                        text_content, progress.event_emitter
                    )
            
            if msg_vector:
                similarity = self.cosine_similarity(query_vector, msg_vector)
                
                # 高优先级内容给予加权
                if self.is_high_priority_content(self.extract_text_from_content(msg_content)):
                    similarity = min(1.0, similarity * 1.3)  # 提高权重
                
                similarities.append((i, similarity, msg))
                self.debug_log(3, f"消息{i}相似度: {similarity:.3f}", "📊")
            else:
                # 没有向量的消息给基础分数
                similarities.append((i, 0.3, msg))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 使用更宽松的阈值过滤
        threshold = self.valves.vector_similarity_threshold
        filtered_similarities = [item for item in similarities if item[1] >= threshold]
        
        # 如果过滤后太少，降低阈值
        if len(filtered_similarities) < len(candidate_messages) * 0.3:
            lower_threshold = max(0.1, threshold - 0.1)
            filtered_similarities = [item for item in similarities if item[1] >= lower_threshold]
            self.debug_log(2, f"降低阈值到{lower_threshold:.2f}，保留更多消息", "🔍")
        
        # 限制数量但保留更多
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
        
        await progress.complete_phase(f"检索到{len(relevant_messages)}条相关消息")
        
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
        self, query_message: dict, candidate_messages: List[dict], progress: ProgressTracker
    ) -> List[dict]:
        """重排序消息"""
        if not candidate_messages or not self.valves.enable_reranking:
            return candidate_messages
        
        await progress.start_phase("重排序", len(candidate_messages))
        
        self.debug_log(1, f"开始重排序: 查询1条，候选{len(candidate_messages)}条", "🔄")
        
        # 更新统计
        self.stats.rerank_operations += 1
        
        # 准备查询文本
        query_text = self.extract_text_from_content(query_message.get("content", ""))
        if not query_text:
            await progress.complete_phase("查询文本为空")
            return candidate_messages
        
        await progress.update_progress(0, 1, "准备文档列表")
        
        # 准备文档列表
        documents = []
        for msg in candidate_messages:
            text = self.extract_text_from_content(msg.get("content", ""))
            if text:
                # 提高文档长度限制
                if len(text) > 5000:
                    text = text[:5000] + "..."
                documents.append(text)
            else:
                documents.append("空消息")
        
        if not documents:
            await progress.complete_phase("无有效文档")
            return candidate_messages
        
        await progress.update_progress(1, 1, "调用重排序API")
        
        # 调用重排序API
        rerank_results = await self.safe_api_call(
            self._rerank_messages_impl, "重排序", query_text, documents, progress.event_emitter
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
            
            await progress.complete_phase(f"重排序到{len(reranked_messages)}条消息")
            
            return reranked_messages
        
        await progress.complete_phase("重排序失败")
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
            if len(description) > 2000:
                description = description[:2000] + "..."
            
            self.vision_cache[image_hash] = description
            self.debug_log(2, f"图片识别完成: {len(description)}字符", "✅")
            return description
        
        return "图片处理失败：无法获取描述"

    async def process_message_images(self, message: dict, progress: ProgressTracker) -> dict:
        """处理单条消息中的图片"""
        content = message.get("content", "")
        if not isinstance(content, list):
            return message
        
        # 检查是否包含图片
        images = [item for item in content if item.get("type") == "image_url"]
        if not images:
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
                    if progress:
                        await progress.update_progress(image_count, len(images), f"处理图片 {image_count}/{len(images)}")
                    
                    description = await self.describe_image(
                        image_url, progress.event_emitter if progress else None
                    )
                    processed_content.append(f"[图片{image_count}描述] {description}")
        
        # 创建新消息
        processed_message = message.copy()
        processed_message["content"] = (
            "\n".join(processed_content) if processed_content else ""
        )
        
        self.debug_log(2, f"消息图片处理完成: {image_count}张图片", "🖼️")
        
        # 更新统计
        self.stats.multimodal_processed += image_count
        
        return processed_message

    async def process_multimodal_content(
        self, messages: List[dict], model_name: str, progress: ProgressTracker
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
        
        await progress.start_phase("多模态处理", total_images)
        
        self.debug_log(1, f"开始处理多模态内容：{total_images} 张图片", "🖼️")
        
        # 处理所有消息
        processed_messages = []
        processed_count = 0
        
        for i, message in enumerate(messages):
            if self.has_images_in_content(message.get("content")):
                image_count = len([item for item in message.get("content", []) if item.get("type") == "image_url"])
                
                # 创建子进度追踪
                class SubProgress:
                    def __init__(self, parent, base_count):
                        self.parent = parent
                        self.base_count = base_count
                        self.event_emitter = parent.event_emitter
                    
                    async def update_progress(self, current, total, detail):
                        await self.parent.update_progress(
                            self.base_count + current, 
                            self.parent.phase_total, 
                            detail
                        )
                
                sub_progress = SubProgress(progress, processed_count)
                
                processed_message = await self.process_message_images(
                    message, sub_progress
                )
                processed_messages.append(processed_message)
                processed_count += image_count
            else:
                processed_messages.append(message)
        
        self.debug_log(1, f"多模态处理完成：{processed_count} 张图片", "✅")
        
        await progress.complete_phase(f"处理完成 {processed_count} 张图片")
        
        return processed_messages

    # ========== 摘要功能 ==========
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

    async def _summarize_messages_impl(self, conversation_text: str, original_tokens: int, iteration: int):
        """实际的摘要实现"""
        client = self.get_summary_client()
        if not client:
            return None
        
        # 动态计算期望的摘要长度
        min_summary_length = self.calculate_min_summary_length()
        target_summary_tokens = max(
            min_summary_length,
            int(original_tokens * self.valves.summary_compression_ratio)
        )
        
        # 摘要提示
        system_prompt = f"""你是专业的对话摘要助手。请为以下对话创建**详细完整**的结构化摘要。

⚠️ 重要要求：
1. 摘要必须达到 {target_summary_tokens} tokens以上（最小{min_summary_length}tokens）
2. 保持对话的完整逻辑脉络和时间顺序
3. 完整保留所有关键信息：技术细节、参数配置、数据、代码片段、错误信息、解决方案、操作步骤
4. 使用清晰的结构化格式
5. 保留用户的具体问题和助手的详细回答
6. 如果内容很重要，必须保留，可以超出长度限制

原始token数：{original_tokens}
期望摘要token数：{target_summary_tokens}+
最小摘要token数：{min_summary_length}
迭代次数：{iteration+1}

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
            summary = response.choices[0].message.content.strip()
            summary_tokens = self.count_tokens(summary)
            
            self.debug_log(2, f"摘要生成: {summary_tokens}tokens (期望{target_summary_tokens}+, 最小{min_summary_length})", "📝")
            
            # 动态摘要质量判断
            if summary_tokens >= min_summary_length:
                return summary
            else:
                self.debug_log(1, f"摘要过短({summary_tokens}tokens < {min_summary_length})，但仍然使用", "📝")
                return summary  # 即使短也使用，而不是丢弃
        
        return None

    async def summarize_messages_batch(
        self, messages: List[dict], progress: ProgressTracker, iteration: int = 0
    ) -> str:
        """批量摘要消息"""
        if not messages:
            return ""
        
        await progress.start_phase("摘要生成", len(messages))
        
        # 计算原始token数
        original_tokens = self.count_messages_tokens(messages)
        
        await progress.update_progress(0, 3, "预处理消息")
        
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
                            msg, None  # 不需要详细进度
                        )
                        processed_messages.append(processed_msg)
                    else:
                        processed_messages.append(msg)
        
        await progress.update_progress(1, 3, "格式化对话")
        
        # 按角色分组处理
        conversation_parts = []
        current_exchange = []
        
        for msg in processed_messages:
            role = msg.get("role", "unknown")
            content = self.extract_text_from_content(msg.get("content", ""))
            
            # 增加内容长度限制
            if len(content) > 12000:
                content = content[:12000] + "...(长内容已截断)"
            
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
        
        await progress.update_progress(2, 3, "调用摘要API")
        
        # 调用摘要API
        summary = await self.safe_api_call(
            self._summarize_messages_impl, "摘要生成", conversation_text, original_tokens, iteration
        )
        
        if summary:
            self.debug_log(1, f"📝 摘要成功: {len(summary)}字符", "📝")
            
            # 更新统计
            self.stats.summarized_messages += 1
            
            await progress.complete_phase(f"摘要生成成功 {len(summary)}字符")
            return summary
        
        self.debug_log(1, f"📝 摘要失败", "⚠️")
        await progress.complete_phase("摘要生成失败")
        return ""

    def format_exchange(self, exchange: List[str]) -> str:
        """格式化对话轮次"""
        return "\n".join(exchange)

    # ========== 核心处理策略 - 迭代处理 ==========
    def smart_message_selection_v7(
        self, messages: List[dict], current_user_message: Optional[dict], target_tokens: int, iteration: int = 0
    ) -> Tuple[List[dict], List[dict]]:
        """
        智能消息选择策略 v7 - 修复当前消息判断
        """
        if not messages:
            return [], []
        
        # 分离不同类型的消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        protected = []
        current_tokens = 0
        
        # 1. 强制保留当前用户消息（最新的用户输入）
        if self.valves.force_preserve_current_user_message and current_user_message:
            protected.append(current_user_message)
            current_tokens += self.count_message_tokens(current_user_message)
            self.debug_log(2, f"🔒 强制保留当前用户消息: {current_tokens}tokens", "💬")
            
            # 更新统计
            self.stats.current_user_message_preserved = True
        
        # 2. 保留系统消息
        for msg in system_messages:
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                protected.append(msg)
                current_tokens += msg_tokens
        
        # 3. 根据迭代次数调整保护策略
        preserve_exchanges = max(2, self.valves.preserve_recent_exchanges - iteration)
        max_preserve_ratio = max(0.2, self.valves.max_preserve_ratio - iteration * 0.05)
        max_preserve_tokens = int(target_tokens * max_preserve_ratio)
        
        self.debug_log(
            2,
            f"🔄 第{iteration+1}次迭代: 保护{preserve_exchanges}轮对话, 最大{max_preserve_tokens}tokens",
            "📊",
        )
        
        # 4. 保护最近的对话轮次（排除当前用户消息）
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
                            protected.insert(-1, prev_msg)
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

    async def iterative_content_processing_v4(
        self, messages: List[dict], target_tokens: int, progress: ProgressTracker
    ) -> List[dict]:
        """迭代内容处理的核心逻辑 v4 - 修复消息判断和进度显示"""
        start_time = time.time()
        
        # 分离当前用户消息和历史消息
        current_user_message, history_messages = self.separate_current_and_history_messages(messages)
        
        # 初始化统计
        self.stats.original_tokens = self.count_messages_tokens(messages)
        self.stats.original_messages = len(messages)
        self.stats.token_limit = target_tokens
        
        current_tokens = self.stats.original_tokens
        
        if current_tokens <= target_tokens:
            self.stats.final_tokens = current_tokens
            self.stats.final_messages = len(messages)
            self.stats.processing_time = time.time() - start_time
            return messages
        
        # 动态计算内容丢失比例阈值
        max_content_loss_ratio = self.calculate_dynamic_loss_ratio(self.stats.original_tokens, target_tokens)
        
        await progress.start_phase("迭代处理", self.valves.max_processing_iterations)
        
        self.debug_log(
            1,
            f"🔄 开始迭代处理: {current_tokens:,} -> {target_tokens:,} tokens (压缩率{target_tokens/current_tokens:.2%})",
            "🔄",
        )
        
        iteration = 0
        processed_messages = messages
        
        while iteration < self.valves.max_processing_iterations:
            await progress.update_progress(iteration + 1, self.valves.max_processing_iterations, f"第{iteration+1}轮迭代")
            
            current_tokens = self.count_messages_tokens(processed_messages)
            
            # 检查是否应该继续处理
            if not self.should_continue_processing(current_tokens, target_tokens, self.stats.original_tokens):
                self.debug_log(
                    1,
                    f"✅ 处理完成: 迭代{iteration+1}次, 最终{current_tokens:,}tokens",
                    "✅",
                )
                break
            
            # 更新统计
            self.stats.iterations = iteration + 1
            
            # 1. 智能选择消息（使用修复后的方法）
            protected_messages, to_process = self.smart_message_selection_v7(
                processed_messages, current_user_message, target_tokens, iteration
            )
            
            if not to_process:
                self.debug_log(1, f"⚠️ 没有可处理的消息，但token仍超限，强制截断", "⚠️")
                # 强制截断到目标大小
                final_messages = self.emergency_truncate_to_target(
                    protected_messages, target_tokens
                )
                break
            
            # 2. 向量检索相关消息
            if self.valves.enable_vector_retrieval and len(to_process) > 3:
                # 使用当前用户消息作为查询（而不是历史消息）
                if current_user_message:
                    self.debug_log(2, f"🔍 使用当前用户消息进行向量检索", "🔍")
                    relevant_messages = await self.vector_retrieve_relevant_messages(
                        current_user_message, to_process, progress
                    )
                    
                    # 3. 重排序
                    if self.valves.enable_reranking and len(relevant_messages) > 3:
                        self.debug_log(
                            2, f"🔄 对{len(relevant_messages)}条消息进行重排序", "🔄"
                        )
                        relevant_messages = await self.rerank_messages(
                            current_user_message, relevant_messages, progress
                        )
                    
                    to_process = relevant_messages
            
            # 4. 处理消息
            new_messages = protected_messages.copy()
            
            # 计算剩余token空间
            remaining_tokens = target_tokens - self.count_messages_tokens(protected_messages)
            
            if to_process and remaining_tokens > 1000:
                # 分析消息类型
                large_messages = []
                normal_messages = []
                
                for msg in to_process:
                    if self.count_message_tokens(msg) > self.valves.max_single_message_tokens:
                        large_messages.append(msg)
                    else:
                        normal_messages.append(msg)
                
                # 处理大消息 - 分片
                if large_messages:
                    sub_progress = ProgressTracker(progress.event_emitter)
                    await sub_progress.start_phase(f"分片处理", len(large_messages))
                    
                    self.debug_log(2, f"🧩 分片处理{len(large_messages)}条大消息", "🧩")
                    
                    for j, large_msg in enumerate(large_messages):
                        await sub_progress.update_progress(j + 1, len(large_messages), f"分片消息 {j+1}/{len(large_messages)}")
                        
                        # 分片处理
                        chunked_messages = self.chunk_message_content_v3(
                            large_msg, self.valves.chunk_target_tokens
                        )
                        
                        # 对分片进行向量检索和重排序
                        if (
                            self.valves.enable_chunk_vector_retrieval 
                            and len(chunked_messages) > 5
                            and current_user_message
                        ):
                            self.debug_log(2, f"🔍 对{len(chunked_messages)}个分片进行向量检索", "🔍")
                            relevant_chunks = await self.vector_retrieve_relevant_messages(
                                current_user_message, chunked_messages, sub_progress
                            )
                            
                            if (
                                self.valves.enable_chunk_reranking 
                                and len(relevant_chunks) > 3
                            ):
                                self.debug_log(2, f"🔄 对{len(relevant_chunks)}个分片进行重排序", "🔄")
                                relevant_chunks = await self.rerank_messages(
                                    current_user_message, relevant_chunks, sub_progress
                                )
                            
                            chunked_messages = relevant_chunks
                        
                        # 选择最相关的分片
                        max_chunks = max(
                            3, 
                            int(len(chunked_messages) * self.valves.chunk_selection_ratio)
                        )
                        selected_chunks = chunked_messages[:max_chunks]
                        
                        self.debug_log(
                            2,
                            f"🧩 分片选择: {len(chunked_messages)} -> {len(selected_chunks)}片",
                            "🧩"
                        )
                        
                        new_messages.extend(selected_chunks)
                    
                    await sub_progress.complete_phase(f"分片处理完成")
                
                # 处理普通消息 - 摘要
                if normal_messages:
                    self.debug_log(2, f"📝 摘要处理{len(normal_messages)}条普通消息", "📝")
                    
                    # 批量摘要
                    sub_progress = ProgressTracker(progress.event_emitter)
                    summary_text = await self.summarize_messages_batch(
                        normal_messages, sub_progress, iteration
                    )
                    
                    if summary_text:
                        summary_message = {
                            "role": "system",
                            "content": f"=== 📋 智能摘要 (第{iteration+1}轮) ===\n{summary_text}\n{'='*60}",
                        }
                        new_messages.append(summary_message)
                        self.debug_log(2, f"📝 摘要成功: {len(summary_text)}字符", "📝")
                    else:
                        # 摘要失败，使用分片
                        self.debug_log(2, f"📝 摘要失败，使用分片处理", "🧩")
                        
                        # 对普通消息进行分片处理
                        for normal_msg in normal_messages:
                            chunked_messages = self.chunk_message_content_v3(
                                normal_msg, self.valves.chunk_target_tokens
                            )
                            
                            # 选择部分分片
                            max_chunks = max(2, len(chunked_messages) // 2)
                            selected_chunks = chunked_messages[:max_chunks]
                            new_messages.extend(selected_chunks)
            
            processed_messages = new_messages
            iteration += 1
            
            # 检查进度
            new_tokens = self.count_messages_tokens(processed_messages)
            reduction = current_tokens - new_tokens
            
            # 计算当前窗口使用率
            current_usage = new_tokens / target_tokens if target_tokens > 0 else 0
            
            self.debug_log(
                1,
                f"📊 第{iteration}轮: {current_tokens:,} -> {new_tokens:,} tokens (减少{reduction:,}, 窗口使用{current_usage:.1%})",
                "📊",
            )
            
            # 检查是否还有足够的减少（但不能因此停止）
            if reduction < self.valves.min_reduction_threshold:
                self.debug_log(1, f"⚠️ 减少幅度过小({reduction:,}tokens)，但继续处理", "📝")
                # 继续处理，不停止
        
        # 最后检查：如果仍然超过限制，强制截断
        final_tokens = self.count_messages_tokens(processed_messages)
        if final_tokens > target_tokens:
            self.debug_log(1, f"⚠️ 仍超过限制，强制截断: {final_tokens:,} -> {target_tokens:,}", "⚠️")
            processed_messages = self.emergency_truncate_to_target(
                processed_messages, target_tokens
            )
            final_tokens = self.count_messages_tokens(processed_messages)
        
        # 更新最终统计
        self.stats.final_tokens = final_tokens
        self.stats.final_messages = len(processed_messages)
        self.stats.processing_time = time.time() - start_time
        
        # 计算最终指标
        retention_ratio = self.stats.calculate_retention_ratio()
        window_usage = self.stats.calculate_window_usage_ratio()
        
        await progress.complete_phase(f"处理完成 保留{retention_ratio:.1%} 窗口使用{window_usage:.1%}")
        
        return processed_messages

    def emergency_truncate_to_target(
        self, messages: List[dict], target_tokens: int
    ) -> List[dict]:
        """紧急截断到目标大小"""
        if not messages:
            return []
        
        self.debug_log(1, f"🆘 紧急截断到目标大小: {target_tokens:,}tokens", "🆘")
        
        # 按重要性排序
        scored_messages = []
        for msg in messages:
            content = self.extract_text_from_content(msg.get("content", ""))
            score = 0
            
            # 用户消息优先
            if msg.get("role") == "user":
                score += 1000
            
            # 系统消息其次
            if msg.get("role") == "system":
                score += 500
            
            # 高优先级内容
            if self.is_high_priority_content(content):
                score += 300
            
            # 当前用户消息最重要
            current_user_message = self.find_current_user_message(messages)
            if current_user_message and msg is current_user_message:
                score += 2000
            
            scored_messages.append((score, msg))
        
        # 按分数排序
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        # 选择消息直到达到token限制
        selected_messages = []
        current_tokens = 0
        
        for score, msg in scored_messages:
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target_tokens:
                selected_messages.append(msg)
                current_tokens += msg_tokens
            else:
                # 如果是重要消息，尝试截断
                if score > 1000:  # 重要消息
                    remaining_tokens = target_tokens - current_tokens
                    if remaining_tokens > 500:  # 至少500tokens才截断
                        content = self.extract_text_from_content(msg.get("content", ""))
                        if content:
                            # 截断内容
                            truncated_content = content[:remaining_tokens*3] + "...(截断)"
                            truncated_msg = msg.copy()
                            truncated_msg["content"] = truncated_content
                            selected_messages.append(truncated_msg)
                            current_tokens += self.count_message_tokens(truncated_msg)
                break
        
        # 保持原始顺序
        original_order = {}
        for i, msg in enumerate(messages):
            original_order[id(msg)] = i
        
        selected_messages.sort(key=lambda x: original_order.get(id(x), 0))
        
        self.debug_log(
            1,
            f"🆘 紧急截断完成: {len(messages)} -> {len(selected_messages)}条 ({current_tokens:,}tokens)",
            "🆘"
        )
        
        return selected_messages

    def print_detailed_stats(self):
        """打印详细统计信息"""
        if not self.valves.enable_detailed_stats:
            return
        
        print("\n" + "="*60)
        print(self.stats.get_summary())
        print("="*60)

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
        
        # 重置统计
        self.stats = ProcessingStats()
        
        # 创建进度追踪器
        progress = ProgressTracker(__event_emitter__)
        
        # Token分析
        original_tokens = self.count_messages_tokens(messages)
        token_limit = self.get_model_token_limit(model_name)
        
        print(f"📊 Token: {original_tokens:,}/{token_limit:,}")
        print(f"🎯 处理策略: {self.valves.processing_strategy}")
        print(f"📏 最小摘要长度: {self.calculate_min_summary_length()}tokens")
        
        # 分析当前用户消息
        current_user_message = self.find_current_user_message(messages)
        if current_user_message:
            current_tokens = self.count_message_tokens(current_user_message)
            print(f"💬 当前用户消息: {current_tokens}tokens")
        else:
            print("⚠️ 未找到当前用户消息")
        
        # 显示动态内容丢失比例
        max_loss_ratio = self.calculate_dynamic_loss_ratio(original_tokens, token_limit)
        print(f"📉 动态内容丢失比例: {max_loss_ratio:.1%}")
        
        # 计算预期压缩率和窗口使用率
        expected_compression = token_limit / original_tokens if original_tokens > 0 else 0
        print(f"📊 预期压缩率: {expected_compression:.2%}")
        print(f"🪟 目标窗口使用率: {self.valves.target_window_usage:.1%}")
        
        try:
            # 1. 多模态处理
            if self.valves.enable_detailed_progress:
                await progress.start_phase("多模态处理", 1)
            
            processed_messages = await self.process_multimodal_content(
                messages, model_name, progress
            )
            processed_tokens = self.count_messages_tokens(processed_messages)
            print(f"📊 多模态处理后: {processed_tokens:,} tokens")
            
            # 2. 内容处理
            if (
                self.valves.enable_content_maximization
                and processed_tokens > token_limit
            ):
                print(f"🔄 Token超限，开始迭代处理...")
                
                # 使用修复后的迭代处理策略
                final_messages = await self.iterative_content_processing_v4(
                    processed_messages, token_limit, progress
                )
                
                # 打印详细统计
                self.print_detailed_stats()
                
                body["messages"] = final_messages
                print("✅ 使用迭代处理后的消息")
            else:
                # 更新统计
                self.stats.original_tokens = original_tokens
                self.stats.original_messages = len(messages)
                self.stats.final_tokens = processed_tokens
                self.stats.final_messages = len(processed_messages)
                self.stats.token_limit = token_limit
                
                # 检查当前用户消息是否保留
                final_current_message = self.find_current_user_message(processed_messages)
                self.stats.current_user_message_preserved = final_current_message is not None
                
                # 计算窗口使用率
                window_usage = self.stats.calculate_window_usage_ratio()
                print(f"🪟 窗口使用率: {window_usage:.1%}")
                
                if self.valves.enable_detailed_progress:
                    await progress.complete_phase("无需处理")
                
                body["messages"] = processed_messages
                print("✅ 直接使用处理后的消息")
                
        except Exception as e:
            print(f"❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()
            
            if self.valves.enable_detailed_progress:
                await progress.update_status(f"处理失败: {str(e)[:50]}", True)
        
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