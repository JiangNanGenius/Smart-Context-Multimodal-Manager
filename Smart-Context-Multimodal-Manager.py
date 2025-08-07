"""
title: 🚀 Advanced Context Manager - Content Maximization Only v2.3.1
author: JiangNanGenius
version: 2.3.1
license: MIT
required_open_webui_version: 0.5.17
description: 修复RAG搜索无结果时的用户消息识别问题，增强容错机制
"""
import json
import hashlib
import asyncio
import re
import base64
import math
import time
import copy
import html
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum

# 导入依赖库
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


class MessageOrder:
    """消息顺序管理器 - 确保消息处理过程中顺序的正确性"""
    
    def __init__(self, original_messages: List[dict]):
        """
        初始化消息顺序管理器
        为每条消息分配唯一ID和顺序标记
        """
        self.original_messages = copy.deepcopy(original_messages)
        self.order_map = {}  # 消息ID到原始索引的映射
        self.message_ids = {}  # 原始索引到消息ID的映射
        self.content_map = {}  # 内容标识到原始索引的映射
        
        # 为每条消息分配唯一ID和顺序
        for i, msg in enumerate(self.original_messages):
            content_key = self._generate_content_key(msg)
            msg_id = hashlib.md5(f"{i}_{content_key}".encode()).hexdigest()
            
            self.order_map[msg_id] = i
            self.message_ids[i] = msg_id
            self.content_map[content_key] = i
            
            # 在消息中添加顺序标记
            msg["_order_id"] = msg_id
            msg["_original_index"] = i
            msg["_content_key"] = content_key

    def _generate_content_key(self, msg: dict) -> str:
        """生成消息内容的唯一标识"""
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # 处理多模态内容
        if isinstance(content, list):
            content_parts = []
            for item in content:
                if item.get("type") == "text":
                    content_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # 对于base64图片，只取前50个字符作为标识
                    image_data = item.get("image_url", {}).get("url", "")
                    if image_data.startswith("data:"):
                        content_parts.append(f"[IMAGE:base64:{image_data[:50]}]")
                    else:
                        content_parts.append(f"[IMAGE:url:{image_data[:50]}]")
            content_str = " ".join(content_parts)
        else:
            content_str = str(content)
        
        return f"{role}:{content_str[:200]}"

    def find_current_user_message_index(self, messages: List[dict]) -> int:
        """找到当前用户消息的索引（最后一条用户消息）"""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                return i
        return -1

    def sort_messages_preserve_user(
        self, messages: List[dict], current_user_message: dict = None
    ) -> List[dict]:
        """
        根据原始顺序排序消息，但保护当前用户消息的位置
        确保当前用户消息始终在最后
        """
        if not messages:
            return messages

        # 分离当前用户消息和其他消息
        other_messages = []
        current_user_in_list = None
        
        for msg in messages:
            if current_user_message and msg.get("_order_id") == current_user_message.get("_order_id"):
                current_user_in_list = msg
            else:
                other_messages.append(msg)

        # 按原始顺序排序其他消息
        def get_order(msg):
            return msg.get("_original_index", 999999)
        
        other_messages.sort(key=get_order)

        # 如果找到当前用户消息，将其放在最后
        if current_user_in_list:
            return other_messages + [current_user_in_list]
        else:
            return other_messages

    def get_message_preview(self, msg: dict) -> str:
        """获取消息预览用于调试"""
        if isinstance(msg.get("content"), list):
            text_parts = []
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    text_parts.append("[图片]")
            content = " ".join(text_parts)
        else:
            content = str(msg.get("content", ""))
        
        # 只做最基本的清理
        content = content.replace("\n", " ").replace("\r", " ")
        content = re.sub(r"\s+", " ", content).strip()
        return content[:100] + "..." if len(content) > 100 else content


class ProcessingStats:
    """处理统计信息记录器"""
    
    def __init__(self):
        # 基础统计
        self.original_tokens = 0
        self.original_messages = 0
        self.final_tokens = 0
        self.final_messages = 0
        self.token_limit = 0
        self.target_tokens = 0
        self.current_user_tokens = 0
        
        # 处理统计
        self.iterations = 0
        self.chunked_messages = 0
        self.summarized_messages = 0
        self.vector_retrievals = 0
        self.rerank_operations = 0
        self.multimodal_processed = 0
        self.processing_time = 0.0
        
        # 内容保留统计
        self.current_user_preserved = False
        self.preserved_messages = 0
        self.processed_messages = 0
        self.summary_messages = 0
        self.emergency_truncations = 0
        self.content_loss_ratio = 0.0
        self.discarded_messages = 0
        self.recovered_messages = 0
        self.window_utilization = 0.0
        
        # 尽量保留统计
        self.try_preserve_tokens = 0
        self.try_preserve_messages = 0
        self.try_preserve_summary_messages = 0
        
        # 新增统计
        self.keyword_generations = 0
        self.context_maximization_detections = 0
        self.chunk_created = 0
        self.chunk_processed = 0
        self.recursive_summaries = 0
        
        # 上下文最大化处理统计
        self.context_max_direct_preserve = 0
        self.context_max_chunked = 0
        self.context_max_summarized = 0
        self.multimodal_extracted = 0
        
        # 容错机制统计
        self.fallback_preserve_applied = 0
        self.user_message_recovery_count = 0
        self.rag_no_results_count = 0

    def calculate_retention_ratio(self) -> float:
        """计算内容保留比例"""
        if self.original_tokens == 0:
            return 0.0
        return self.final_tokens / self.original_tokens

    def calculate_window_usage_ratio(self) -> float:
        """计算对话窗口使用率"""
        if self.target_tokens == 0:
            return 0.0
        return self.final_tokens / self.target_tokens

    def calculate_compression_ratio(self) -> float:
        """计算压缩比例"""
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.final_tokens) / self.original_tokens

    def calculate_processing_efficiency(self) -> float:
        """计算处理效率"""
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
📊 内容最大化处理统计报告:
├─ 📥 输入: {self.original_messages}条消息, {self.original_tokens:,}tokens
├─ 📤 输出: {self.final_messages}条消息, {self.final_tokens:,}tokens
├─ 🎯 模型限制: {self.token_limit:,}tokens
├─ 🪟 目标窗口: {self.target_tokens:,}tokens
├─ 👤 当前用户: {self.current_user_tokens:,}tokens
├─ 📈 内容保留率: {retention:.2%}
├─ 🪟 窗口使用率: {window_usage:.2%}
├─ 📉 压缩比例: {compression:.2%}
├─ ⚡ 处理效率: {efficiency:.0f}tokens/s
├─ 🔄 迭代次数: {self.iterations}
├─ 🧩 分片处理: {self.chunk_created}个分片，{self.chunk_processed}个处理
├─ 📝 摘要压缩: {self.summarized_messages}条
├─ 🔄 递归摘要: {self.recursive_summaries}次
├─ 🔍 向量检索: {self.vector_retrievals}次
├─ 🔄 重排序: {self.rerank_operations}次
├─ 🖼️ 多模态处理: {self.multimodal_processed}张图片
├─ 🔑 关键字生成: {self.keyword_generations}次
├─ 📚 上下文最大化检测: {self.context_maximization_detections}次
├─ 📚 上下文最大化处理: 直接保留{self.context_max_direct_preserve}条, 分片{self.context_max_chunked}条, 摘要{self.context_max_summarized}条
├─ 🎨 多模态提取: {self.multimodal_extracted}个多模态消息
├─ 💬 当前用户: {'✅已保留' if self.current_user_preserved else '❌未保留'}
├─ 🔒 尽量保留: {self.try_preserve_messages}条消息({self.try_preserve_tokens:,}tokens)
├─ 📝 尽量保留摘要: {self.try_preserve_summary_messages}条
├─ 🔄 合并内容: {self.recovered_messages}条
├─ 🆘 紧急截断: {self.emergency_truncations}次
├─ 🛡️ 容错保护: 后备保留{self.fallback_preserve_applied}次, 用户消息恢复{self.user_message_recovery_count}次
├─ 🔍 RAG无结果: {self.rag_no_results_count}次
└─ ⏱️ 处理时间: {self.processing_time:.2f}秒"""


class ProgressTracker:
    """进度追踪器 - 用于显示处理进度"""
    
    def __init__(self, event_emitter):
        self.event_emitter = event_emitter
        self.current_step = 0
        self.total_steps = 0
        self.current_phase = ""
        self.phase_progress = 0
        self.phase_total = 0
        self.logged_phases = set()  # 防止重复日志

    def create_progress_bar(self, percentage: float, width: int = 15) -> str:
        """创建美观的进度条"""
        filled = int(percentage * width / 100)
        if percentage >= 100:
            bar = "█" * width
        else:
            bar = "█" * filled + "▓" * max(0, 1) + "░" * max(0, width - filled - 1)
        return f"[{bar}] {percentage:.1f}%"

    async def start_phase(self, phase_name: str, total_items: int = 0):
        """开始新阶段"""
        self.current_phase = phase_name
        self.phase_progress = 0
        self.phase_total = total_items
        self.logged_phases.add(phase_name)
        await self.update_status(f"🚀 开始 {phase_name}")

    async def update_progress(self, completed: int, total: int = None, detail: str = ""):
        """更新进度"""
        if total is None:
            total = self.phase_total
        
        self.phase_progress = completed
        if total > 0:
            percentage = (completed / total) * 100
            progress_bar = self.create_progress_bar(percentage)
            status = f"🔄 {self.current_phase} {progress_bar} ({completed}/{total})"
            if detail:
                status += f" - {detail}"
        else:
            status = f"🔄 {self.current_phase}"
            if detail:
                status += f" - {detail}"
        
        await self.update_status(status, False)

    async def complete_phase(self, message: str = ""):
        """完成当前阶段"""
        final_message = f"✅ {self.current_phase} 完成"
        if message:
            final_message += f" - {message}"
        await self.update_status(final_message, True)

    async def update_status(self, message: str, done: bool = False):
        """更新状态"""
        if self.event_emitter:
            try:
                # 基本清理，保留主要内容
                message = message.replace("\n", " ").replace("\r", " ")
                message = re.sub(r"\s+", " ", message).strip()
                await self.event_emitter({
                    "type": "status",
                    "data": {"description": message, "done": done},
                })
            except Exception as e:
                # 避免重复日志
                if str(e) not in self.logged_phases:
                    print(f"⚠️ 进度更新失败: {e}")
                    self.logged_phases.add(str(e))


class ModelMatcher:
    """智能模型匹配器 - 支持模糊匹配但避免thinking模型误匹配"""
    
    def __init__(self):
        # 定义模型匹配规则
        self.exact_matches = {
            # GPT系列
            "gpt-4o": {"family": "gpt", "multimodal": True, "limit": 128000},
            "gpt-4o-mini": {"family": "gpt", "multimodal": True, "limit": 128000},
            "gpt-4": {"family": "gpt", "multimodal": False, "limit": 8192},
            "gpt-4-turbo": {"family": "gpt", "multimodal": True, "limit": 128000},
            "gpt-4-vision-preview": {"family": "gpt", "multimodal": True, "limit": 128000},
            "gpt-3.5-turbo": {"family": "gpt", "multimodal": False, "limit": 16385},
            
            # Claude系列
            "claude-3-5-sonnet": {"family": "claude", "multimodal": True, "limit": 200000},
            "claude-3-opus": {"family": "claude", "multimodal": True, "limit": 200000},
            "claude-3-haiku": {"family": "claude", "multimodal": True, "limit": 200000},
            "claude-3": {"family": "claude", "multimodal": True, "limit": 200000},
            "anthropic.claude-4-sonnet-latest-extended-thinking": {
                "family": "claude", "multimodal": True, "limit": 200000, "special": "thinking"
            },
            
            # Doubao系列
            "doubao-1.5-vision-pro": {"family": "doubao", "multimodal": True, "limit": 128000},
            "doubao-1.5-vision-lite": {"family": "doubao", "multimodal": True, "limit": 128000},
            "doubao-1.5-thinking-pro": {"family": "doubao", "multimodal": False, "limit": 128000, "special": "thinking"},
            "doubao-seed-1-6-250615": {"family": "doubao", "multimodal": True, "limit": 50000},
            "doubao-seed": {"family": "doubao", "multimodal": False, "limit": 50000},
            "doubao": {"family": "doubao", "multimodal": False, "limit": 50000},
            "doubao-1-5-pro-256k": {"family": "doubao", "multimodal": False, "limit": 200000},
            
            # Gemini系列
            "gemini-pro": {"family": "gemini", "multimodal": False, "limit": 128000},
            "gemini-pro-vision": {"family": "gemini", "multimodal": True, "limit": 128000},
            
            # Qwen系列
            "qwen-vl": {"family": "qwen", "multimodal": True, "limit": 32000},
        }
        
        # 模糊匹配规则（按优先级排序）
        self.fuzzy_patterns = [
            # Thinking模型优先匹配（避免误匹配）
            {"pattern": r".*thinking.*", "family": "thinking", "multimodal": False, "limit": 200000, "special": "thinking"},
            
            # GPT系列模糊匹配
            {"pattern": r"gpt-4o.*", "family": "gpt", "multimodal": True, "limit": 128000},
            {"pattern": r"gpt-4.*vision.*", "family": "gpt", "multimodal": True, "limit": 128000},
            {"pattern": r"gpt-4.*turbo.*", "family": "gpt", "multimodal": True, "limit": 128000},
            {"pattern": r"gpt-4.*", "family": "gpt", "multimodal": False, "limit": 8192},
            {"pattern": r"gpt-3\.5.*", "family": "gpt", "multimodal": False, "limit": 16385},
            {"pattern": r"gpt.*", "family": "gpt", "multimodal": False, "limit": 16385},
            
            # Claude系列模糊匹配
            {"pattern": r"claude.*3.*5.*sonnet.*", "family": "claude", "multimodal": True, "limit": 200000},
            {"pattern": r"claude.*3.*opus.*", "family": "claude", "multimodal": True, "limit": 200000},
            {"pattern": r"claude.*3.*haiku.*", "family": "claude", "multimodal": True, "limit": 200000},
            {"pattern": r"claude.*3.*", "family": "claude", "multimodal": True, "limit": 200000},
            {"pattern": r"claude.*", "family": "claude", "multimodal": True, "limit": 200000},
            {"pattern": r"anthropic.*claude.*", "family": "claude", "multimodal": True, "limit": 200000},
            
            # Doubao系列模糊匹配
            {"pattern": r"doubao.*vision.*", "family": "doubao", "multimodal": True, "limit": 128000},
            {"pattern": r"doubao.*seed.*", "family": "doubao", "multimodal": True, "limit": 50000},
            {"pattern": r"doubao.*256k.*", "family": "doubao", "multimodal": False, "limit": 200000},
            {"pattern": r"doubao.*1\.5.*", "family": "doubao", "multimodal": False, "limit": 128000},
            {"pattern": r"doubao.*", "family": "doubao", "multimodal": False, "limit": 50000},
            
            # Gemini系列模糊匹配
            {"pattern": r"gemini.*vision.*", "family": "gemini", "multimodal": True, "limit": 128000},
            {"pattern": r"gemini.*pro.*", "family": "gemini", "multimodal": False, "limit": 128000},
            {"pattern": r"gemini.*", "family": "gemini", "multimodal": False, "limit": 128000},
            
            # Qwen系列模糊匹配
            {"pattern": r"qwen.*vl.*", "family": "qwen", "multimodal": True, "limit": 32000},
            {"pattern": r"qwen.*", "family": "qwen", "multimodal": False, "limit": 32000},
        ]

    def match_model(self, model_name: str) -> Dict[str, Any]:
        """智能匹配模型信息"""
        if not model_name:
            return {"family": "unknown", "multimodal": False, "limit": 200000}
        
        model_lower = model_name.lower().strip()
        
        # 1. 精确匹配
        for exact_name, info in self.exact_matches.items():
            if exact_name.lower() == model_lower:
                return {**info, "matched_name": exact_name, "match_type": "exact"}
        
        # 2. 模糊匹配
        for pattern_info in self.fuzzy_patterns:
            pattern = pattern_info["pattern"]
            if re.match(pattern, model_lower):
                return {
                    "family": pattern_info["family"],
                    "multimodal": pattern_info["multimodal"],
                    "limit": pattern_info["limit"],
                    "special": pattern_info.get("special"),
                    "matched_pattern": pattern,
                    "match_type": "fuzzy"
                }
        
        # 3. 默认匹配
        return {"family": "unknown", "multimodal": False, "limit": 200000, "match_type": "default"}


class TokenCalculator:
    """简化的Token计算器 - 只用tiktoken"""
    
    def __init__(self):
        self._encoding = None

    def get_encoding(self):
        """获取tiktoken编码器"""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """简化的token计算 - 只用tiktoken"""
        if not text:
            return 0
        
        # 获取tiktoken编码器
        encoding = self.get_encoding()
        if encoding:
            try:
                return len(encoding.encode(str(text)))
            except Exception:
                pass
        
        # 简单fallback
        return len(str(text)) // 4

    def calculate_image_tokens(self, image_) -> int:
        """简化的图片token计算"""
        if not image_data:
            return 0
        # 简单估算：每个图片按1500tokens计算
        return 1500


class Filter:
    class Valves(BaseModel):
        # 基础控制
        enable_processing: bool = Field(default=True, description="🔄 启用内容最大化处理")
        excluded_models: str = Field(default="", description="🚫 排除模型列表(逗号分隔)")
        
        # 核心配置 - 内容最大化专用
        max_window_utilization: float = Field(default=0.95, description="🪟 最大窗口利用率(95%)")
        aggressive_content_recovery: bool = Field(default=True, description="🔄 激进内容合并模式")
        min_preserve_ratio: float = Field(default=0.75, description="🔒 最小内容保留比例(75%)")
        
        # 尽量保留配置
        enable_try_preserve: bool = Field(default=True, description="🔒 启用尽量保留机制")
        try_preserve_ratio: float = Field(default=0.40, description="🔒 尽量保留预算比例(40%)")
        try_preserve_exchanges: int = Field(default=3, description="🔒 尽量保留对话轮次数")
        
        # 响应空间配置
        response_buffer_ratio: float = Field(default=0.06, description="📝 响应空间预留比例(6%)")
        response_buffer_max: int = Field(default=3000, description="📝 响应空间最大值(tokens)")
        response_buffer_min: int = Field(default=1000, description="📝 响应空间最小值(tokens)")
        
        # 多模态处理配置
        multimodal_direct_threshold: float = Field(default=0.70, description="🎯 多模态直接输入Token预算阈值(70%)")
        preserve_images_in_multimodal: bool = Field(default=True, description="📸 多模态模型是否保留原始图片")
        always_process_images_before_summary: bool = Field(default=True, description="📝 摘要前总是先处理图片")
        
        # 上下文最大化处理配置
        enable_context_maximization: bool = Field(default=True, description="📚 启用上下文最大化处理")
        context_max_direct_preserve_ratio: float = Field(default=0.30, description="📚 上下文最大化直接保留比例(30%)")
        context_max_skip_rag: bool = Field(default=True, description="📚 上下文最大化跳过RAG处理")
        context_max_prioritize_recent: bool = Field(default=True, description="📚 上下文最大化优先保留最近内容")
        
        # 容错机制配置
        enable_fallback_preservation: bool = Field(default=True, description="🛡️ 启用容错保护机制")
        fallback_preserve_ratio: float = Field(default=0.20, description="🛡️ 容错保护预留比例(20%)")
        min_history_messages: int = Field(default=5, description="🛡️ 最少历史消息数量")
        force_preserve_recent_user_exchanges: int = Field(default=2, description="🛡️ 强制保留最近用户对话轮次")
        
        # 功能开关
        enable_multimodal: bool = Field(default=True, description="🖼️ 启用多模态处理")
        enable_vision_preprocessing: bool = Field(default=True, description="👁️ 启用图片预处理")
        enable_vector_retrieval: bool = Field(default=True, description="🔍 启用向量检索")
        enable_intelligent_chunking: bool = Field(default=True, description="🧩 启用智能分片")
        enable_recursive_summarization: bool = Field(default=True, description="🔄 启用递归摘要")
        enable_reranking: bool = Field(default=True, description="🔄 启用重排序")
        
        # 智能关键字生成和上下文最大化检测
        enable_keyword_generation: bool = Field(default=True, description="🔑 启用智能关键字生成")
        enable_ai_context_max_detection: bool = Field(default=True, description="🧠 启用AI上下文最大化检测")
        keyword_generation_for_context_max: bool = Field(default=True, description="🔑 对上下文最大化启用关键字生成")
        
        # 统计和调试
        enable_detailed_stats: bool = Field(default=True, description="📊 启用详细统计")
        enable_detailed_progress: bool = Field(default=True, description="📱 启用详细进度显示")
        debug_level: int = Field(default=2, description="🐛 调试级别 0-3")
        show_frontend_progress: bool = Field(default=True, description="📱 显示处理进度")
        
        # API配置
        api_error_retry_times: int = Field(default=2, description="🔄 API错误重试次数")
        api_error_retry_delay: float = Field(default=1.0, description="⏱️ API错误重试延迟(秒)")
        
        # Token管理
        default_token_limit: int = Field(default=200000, description="⚖️ 默认token限制")
        token_safety_ratio: float = Field(default=0.92, description="🛡️ Token安全比例(92%)")
        target_window_usage: float = Field(default=0.85, description="🪟 目标窗口使用率(85%)")
        max_processing_iterations: int = Field(default=5, description="🔄 最大处理迭代次数")
        
        # 保护策略
        force_preserve_current_user_message: bool = Field(default=True, description="🔒 强制保留当前用户消息(最后一条用户消息)")
        preserve_recent_exchanges: int = Field(default=4, description="💬 保护最近完整对话轮次")
        max_preserve_ratio: float = Field(default=0.3, description="🔒 保护消息最大token比例")
        max_single_message_tokens: int = Field(default=20000, description="📝 单条消息最大token")
        
        # 智能分片配置
        enable_smart_chunking: bool = Field(default=True, description="🧩 启用智能分片")
        chunk_target_tokens: int = Field(default=4000, description="🧩 分片目标token数")
        chunk_overlap_tokens: int = Field(default=400, description="🔗 分片重叠token数")
        chunk_min_tokens: int = Field(default=1000, description="📏 分片最小token数")
        chunk_max_tokens: int = Field(default=8000, description="📏 分片最大token数")
        large_message_threshold: int = Field(default=10000, description="📏 大消息分片阈值")
        preserve_paragraph_integrity: bool = Field(default=True, description="📝 保持段落完整性")
        preserve_sentence_integrity: bool = Field(default=True, description="📝 保持句子完整性")
        preserve_code_blocks: bool = Field(default=True, description="💻 保持代码块完整性")
        
        # 内容优先级设置
        high_priority_content: str = Field(
            default="代码,配置,参数,数据,错误,解决方案,步骤,方法,技术细节,API,函数,类,变量,问题,bug,修复,实现,算法,架构,用户问题,关键回答",
            description="🎯 高优先级内容关键词(逗号分隔)"
        )
        
        # 统一的API配置 - 简化模型配置
        api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="🔗 API基础地址")
        api_key: str = Field(default="", description="🔑 API密钥")
        
        # 多模态模型配置（Vision和多模态摘要共用）
        multimodal_model: str = Field(default="doubao-1.5-vision-pro-250328", description="🖼️ 多模态模型")
        
        # 文本模型配置（摘要、关键字生成、上下文最大化检测共用）
        text_model: str = Field(default="doubao-1-5-lite-32k-250115", description="📝 文本处理模型")
        
        # 向量模型配置
        text_vector_model: str = Field(default="doubao-embedding-large-text-250515", description="🧠 文本向量模型")
        multimodal_vector_model: str = Field(default="doubao-embedding-vision-250615", description="🧠 多模态向量模型")
        
        # Vision相关配置
        vision_prompt_template: str = Field(
            default="请详细描述这张图片的内容，包括主要对象、场景、文字、颜色、布局等所有可见信息。特别注意代码、配置、数据等技术信息。保持客观准确，重点突出关键信息。",
            description="👁️ Vision提示词"
        )
        vision_max_tokens: int = Field(default=2000, description="👁️ Vision最大输出tokens")
        
        # 关键字生成配置
        keyword_generation_prompt: str = Field(
            default="""你是专业的搜索关键字生成助手。用户输入了一个查询，你需要生成多个相关的搜索关键字来帮助在对话历史中找到相关内容。

📋 任务要求：
1. 分析用户查询的意图和主题
2. 生成5-10个相关的搜索关键字
3. 包含同义词、相关词、技术术语
4. 对于宽泛查询（如"聊了什么"、"说了什么"），生成通用但有效的关键字
5. 关键字应该能覆盖可能的对话主题

📝 输出格式：
直接输出关键字，用逗号分隔，不要其他解释。

现在请为以下查询生成关键字：""",
            description="🔑 关键字生成提示词"
        )
        
        # 上下文最大化检测配置
        context_max_detection_prompt: str = Field(
            default="""你是专业的查询意图分析助手。请分析用户的查询是否需要上下文最大化处理。

📋 判断标准：
需要上下文最大化的查询特征：
- 询问"聊了什么"、"说了什么"、"讨论了什么"等宽泛内容
- 询问"之前的内容"、"历史记录"、"对话历史"等
- 缺乏具体的主题、关键词或明确的搜索意图
- 查询词汇少于3个有效词汇

不需要上下文最大化的查询特征：
- 包含明确的主题、技术术语、产品名称等
- 有具体的问题指向
- 包含详细的描述或背景信息

📝 输出格式：
只输出 "需要上下文最大化" 或 "不需要上下文最大化"，不要其他解释。

现在请分析以下查询：""",
            description="🧠 上下文最大化检测提示词"
        )
        
        # 向量检索配置
        vector_similarity_threshold: float = Field(default=0.06, description="🎯 基础相似度阈值")
        multimodal_similarity_threshold: float = Field(default=0.04, description="🖼️ 多模态相似度阈值")
        text_similarity_threshold: float = Field(default=0.08, description="📝 文本相似度阈值")
        vector_top_k: int = Field(default=150, description="🔝 向量检索Top-K数量")
        
        # 重排序API配置
        rerank_api_base: str = Field(default="https://api.bochaai.com", description="🔄 重排序API")
        rerank_api_key: str = Field(default="", description="🔑 重排序密钥")
        rerank_model: str = Field(default="gte-rerank", description="🧠 重排序模型")
        rerank_top_k: int = Field(default=100, description="🔝 重排序返回数量")
        
        # 摘要配置
        max_summary_length: int = Field(default=25000, description="📏 摘要最大长度")
        min_summary_ratio: float = Field(default=0.30, description="📏 摘要最小长度比例")
        summary_compression_ratio: float = Field(default=0.40, description="📊 摘要压缩比例")
        max_recursion_depth: int = Field(default=3, description="🔄 最大递归深度")
        
        # 性能配置
        max_concurrent_requests: int = Field(default=6, description="⚡ 最大并发数")
        request_timeout: int = Field(default=45, description="⏱️ 请求超时(秒)")

    def __init__(self):
        print("\n" + "=" * 60)
        print("🚀 Advanced Context Manager v2.3.1 - 修复RAG搜索无结果问题")
        print("📍 插件正在初始化...")
        print("🔧 增强容错机制，确保用户消息正确识别...")
        
        self.valves = self.Valves()
        
        # 初始化组件
        self.model_matcher = ModelMatcher()
        self.token_calculator = TokenCalculator()
        
        # 处理统计
        self.stats = ProcessingStats()
        
        # 消息顺序管理器
        self.message_order = None
        self.current_processing_id = None
        self.current_user_message = None
        self.current_model_info = None
        
        # 解析配置
        self._parse_configurations()
        
        print(f"✅ 插件初始化完成")
        print(f"🔥 内容最大化模式: 启用")
        print(f"🪟 最大窗口利用率: {self.valves.max_window_utilization:.1%}")
        print(f"🔄 激进内容合并: {self.valves.aggressive_content_recovery}")
        print(f"🔒 尽量保留机制: {self.valves.enable_try_preserve} (预算:{self.valves.try_preserve_ratio:.1%})")
        print(f"📚 上下文最大化: {self.valves.enable_context_maximization} (直接保留:{self.valves.context_max_direct_preserve_ratio:.1%})")
        print(f"🛡️ 容错保护机制: {self.valves.enable_fallback_preservation} (预留:{self.valves.fallback_preserve_ratio:.1%})")
        print(f"🔑 智能关键字生成: {self.valves.enable_keyword_generation}")
        print(f"🧠 AI上下文最大化检测: {self.valves.enable_ai_context_max_detection}")
        print(f"🧩 智能分片: {self.valves.enable_smart_chunking} (阈值:{self.valves.large_message_threshold:,}tokens)")
        print(f"📊 Token计算器: 简化版（仅用tiktoken）")
        print(f"🎯 模型匹配器: 智能模糊匹配")
        print(f"🖼️ 多模态模型: {self.valves.multimodal_model}")
        print(f"📝 文本处理模型: {self.valves.text_model}")
        print(f"📊 详细统计: {self.valves.enable_detailed_stats}")
        print(f"🐛 调试级别: {self.valves.debug_level}")
        print("=" * 60 + "\n")

    def _parse_configurations(self):
        """解析配置项"""
        # 解析高优先级内容关键词
        self.high_priority_keywords = set()
        if self.valves.high_priority_content:
            self.high_priority_keywords = {
                keyword.strip().lower() 
                for keyword in self.valves.high_priority_content.split(",") 
                if keyword.strip()
            }

    def reset_processing_state(self):
        """重置处理状态"""
        self.current_processing_id = None
        self.message_order = None
        self.current_user_message = None
        self.current_model_info = None
        self.stats = ProcessingStats()

    def debug_log(self, level: int, message: str, emoji: str = "🔧"):
        """分级调试日志"""
        if self.valves.debug_level >= level:
            prefix = ["", "🐛[DEBUG]", "🔍[DETAIL]", "📋[VERBOSE]"][min(level, 3)]
            # 基本清理，保留主要内容
            message = message.replace("\n", " ").replace("\r", " ")
            message = re.sub(r"\s+", " ", message).strip()
            print(f"{prefix} {emoji} {message}")

    def is_model_excluded(self, model_name: str) -> bool:
        """检查模型是否被排除"""
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

    def analyze_model(self, model_name: str) -> Dict[str, Any]:
        """分析模型信息 - 使用智能匹配器"""
        model_info = self.model_matcher.match_model(model_name)
        
        self.debug_log(2, f"模型分析: {model_name} -> {model_info['family']} "
                          f"({'多模态' if model_info['multimodal'] else '文本'}) "
                          f"{model_info['limit']:,}tokens "
                          f"[{model_info['match_type']}匹配]", "🎯")
        
        if model_info.get("special") == "thinking":
            self.debug_log(1, f"检测到Thinking模型: {model_name}", "🧠")
        
        return model_info

    def count_tokens(self, text: str) -> int:
        """简化的token计算"""
        if not text:
            return 0
        return self.token_calculator.count_tokens(text)

    def count_message_tokens(self, message: dict) -> int:
        """计算单条消息的token数量 - 简化版本"""
        if not message:
            return 0
        
        content = message.get("content", "")
        role = message.get("role", "")
        total_tokens = 0
        
        # 处理多模态内容
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += self.count_tokens(text)
                elif item.get("type") == "image_url":
                    # 简化图片token计算
                    total_tokens += self.token_calculator.calculate_image_tokens("")
        else:
            # 纯文本内容
            total_tokens = self.count_tokens(content)
        
        # 加上角色和格式开销
        total_tokens += self.count_tokens(role) + 20  # 格式开销
        
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数量"""
        if not messages:
            return 0
        
        total_tokens = sum(self.count_message_tokens(msg) for msg in messages)
        self.debug_log(2, f"消息列表token计算: {len(messages)}条消息 -> {total_tokens:,}tokens", "📊")
        return total_tokens

    def get_model_token_limit(self, model_name: str) -> int:
        """获取模型的token限制（应用安全系数）"""
        model_info = self.analyze_model(model_name)
        limit = model_info.get("limit", self.valves.default_token_limit)
        safe_limit = int(limit * self.valves.token_safety_ratio)
        
        self.debug_log(2, f"模型token限制: {model_name} -> {limit} -> {safe_limit}", "⚖️")
        return safe_limit

    def is_multimodal_model(self, model_name: str) -> bool:
        """判断模型是否支持多模态输入"""
        model_info = self.analyze_model(model_name)
        return model_info.get("multimodal", False)

    def should_force_vision_processing(self, model_name: str) -> bool:
        """判断是否强制进行视觉处理"""
        # 对于非多模态模型，强制进行视觉处理
        return not self.is_multimodal_model(model_name)

    def find_current_user_message(self, messages: List[dict]) -> Optional[dict]:
        """查找当前用户消息（最新的用户输入）- 修复版本"""
        if not messages:
            return None
        
        # 从最后一条消息开始查找，找到最新的用户消息
        for msg in reversed(messages):
            if msg.get("role") == "user":
                self.debug_log(2, f"找到当前用户消息: {len(self.extract_text_from_content(msg.get('content', '')))}字符", "💬")
                return msg
        
        return None

    def separate_current_and_history_messages(self, messages: List[dict]) -> Tuple[Optional[dict], List[dict]]:
        """分离当前用户消息和历史消息 - 修复版本，使用消息ID而不是对象引用"""
        if not messages:
            return None, []
        
        # 找到当前用户消息（最新的用户输入）
        current_user_message = self.find_current_user_message(messages)
        if not current_user_message:
            return None, messages
        
        # 获取当前用户消息的唯一标识
        current_user_id = current_user_message.get("_order_id")
        if not current_user_id:
            # 如果没有order_id，使用内容key作为备选
            current_user_id = current_user_message.get("_content_key")
        
        # 分离历史消息（除了当前用户消息之外的所有消息）
        history_messages = []
        for msg in messages:
            msg_id = msg.get("_order_id") or msg.get("_content_key")
            # 使用ID匹配而不是对象引用匹配
            if msg_id != current_user_id:
                history_messages.append(msg)
        
        self.debug_log(1, f"消息分离: 当前用户消息1条({self.count_message_tokens(current_user_message)}tokens), "
                          f"历史消息{len(history_messages)}条({self.count_messages_tokens(history_messages):,}tokens)", "📋")
        
        return current_user_message, history_messages

    def calculate_target_tokens(self, model_name: str, current_user_tokens: int) -> int:
        """计算目标token数：模型限制 - 当前用户消息 - 响应空间"""
        model_token_limit = self.get_model_token_limit(model_name)
        
        # 计算响应空间
        response_buffer = min(
            self.valves.response_buffer_max,
            max(
                self.valves.response_buffer_min,
                int(model_token_limit * self.valves.response_buffer_ratio)
            )
        )
        
        # 计算目标：总限制 - 当前用户消息 - 响应缓冲区
        target_tokens = model_token_limit - current_user_tokens - response_buffer
        
        # 确保不小于基础值
        min_target = max(10000, model_token_limit * 0.3)
        target_tokens = max(target_tokens, min_target)
        
        self.debug_log(1, f"🎯 目标token计算: {model_token_limit} - {current_user_tokens} - {response_buffer} = {target_tokens}", "🎯")
        return int(target_tokens)

    # ========== 多模态处理相关方法 ==========
    
    def has_images_in_content(self, content) -> bool:
        """检查内容中是否包含图片"""
        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)
        return False

    def has_images_in_messages(self, messages: List[dict]) -> bool:
        """检查消息列表中是否包含图片"""
        return any(self.has_images_in_content(msg.get("content")) for msg in messages)

    def extract_text_from_content(self, content) -> str:
        """从内容中提取文本 - 最小化清理"""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    # 直接使用原始文本，不过度清理
                    text = item.get("text", "")
                    text_parts.append(text)
            return " ".join(text_parts)
        else:
            # 直接返回原始内容，只做基本的字符串转换
            return str(content) if content else ""

    def extract_images_from_content(self, content) -> List[dict]:
        """从内容中提取图片信息"""
        if isinstance(content, list):
            images = []
            for item in content:
                if item.get("type") == "image_url":
                    images.append(item)
            return images
        return []

    def extract_multimodal_messages(self, messages: List[dict]) -> Tuple[List[dict], List[dict]]:
        """提取多模态消息和纯文本消息"""
        multimodal_messages = []
        text_messages = []
        
        for msg in messages:
            if self.has_images_in_content(msg.get("content")):
                multimodal_messages.append(msg)
            else:
                text_messages.append(msg)
        
        self.debug_log(2, f"多模态消息提取: {len(multimodal_messages)}条多模态, {len(text_messages)}条文本", "🎨")
        return multimodal_messages, text_messages

    def is_high_priority_content(self, text: str) -> bool:
        """判断是否为高优先级内容"""
        if not text or not self.high_priority_keywords:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.high_priority_keywords)

    def calculate_multimodal_budget_sufficient(self, messages: List[dict], target_tokens: int) -> bool:
        """计算多模态模型的Token预算是否充足"""
        current_tokens = self.count_messages_tokens(messages)
        usage_ratio = current_tokens / target_tokens if target_tokens > 0 else 1.0
        threshold = self.valves.multimodal_direct_threshold
        is_sufficient = usage_ratio <= threshold
        
        self.debug_log(1, f"🎯 多模态预算检查: {current_tokens:,}/{target_tokens:,} = {usage_ratio:.2%} "
                          f"{'≤' if is_sufficient else '>'} {threshold:.1%}", "💰")
        
        return is_sufficient

    # ========== 统一的API客户端管理 ==========
    
    def get_api_client(self, client_type: str = "default"):
        """获取API客户端 - 统一管理"""
        if not OPENAI_AVAILABLE:
            return None
        
        if self.valves.api_key:
            return AsyncOpenAI(
                base_url=self.valves.api_base,
                api_key=self.valves.api_key,
                timeout=self.valves.request_timeout
            )
        return None

    # ========== 安全API调用 ==========
    
    def is_json_response(self, content: str) -> bool:
        """检查响应是否为JSON格式"""
        if not content:
            return False
        content = content.strip()
        return content.startswith("{") or content.startswith("[")

    def extract_error_info(self, content: str) -> str:
        """从错误响应中提取关键信息"""
        if not content:
            return "空响应"
        
        # 基本清理
        content = content.replace("\n", " ").replace("\r", " ")
        content = re.sub(r"\s+", " ", content).strip()
        
        # 检查是否为HTML错误页面
        if content.strip().startswith("<!DOCTYPE") or "<html" in content:
            title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
            if title_match:
                return f"HTML错误页面: {title_match.group(1)}"
            return "HTML错误页面"
        
        # 检查是否为JSON错误
        try:
            if self.is_json_response(content):
                error_data = json.loads(content)
                if isinstance(error_data, dict):
                    error_msg = error_data.get("error", error_data.get("message", ""))
                    if error_msg:
                        return f"API错误: {error_msg}"
            
            return f"响应内容: {content[:200]}..."
        except Exception:
            return f"响应内容: {content[:200]}..."

    async def safe_api_call(self, call_func, call_name: str, *args, **kwargs):
        """安全的API调用包装器"""
        for attempt in range(self.valves.api_error_retry_times + 1):
            try:
                result = await call_func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否为JSON解析错误
                if "is not valid JSON" in error_msg or "Unexpected token" in error_msg:
                    self.debug_log(1, f"{call_name} JSON解析错误: {error_msg}", "❌")
                    return None
                
                if attempt < self.valves.api_error_retry_times:
                    self.debug_log(1, f"{call_name} 第{attempt+1}次尝试失败，{self.valves.api_error_retry_delay}秒后重试: {error_msg}", "🔄")
                    await asyncio.sleep(self.valves.api_error_retry_delay)
                else:
                    self.debug_log(1, f"{call_name} 最终失败: {error_msg}", "❌")
                    return None
        
        return None

    # ========== 智能上下文最大化检测 ==========
    
    async def _detect_context_max_need_impl(self, query_text: str, event_emitter):
        """实际的上下文最大化检测实现"""
        client = self.get_api_client()
        if not client:
            return None
        
        # 基本清理查询文本
        cleaned_query = query_text.replace("\n", " ").replace("\r", " ")
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
        
        # 构建提示
        prompt = f"{self.valves.context_max_detection_prompt}\n\n{cleaned_query}"
        
        response = await client.chat.completions.create(
            model=self.valves.text_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
            timeout=self.valves.request_timeout
        )
        
        if response.choices and response.choices[0].message.content:
            result = response.choices[0].message.content.strip()
            # 基本清理结果
            result = result.replace("\n", " ").replace("\r", " ")
            result = re.sub(r"\s+", " ", result).strip()
            
            need_context_max = "需要上下文最大化" in result
            self.debug_log(2, f"AI上下文最大化检测结果: {result} -> {need_context_max}", "🧠")
            return need_context_max
        
        return None

    async def detect_context_max_need(self, query_text: str, event_emitter) -> bool:
        """使用AI检测是否需要上下文最大化"""
        if not self.valves.enable_ai_context_max_detection:
            # 回退到简单的模式匹配
            return self.is_context_max_need_simple(query_text)
        
        self.debug_log(1, f"🧠 AI检测上下文最大化需求: {query_text}", "🧠")
        
        # 调用AI检测
        need_context_max = await self.safe_api_call(
            self._detect_context_max_need_impl, "上下文最大化检测", query_text, event_emitter
        )
        
        if need_context_max is not None:
            self.stats.context_maximization_detections += 1
            self.debug_log(1, f"🧠 AI上下文最大化检测完成: {'需要' if need_context_max else '不需要'}", "🧠")
            return need_context_max
        else:
            # AI检测失败，回退到简单方法
            self.debug_log(1, f"🧠 AI检测失败，使用简单方法", "⚠️")
            return self.is_context_max_need_simple(query_text)

    def is_context_max_need_simple(self, query_text: str) -> bool:
        """简单的上下文最大化需求判断（备用方法）"""
        if not query_text:
            return True
        
        # 需要上下文最大化的特征
        context_max_patterns = [
            r".*聊.*什么.*",
            r".*说.*什么.*",
            r".*讨论.*什么.*",
            r".*谈.*什么.*",
            r".*内容.*",
            r".*话题.*",
            r".*历史.*",
            r".*记录.*",
            r".*之前.*",
            r"what.*discuss.*",
            r"what.*talk.*",
            r"what.*chat.*",
            r".*conversation.*",
            r".*history.*",
        ]
        
        query_lower = query_text.lower()
        for pattern in context_max_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return len(query_text.split()) <= 3  # 3个词以内也认为需要上下文最大化

    # ========== 智能关键字生成 ==========
    
    async def _generate_keywords_impl(self, query_text: str, event_emitter):
        """实际的关键字生成实现"""
        client = self.get_api_client()
        if not client:
            return None
        
        # 基本清理查询文本
        cleaned_query = query_text.replace("\n", " ").replace("\r", " ")
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
        
        # 构建提示
        prompt = f"{self.valves.keyword_generation_prompt}\n\n{cleaned_query}"
        
        response = await client.chat.completions.create(
            model=self.valves.text_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
            timeout=self.valves.request_timeout
        )
        
        if response.choices and response.choices[0].message.content:
            keywords_text = response.choices[0].message.content.strip()
            # 基本清理关键字
            keywords_text = keywords_text.replace("\n", " ").replace("\r", " ")
            keywords_text = re.sub(r"\s+", " ", keywords_text).strip()
            
            # 解析关键字
            keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
            # 过滤太短的关键字
            keywords = [kw for kw in keywords if len(kw) >= 2]
            
            self.debug_log(2, f"生成关键字: {keywords[:5]}...", "🔑")
            return keywords
        
        return None

    async def generate_search_keywords(self, query_text: str, event_emitter) -> List[str]:
        """生成搜索关键字"""
        if not self.valves.enable_keyword_generation:
            return [query_text]
        
        # 使用AI检测是否需要上下文最大化
        need_context_max = await self.detect_context_max_need(query_text, event_emitter)
        
        # 如果不需要上下文最大化且不强制生成关键字，直接返回原文
        if not need_context_max and not self.valves.keyword_generation_for_context_max:
            self.debug_log(2, f"具体查询，使用原始文本: {query_text}", "🔑")
            return [query_text]
        
        self.debug_log(1, f"🔑 生成搜索关键字: {query_text}", "🔑")
        
        # 调用关键字生成API
        keywords = await self.safe_api_call(
            self._generate_keywords_impl, "关键字生成", query_text, event_emitter
        )
        
        if keywords:
            # 添加原始查询作为备选
            final_keywords = [query_text] + keywords
            # 去重
            final_keywords = list(dict.fromkeys(final_keywords))
            
            self.stats.keyword_generations += 1
            self.debug_log(1, f"🔑 关键字生成完成: {len(final_keywords)}个", "🔑")
            return final_keywords
        else:
            self.debug_log(1, f"🔑 关键字生成失败，使用原始查询", "⚠️")
            return [query_text]

    # ========== 公用的智能分片功能 ==========
    
    async def smart_chunk_and_summarize_messages(
        self, messages: List[dict], target_tokens: int, progress: ProgressTracker, purpose: str = "处理"
    ) -> List[dict]:
        """公用的智能分片和摘要功能 - 供RAG和上下文最大化共用"""
        if not messages:
            return messages
        
        current_tokens = self.count_messages_tokens(messages)
        self.debug_log(1, f"🧩 开始智能分片和摘要({purpose}): {len(messages)}条消息({current_tokens:,}tokens) -> 目标{target_tokens:,}tokens", "🧩")
        
        # 如果当前token数量已经合适，直接返回
        if current_tokens <= target_tokens:
            self.debug_log(1, f"🧩 消息已符合目标大小，无需分片", "🧩")
            return messages
        
        # 1. 智能分片处理
        await progress.update_progress(0, 2, f"智能分片处理({purpose})")
        chunked_messages = []
        
        if self.valves.enable_smart_chunking:
            # 分离多模态和文本消息
            multimodal_messages, text_messages = self.extract_multimodal_messages(messages)
            
            # 多模态消息直接保留
            chunked_messages.extend(multimodal_messages)
            
            # 文本消息进行分片
            for msg in text_messages:
                msg_tokens = self.count_message_tokens(msg)
                if msg_tokens > self.valves.large_message_threshold:
                    # 大消息需要分片
                    text_content = self.extract_text_from_content(msg.get("content", ""))
                    
                    # 智能分片
                    chunks = self._split_text_smart(text_content, self.valves.chunk_target_tokens)
                    
                    # 创建分片消息
                    for i, chunk in enumerate(chunks):
                        chunk_msg = copy.deepcopy(msg)
                        chunk_msg["content"] = chunk
                        chunk_msg["_chunk_id"] = f"chunk_{i}"
                        chunk_msg["_is_chunk"] = True
                        chunk_msg["_original_message_id"] = msg.get("_order_id")
                        chunked_messages.append(chunk_msg)
                    
                    self.debug_log(2, f"🧩 大消息分片: {msg_tokens}tokens -> {len(chunks)}个分片", "🧩")
                    
                    # 更新统计
                    self.stats.chunk_created += len(chunks)
                    self.stats.chunked_messages += 1
                else:
                    # 小消息直接保留
                    chunked_messages.append(msg)
        else:
            # 不启用分片，直接使用原始消息
            chunked_messages = messages
        
        # 2. 递归摘要处理
        await progress.update_progress(1, 2, f"递归摘要处理({purpose})")
        if self.valves.enable_recursive_summarization:
            chunked_tokens = self.count_messages_tokens(chunked_messages)
            if chunked_tokens > target_tokens:
                self.debug_log(1, f"🔄 开始递归摘要: {len(chunked_messages)}条消息({chunked_tokens:,}tokens) -> 目标{target_tokens:,}tokens", "🔄")
                
                summarized_messages = await self.recursive_summarize_messages(
                    chunked_messages, target_tokens, progress
                )
                
                self.debug_log(1, f"🔄 递归摘要完成: {len(chunked_messages)} -> {len(summarized_messages)}条", "🔄")
                return summarized_messages
        
        return chunked_messages

    def _split_text_smart(self, text: str, target_tokens: int) -> List[str]:
        """智能文本分片"""
        if not text:
            return []
        
        # 计算文本tokens
        text_tokens = self.count_tokens(text)
        
        # 如果文本已经足够小，直接返回
        if text_tokens <= target_tokens:
            return [text]
        
        # 智能分片
        chunks = []
        
        # 首先按段落分割
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            current_tokens = self.count_tokens(current_chunk)
            
            if current_tokens + paragraph_tokens <= target_tokens:
                # 可以添加到当前分片
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 当前分片已满，保存并开始新分片
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落就超过目标大小，需要进一步分割
                if paragraph_tokens > target_tokens:
                    sub_chunks = self._split_paragraph(paragraph, target_tokens)
                    chunks.extend(sub_chunks[:-1])  # 除了最后一个
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = paragraph
        
        # 添加最后一个分片
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]

    def _split_paragraph(self, paragraph: str, target_tokens: int) -> List[str]:
        """分割长段落"""
        sentences = re.split(r"[.!?。！？]+\s*", paragraph)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence_tokens = self.count_tokens(sentence)
            current_tokens = self.count_tokens(current_chunk)
            
            if current_tokens + sentence_tokens <= target_tokens:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [paragraph]

    # ========== 递归摘要处理 ==========
    
    async def _summarize_messages_impl(self, messages_text: str, summary_target: int, event_emitter, has_multimodal: bool = False):
        """实际的消息摘要实现"""
        client = self.get_api_client()
        if not client:
            return None
        
        # 基本清理文本
        cleaned_text = messages_text.replace("\n", " ").replace("\r", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        
        prompt = f"""请对以下对话内容进行智能摘要，保留关键信息、重要细节和核心内容。摘要应该：
1. 保持逻辑连贯性和时间顺序
2. 重点保留技术细节、代码、配置、数据等关键信息
3. 保留用户问题和重要回答
4. 保持原文的专业术语和关键词
5. 控制长度在{summary_target}字符以内

对话内容：
{cleaned_text}

请生成摘要："""
        
        # 根据是否有多模态内容选择模型
        model_to_use = self.valves.multimodal_model if has_multimodal else self.valves.text_model
        
        response = await client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(8000, summary_target // 2),
            temperature=0.3,
            timeout=self.valves.request_timeout
        )
        
        if response.choices and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            # 基本清理摘要
            summary = summary.replace("\n", " ").replace("\r", " ")
            summary = re.sub(r"\s+", " ", summary).strip()
            return summary
        
        return None

    async def recursive_summarize_messages(
        self, messages: List[dict], target_tokens: int, progress: ProgressTracker, depth: int = 0
    ) -> List[dict]:
        """递归摘要处理超大消息集合"""
        if not self.valves.enable_recursive_summarization or depth >= self.valves.max_recursion_depth:
            return messages
        
        current_tokens = self.count_messages_tokens(messages)
        
        # 如果当前token数量合适，直接返回
        if current_tokens <= target_tokens:
            return messages
        
        self.debug_log(1, f"🔄 开始递归摘要(深度{depth+1}): {current_tokens:,} -> {target_tokens:,}tokens", "🔄")
        
        # 分批处理消息
        batch_size = max(5, len(messages) // 4)  # 每批至少5条消息
        batches = [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]
        
        summarized_messages = []
        
        for i, batch in enumerate(batches):
            # 检查批次是否包含多模态内容
            has_multimodal = any(self.has_images_in_content(msg.get("content", "")) for msg in batch)
            
            # 合并批次消息为文本
            batch_text = ""
            for msg in batch:
                role = msg.get("role", "")
                content = self.extract_text_from_content(msg.get("content", ""))
                batch_text += f"[{role}] {content}\n\n"
            
            # 计算摘要目标长度
            batch_tokens = self.count_tokens(batch_text)
            summary_target = int(batch_tokens * self.valves.summary_compression_ratio)
            
            # 调用摘要API
            summary = await self.safe_api_call(
                self._summarize_messages_impl, "消息摘要", 
                batch_text, summary_target, progress.event_emitter, has_multimodal
            )
            
            if summary:
                # 创建摘要消息
                summary_msg = {
                    "role": "assistant",
                    "content": f"[摘要] {summary}",
                    "_is_summary": True,
                    "_original_count": len(batch),
                    "_summary_depth": depth + 1
                }
                summarized_messages.append(summary_msg)
                
                self.debug_log(2, f"🔄 批次摘要完成: {len(batch)}条 -> 1条摘要({self.count_message_tokens(summary_msg)}tokens)", "🔄")
            else:
                # 摘要失败，保留原始消息（但可能需要截断）
                summarized_messages.extend(batch[:3])  # 只保留前3条
        
        # 更新统计
        self.stats.recursive_summaries += 1
        self.stats.summarized_messages += len(messages) - len(summarized_messages)
        
        # 检查是否需要进一步递归
        final_tokens = self.count_messages_tokens(summarized_messages)
        if final_tokens > target_tokens and depth < self.valves.max_recursion_depth - 1:
            summarized_messages = await self.recursive_summarize_messages(
                summarized_messages, target_tokens, progress, depth + 1
            )
        
        self.debug_log(1, f"🔄 递归摘要完成(深度{depth+1}): {len(messages)}条 -> {len(summarized_messages)}条", "✅")
        return summarized_messages

    # ========== 容错保护机制 ==========
    
    def apply_fallback_preservation(self, history_messages: List[dict], available_tokens: int) -> List[dict]:
        """应用容错保护机制，确保即使RAG搜索无结果也有足够的历史上下文"""
        if not self.valves.enable_fallback_preservation or not history_messages:
            return history_messages
        
        # 计算容错保护预算
        fallback_budget = int(available_tokens * self.valves.fallback_preserve_ratio)
        
        self.debug_log(1, f"🛡️ 应用容错保护机制: 预算{fallback_budget:,}tokens", "🛡️")
        
        # 从最近的消息开始保留
        fallback_messages = []
        used_tokens = 0
        
        # 确保至少保留几条最近的用户-助手对话
        user_exchange_count = 0
        target_exchanges = self.valves.force_preserve_recent_user_exchanges
        
        for msg in reversed(history_messages):
            msg_tokens = self.count_message_tokens(msg)
            
            # 检查是否超出预算
            if used_tokens + msg_tokens > fallback_budget:
                # 如果还没有达到最小要求，强制保留
                if (len(fallback_messages) < self.valves.min_history_messages or 
                    user_exchange_count < target_exchanges):
                    # 强制保留，但截断过长的消息
                    if msg_tokens > fallback_budget // 4:
                        # 消息太长，截断
                        content = self.extract_text_from_content(msg.get("content", ""))
                        if content:
                            # 保留前半部分
                            truncated_content = content[:len(content)//2] + "...[截断]"
                            truncated_msg = copy.deepcopy(msg)
                            truncated_msg["content"] = truncated_content
                            truncated_msg["_is_truncated"] = True
                            fallback_messages.insert(0, truncated_msg)
                            used_tokens += self.count_message_tokens(truncated_msg)
                    else:
                        fallback_messages.insert(0, msg)
                        used_tokens += msg_tokens
                else:
                    break
            else:
                fallback_messages.insert(0, msg)
                used_tokens += msg_tokens
            
            # 统计用户对话轮次
            if msg.get("role") == "user":
                user_exchange_count += 1
        
        # 更新统计
        self.stats.fallback_preserve_applied += 1
        
        self.debug_log(1, f"🛡️ 容错保护完成: 保留{len(fallback_messages)}条消息({used_tokens:,}tokens), "
                          f"{user_exchange_count}个用户对话轮次", "🛡️")
        
        return fallback_messages

    def ensure_current_user_message_preserved(self, final_messages: List[dict]) -> List[dict]:
        """确保当前用户消息被正确保留在最后位置"""
        if not self.current_user_message:
            return final_messages
        
        # 检查当前用户消息是否在最后位置
        if final_messages and final_messages[-1].get("role") == "user":
            current_id = self.current_user_message.get("_order_id")
            last_id = final_messages[-1].get("_order_id")
            
            if current_id == last_id:
                # 当前用户消息已经在正确位置
                return final_messages
        
        # 当前用户消息不在最后位置，需要修复
        self.debug_log(1, "🛡️ 检测到当前用户消息位置错误，开始修复", "🛡️")
        
        # 移除所有当前用户消息的副本
        current_id = self.current_user_message.get("_order_id")
        filtered_messages = []
        
        for msg in final_messages:
            if msg.get("_order_id") != current_id:
                filtered_messages.append(msg)
        
        # 将当前用户消息添加到最后
        filtered_messages.append(self.current_user_message)
        
        # 更新统计
        self.stats.user_message_recovery_count += 1
        
        self.debug_log(1, "🛡️ 当前用户消息位置修复完成", "🛡️")
        return filtered_messages

    # ========== 上下文最大化专用处理策略 - 修复处理流程 ==========
    
    async def process_context_maximization(
        self, history_messages: List[dict], available_tokens: int, progress: ProgressTracker, need_context_max: bool = True
    ) -> List[dict]:
        """上下文最大化处理策略 - 修复处理流程，增强容错机制"""
        if not self.valves.enable_context_maximization or not need_context_max:
            return history_messages
        
        await progress.start_phase("上下文最大化处理", len(history_messages))
        
        self.debug_log(1, f"📚 上下文最大化处理: {len(history_messages)}条消息, 可用预算: {available_tokens:,}tokens", "📚")
        
        # 1. 先提取多模态消息，保留给多模态模型处理
        multimodal_messages, text_messages = self.extract_multimodal_messages(history_messages)
        
        # 更新统计
        self.stats.multimodal_extracted = len(multimodal_messages)
        
        self.debug_log(2, f"📚 消息分类: {len(multimodal_messages)}条多模态, {len(text_messages)}条文本", "📚")
        
        # 2. 计算预算分配
        direct_preserve_budget = int(available_tokens * self.valves.context_max_direct_preserve_ratio)
        # 为容错保护预留预算
        fallback_budget = int(available_tokens * self.valves.fallback_preserve_ratio)
        processing_budget = available_tokens - direct_preserve_budget - fallback_budget
        
        self.debug_log(1, f"💰 预算分配: 直接保留 {direct_preserve_budget:,}tokens, "
                          f"处理剩余 {processing_budget:,}tokens, 容错保护 {fallback_budget:,}tokens", "💰")
        
        # 3. 优先保留多模态消息和最近的文本消息
        preserved_messages = []
        used_tokens = 0
        
        # 先保留多模态消息（优先最新的）
        for msg in reversed(multimodal_messages):
            msg_tokens = self.count_message_tokens(msg)
            if used_tokens + msg_tokens <= direct_preserve_budget:
                preserved_messages.insert(0, msg)
                used_tokens += msg_tokens
                self.debug_log(3, f"📚 保留多模态消息: {msg_tokens}tokens, ID: {msg.get('_order_id', 'None')[:8]}", "📚")
            else:
                break
        
        # 再保留最近的文本消息
        for msg in reversed(text_messages):
            msg_tokens = self.count_message_tokens(msg)
            if used_tokens + msg_tokens <= direct_preserve_budget:
                preserved_messages.insert(0, msg)
                used_tokens += msg_tokens
                self.debug_log(3, f"📚 保留文本消息: {msg_tokens}tokens, ID: {msg.get('_order_id', 'None')[:8]}", "📚")
            else:
                break
        
        self.stats.context_max_direct_preserve = len(preserved_messages)
        self.debug_log(1, f"📚 直接保留完成: {len(preserved_messages)}条消息({used_tokens:,}tokens)", "📚")
        
        # 4. 处理剩余消息 - 修复：确保强制处理所有剩余消息
        preserved_ids = {msg.get("_order_id") for msg in preserved_messages if msg.get("_order_id")}
        self.debug_log(2, f"📚 已保留消息IDs: {[msg_id[:8] for msg_id in preserved_ids]}", "📚")
        
        remaining_messages = []
        for msg in history_messages:
            msg_id = msg.get("_order_id")
            if msg_id not in preserved_ids:
                remaining_messages.append(msg)
            else:
                self.debug_log(3, f"📚 跳过已保留消息: ID {msg_id[:8] if msg_id else 'None'}", "📚")
        
        self.debug_log(1, f"📚 剩余消息: {len(remaining_messages)}条，处理预算: {processing_budget:,}tokens", "📚")
        
        processed_remaining = []
        
        # 强制处理剩余消息，确保最大化保留上下文
        if remaining_messages and processing_budget > 5000:
            self.debug_log(1, f"📚 开始处理剩余消息: {len(remaining_messages)}条 ({self.count_messages_tokens(remaining_messages):,}tokens)", "📚")
            
            # 使用公用的分片摘要功能
            processed_remaining = await self.smart_chunk_and_summarize_messages(
                remaining_messages, processing_budget, progress, "上下文最大化"
            )
            
            # 更新统计
            self.stats.context_max_chunked = self.stats.chunk_created
            self.stats.context_max_summarized = self.stats.summarized_messages
            
            self.debug_log(1, f"📚 剩余消息处理完成: {len(remaining_messages)} -> {len(processed_remaining)}条 "
                              f"({self.count_messages_tokens(processed_remaining):,}tokens)", "📚")
        else:
            self.debug_log(1, f"📚 处理预算不足或无剩余消息，跳过处理", "📚")
        
        # 5. 合并结果
        preliminary_messages = preserved_messages + processed_remaining
        
        # 6. 应用容错保护机制 - 如果处理结果太少，使用容错保护
        if len(preliminary_messages) < self.valves.min_history_messages:
            self.debug_log(1, f"🛡️ 处理结果过少({len(preliminary_messages)}条)，应用容错保护", "🛡️")
            
            # 使用容错保护机制补充消息
            fallback_messages = self.apply_fallback_preservation(
                remaining_messages, fallback_budget
            )
            
            # 合并去重
            all_message_ids = {msg.get("_order_id") for msg in preliminary_messages if msg.get("_order_id")}
            for msg in fallback_messages:
                if msg.get("_order_id") not in all_message_ids:
                    preliminary_messages.append(msg)
                    all_message_ids.add(msg.get("_order_id"))
        
        # 7. 确保消息顺序正确
        final_messages = preliminary_messages
        if self.message_order:
            final_messages = self.message_order.sort_messages_preserve_user(
                final_messages, self.current_user_message
            )
        
        final_tokens = self.count_messages_tokens(final_messages)
        
        self.debug_log(1, f"📚 上下文最大化处理完成: {len(history_messages)} -> {len(final_messages)}条消息({final_tokens:,}tokens)", "✅")
        
        await progress.complete_phase(f"处理完成 {len(final_messages)}条消息({final_tokens:,}tokens)")
        
        return final_messages

    # ========== 尽量保留机制 ==========
    
    async def try_preserve_recent_messages(
        self, history_messages: List[dict], available_tokens: int, progress: ProgressTracker, need_context_max: bool = False
    ) -> Tuple[List[dict], int]:
        """尽量保留最近的消息"""
        if not self.valves.enable_try_preserve or not history_messages:
            return [], available_tokens
        
        # 对于上下文最大化，调整保留比例
        if need_context_max and self.valves.enable_context_maximization:
            # 上下文最大化时，减少尽量保留的比例，为后续直接保留留出空间
            try_preserve_ratio = min(self.valves.try_preserve_ratio, 0.25)
        else:
            try_preserve_ratio = self.valves.try_preserve_ratio
        
        # 计算尽量保留的预算
        try_preserve_budget = int(available_tokens * try_preserve_ratio)
        
        await progress.start_phase("尽量保留机制", len(history_messages))
        
        self.debug_log(1, f"🔒 尽量保留预算: {try_preserve_budget:,}tokens ({try_preserve_ratio:.1%}) "
                          f"{'[上下文最大化优化]' if need_context_max else ''}", "🔒")
        
        preserved_messages = []
        used_tokens = 0
        
        # 从后往前遍历历史消息，寻找完整的对话轮次
        i = len(history_messages) - 1
        preserved_exchanges = 0
        
        while i >= 0 and preserved_exchanges < self.valves.try_preserve_exchanges:
            msg = history_messages[i]
            msg_tokens = self.count_message_tokens(msg)
            
            # 检查是否是用户消息（开始一个新的交换）
            if msg.get("role") == "user":
                # 查找这个用户消息对应的助手回复
                user_msg = msg
                assistant_msg = None
                
                # 查找下一条助手消息
                if i + 1 < len(history_messages) and history_messages[i + 1].get("role") == "assistant":
                    assistant_msg = history_messages[i + 1]
                
                # 计算这个对话轮次的总token数
                exchange_tokens = msg_tokens
                if assistant_msg:
                    exchange_tokens += self.count_message_tokens(assistant_msg)
                
                # 检查是否能完整保留这个对话轮次
                if used_tokens + exchange_tokens <= try_preserve_budget:
                    # 完整保留
                    preserved_messages.insert(0, user_msg)
                    used_tokens += msg_tokens
                    
                    if assistant_msg:
                        preserved_messages.insert(1, assistant_msg)
                        used_tokens += self.count_message_tokens(assistant_msg)
                    
                    preserved_exchanges += 1
                    
                    self.debug_log(2, f"🔒 完整保留对话轮次{preserved_exchanges}: {exchange_tokens}tokens", "🔒")
                    
                    # 跳过已处理的助手消息
                    if assistant_msg:
                        i -= 1
                else:
                    # 无法完整保留，跳出循环
                    break
            else:
                # 不是用户消息，单独处理
                if used_tokens + msg_tokens <= try_preserve_budget:
                    preserved_messages.insert(0, msg)
                    used_tokens += msg_tokens
                    self.debug_log(3, f"🔒 单独保留消息: {msg_tokens}tokens", "🔒")
                else:
                    # 无法保留，跳出循环
                    break
            
            i -= 1
        
        # 更新统计
        self.stats.try_preserve_messages = len(preserved_messages)
        self.stats.try_preserve_tokens = used_tokens
        
        # 计算剩余预算
        remaining_budget = available_tokens - used_tokens
        
        self.debug_log(1, f"🔒 尽量保留完成: {len(preserved_messages)}条消息({used_tokens:,}tokens), "
                          f"剩余预算: {remaining_budget:,}tokens", "🔒")
        
        await progress.complete_phase(f"保留{len(preserved_messages)}条消息({used_tokens:,}tokens)")
        
        return preserved_messages, remaining_budget

    # ========== 视觉处理 ==========
    
    async def _describe_image_impl(self, image_, event_emitter):
        """实际的图片描述实现"""
        client = self.get_api_client()
        if not client:
            return None
        
        # 确保图片数据是base64格式
        if not image_data.startswith("data:"):
            self.debug_log(1, "图片数据不是base64格式", "⚠️")
            return None
        
        response = await client.chat.completions.create(
            model=self.valves.multimodal_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.valves.vision_prompt_template},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }],
            max_tokens=self.valves.vision_max_tokens,
            temperature=0.2
        )
        
        if response.choices:
            description = response.choices[0].message.content.strip()
            # 基本清理描述
            description = description.replace("\n", " ").replace("\r", " ")
            description = re.sub(r"\s+", " ", description).strip()
            return description
        
        return None

    async def describe_image(self, image_data: str, event_emitter) -> str:
        """描述单张图片"""
        image_hash = hashlib.md5(image_data.encode()).hexdigest()
        self.debug_log(2, f"开始识别图片: {image_hash[:8]}", "👁️")
        
        description = await self.safe_api_call(
            self._describe_image_impl, "图片识别", image_data, event_emitter
        )
        
        if description:
            # 提高描述长度限制
            if len(description) > 2500:
                description = description[:2500] + "..."
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
                # 直接使用原始文本
                text = item.get("text", "")
                processed_content.append(text)
            elif item.get("type") == "image_url":
                image_count += 1
                image_data = item.get("image_url", {}).get("url", "")
                if image_data:
                    if progress:
                        await progress.update_progress(
                            image_count, len(images), f"处理图片 {image_count}/{len(images)}"
                        )
                    description = await self.describe_image(
                        image_data, progress.event_emitter if progress else None
                    )
                    processed_content.append(f"[图片{image_count}描述] {description}")
        
        # 创建新消息 - 保持原始顺序信息
        processed_message = copy.deepcopy(message)
        processed_message["content"] = "\n".join(processed_content) if processed_content else ""
        
        self.debug_log(2, f"消息图片处理完成: {image_count}张图片", "🖼️")
        
        # 更新统计
        self.stats.multimodal_processed += image_count
        
        return processed_message

    # ========== 向量化处理 ==========
    
    async def _get_text_embedding_impl(self, text: str, event_emitter):
        """实际的文本向量获取实现"""
        client = self.get_api_client()
        if not client:
            return None
        
        # 基本清理文本
        cleaned_text = text.replace("\n", " ").replace("\r", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        
        response = await client.embeddings.create(
            model=self.valves.text_vector_model,
            input=[cleaned_text[:8000]],
            encoding_format="float"
        )
        
        if response and response.data and len(response.data) > 0 and response.data[0].embedding:
            return response.data[0].embedding
        
        return None

    async def get_text_embedding(self, text: str, event_emitter) -> Optional[List[float]]:
        """获取文本向量"""
        if not text:
            return None
        
        # 基本清理文本
        cleaned_text = text.replace("\n", " ").replace("\r", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        
        embedding = await self.safe_api_call(
            self._get_text_embedding_impl, "文本向量", cleaned_text, event_emitter
        )
        
        if embedding:
            self.debug_log(3, f"文本向量获取成功: {len(embedding)}维", "📝")
        
        return embedding

    async def _get_multimodal_embedding_impl(self, content, event_emitter):
        """实际的多模态向量获取实现"""
        client = self.get_api_client()
        if not client:
            return None
        
        # 处理输入格式，确保符合API要求
        if isinstance(content, list):
            # 已经是列表格式，清理文本内容
            cleaned_content = []
            for item in content:
                if item.get("type") == "text":
                    cleaned_item = item.copy()
                    # 基本清理文本
                    text = item.get("text", "")
                    cleaned_text = text.replace("\n", " ").replace("\r", " ")
                    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
                    cleaned_item["text"] = cleaned_text
                    cleaned_content.append(cleaned_item)
                elif item.get("type") == "image_url":
                    # 图片内容保持不变，base64编码
                    cleaned_content.append(item)
                else:
                    # 其他类型也保持不变
                    cleaned_content.append(item)
            input_data = cleaned_content
        else:
            # 纯文本内容，转换为列表格式
            text = str(content)
            cleaned_text = text.replace("\n", " ").replace("\r", " ")
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            input_data = [{"type": "text", "text": cleaned_text[:8000]}]
        
        self.debug_log(3, f"多模态向量输入: {len(input_data)}个元素", "🖼️")
        
        # 使用正确的API调用方式
        try:
            response = await client.embeddings.create(
                model=self.valves.multimodal_vector_model,
                input=input_data,
                encoding_format="float"
            )
            
            # 处理响应
            if hasattr(response, "data") and hasattr(response.data, "embedding"):
                return response.data.embedding
            elif hasattr(response, "data") and isinstance(response.data, list) and len(response.data) > 0:
                return response.data[0].embedding
            else:
                self.debug_log(1, f"多模态向量响应格式异常: {type(response.data)}", "⚠️")
                return None
        except Exception as e:
            self.debug_log(1, f"多模态向量调用失败: {e}", "❌")
            raise

    async def get_multimodal_embedding(self, content, event_emitter) -> Optional[List[float]]:
        """获取多模态向量"""
        if not content:
            return None
        
        # 检查是否为多模态内容
        has_multimodal_content = False
        if isinstance(content, list):
            has_multimodal_content = any(item.get("type") in ["image_url", "video_url"] for item in content)
        
        if not has_multimodal_content:
            self.debug_log(3, "内容不包含多模态元素，不使用多模态向量", "📝")
            return None
        
        embedding = await self.safe_api_call(
            self._get_multimodal_embedding_impl, "多模态向量", content, event_emitter
        )
        
        if embedding:
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

    # ========== 向量检索 ==========
    
    async def vector_retrieve_relevant_messages(
        self, query_message: dict, candidate_messages: List[dict], progress: ProgressTracker
    ) -> List[dict]:
        """基于向量相似度检索相关消息 - 增强容错机制"""
        if not candidate_messages or not self.valves.enable_vector_retrieval:
            return candidate_messages
        
        await progress.start_phase("智能向量检索", len(candidate_messages))
        
        self.debug_log(1, f"开始向量检索: 查询1条，候选{len(candidate_messages)}条", "🔍")
        
        # 更新统计
        self.stats.vector_retrievals += 1
        
        # 获取查询内容
        query_content = query_message.get("content", "")
        query_text = self.extract_text_from_content(query_content)
        
        # 生成搜索关键字
        await progress.update_progress(0, len(candidate_messages), "生成搜索关键字")
        search_keywords = await self.generate_search_keywords(query_text, progress.event_emitter)
        
        self.debug_log(1, f"🔑 搜索关键字({len(search_keywords)}个): {search_keywords[:3]}...", "🔑")
        
        # 对每个关键字进行向量检索
        all_similarities = []
        
        for keyword_idx, keyword in enumerate(search_keywords):
            await progress.update_progress(
                keyword_idx * len(candidate_messages) // len(search_keywords),
                len(candidate_messages),
                f"关键字{keyword_idx+1}/{len(search_keywords)}: {keyword[:20]}..."
            )
            
            # 获取关键字向量
            keyword_vector = None
            
            # 智能向量化策略
            if self.has_images_in_content(query_content):
                # 多模态内容：优先使用多模态向量
                keyword_vector = await self.get_multimodal_embedding(query_content, progress.event_emitter)
                self.debug_log(3, f"关键字使用多模态向量: {'成功' if keyword_vector else '失败'}", "🖼️")
                
                # 如果多模态向量失败，转换为文本向量
                if not keyword_vector:
                    keyword_vector = await self.get_text_embedding(keyword, progress.event_emitter)
                    self.debug_log(3, f"关键字使用文本向量: {'成功' if keyword_vector else '失败'}", "📝")
            else:
                # 纯文本内容：使用文本向量
                keyword_vector = await self.get_text_embedding(keyword, progress.event_emitter)
                self.debug_log(3, f"关键字使用文本向量: {'成功' if keyword_vector else '失败'}", "📝")
            
            if not keyword_vector:
                continue
            
            # 计算与候选消息的相似度
            for msg_idx, msg in enumerate(candidate_messages):
                msg_content = msg.get("content", "")
                msg_vector = None
                
                # 为候选消息获取向量
                if self.has_images_in_content(msg_content):
                    # 多模态内容：优先使用多模态向量
                    msg_vector = await self.get_multimodal_embedding(msg_content, progress.event_emitter)
                    if not msg_vector:
                        # 多模态向量失败，转换为文本向量
                        text_content = self.extract_text_from_content(msg_content)
                        if text_content:
                            msg_vector = await self.get_text_embedding(text_content, progress.event_emitter)
                else:
                    # 纯文本内容：使用文本向量
                    text_content = self.extract_text_from_content(msg_content)
                    if text_content:
                        msg_vector = await self.get_text_embedding(text_content, progress.event_emitter)
                
                if msg_vector:
                    similarity = self.cosine_similarity(keyword_vector, msg_vector)
                    
                    # 根据关键字权重调整相似度
                    if keyword_idx == 0:  # 原始查询权重更高
                        similarity *= 1.2
                    
                    # 高优先级内容给予加权
                    if self.is_high_priority_content(self.extract_text_from_content(msg_content)):
                        similarity = min(1.0, similarity * 1.25)
                    
                    # 给最近的消息额外加权
                    if msg_idx >= len(candidate_messages) - self.valves.preserve_recent_exchanges * 2:
                        similarity = min(1.0, similarity * 1.15)
                    
                    all_similarities.append((msg_idx, similarity, msg, keyword_idx))
        
        if not all_similarities:
            self.debug_log(1, "向量检索失败，返回原始消息", "⚠️")
            # 更新RAG无结果统计
            self.stats.rag_no_results_count += 1
            await progress.complete_phase("向量检索失败")
            return candidate_messages
        
        # 按消息分组，取最高相似度
        msg_best_similarity = {}
        for msg_idx, similarity, msg, keyword_idx in all_similarities:
            if msg_idx not in msg_best_similarity or similarity > msg_best_similarity[msg_idx][1]:
                msg_best_similarity[msg_idx] = (msg_idx, similarity, msg)
        
        # 按相似度排序
        similarities = list(msg_best_similarity.values())
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 使用更宽松的阈值过滤
        threshold = self.valves.vector_similarity_threshold
        filtered_similarities = [item for item in similarities if item[1] >= threshold]
        
        # 如果过滤后太少，降低阈值确保保留足够消息 - 增强容错机制
        min_keep_ratio = 0.4  # 至少保留40%的消息
        min_keep = max(20, int(len(candidate_messages) * min_keep_ratio))
        
        if len(filtered_similarities) < min_keep:
            lower_threshold = max(0.02, threshold - 0.04)
            filtered_similarities = [item for item in similarities if item[1] >= lower_threshold]
            self.debug_log(2, f"降低阈值到{lower_threshold:.3f}，保留更多消息", "🔍")
        
        # 确保至少保留一定数量的消息
        if len(filtered_similarities) < min_keep:
            filtered_similarities = similarities[:min_keep]
            self.debug_log(2, f"强制保留{min_keep}条消息，确保不丢失数据", "🔍")
        
        # 如果仍然没有足够的结果，记录统计
        if len(filtered_similarities) < self.valves.min_history_messages:
            self.stats.rag_no_results_count += 1
            self.debug_log(1, f"🔍 RAG检索结果过少: {len(filtered_similarities)}条", "⚠️")
            
            # 应用容错保护机制
            if self.valves.enable_fallback_preservation:
                fallback_messages = self.apply_fallback_preservation(
                    candidate_messages, self.count_messages_tokens(candidate_messages)
                )
                self.debug_log(1, f"🛡️ 应用容错保护，补充到{len(fallback_messages)}条消息", "🛡️")
                
                # 合并检索结果和容错保护结果
                result_message_ids = {similarities[i][0] for i in range(len(filtered_similarities))}
                for i, msg in enumerate(fallback_messages):
                    if i not in result_message_ids:
                        filtered_similarities.append((i, 0.5, msg))  # 给容错保护的消息一个中等相似度
        
        # 限制数量
        top_similarities = filtered_similarities[:self.valves.vector_top_k]
        
        # 提取消息并保持原始顺序
        relevant_messages = []
        selected_indices = sorted([item[0] for item in top_similarities])
        
        for idx in selected_indices:
            if idx < len(candidate_messages):  # 防止索引越界
                relevant_messages.append(candidate_messages[idx])
        
        # 使用消息顺序管理器确保顺序正确，但不包括当前用户消息
        if self.message_order:
            relevant_messages = self.message_order.sort_messages_preserve_user(
                relevant_messages, self.current_user_message
            )
        
        self.debug_log(1, f"🔍 智能向量检索完成: {len(candidate_messages)} -> {len(relevant_messages)}条", "✅")
        
        await progress.complete_phase(f"检索到{len(relevant_messages)}条相关消息")
        
        return relevant_messages

    # ========== 重排序 ==========
    
    async def _rerank_messages_impl(self, query_text: str, documents: List[str], event_emitter):
        """实际的重排序实现"""
        if not HTTPX_AVAILABLE:
            return None
        
        # 基本清理查询文本和文档
        cleaned_query_text = query_text.replace("\n", " ").replace("\r", " ")
        cleaned_query_text = re.sub(r"\s+", " ", cleaned_query_text).strip()
        
        cleaned_documents = []
        for doc in documents:
            cleaned_doc = doc.replace("\n", " ").replace("\r", " ")
            cleaned_doc = re.sub(r"\s+", " ", cleaned_doc).strip()
            cleaned_documents.append(cleaned_doc)
        
        async with httpx.AsyncClient(timeout=self.valves.request_timeout) as client:
            headers = {
                "Authorization": f"Bearer {self.valves.rerank_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.valves.rerank_model,
                "query": cleaned_query_text,
                "documents": cleaned_documents,
                "top_n": min(self.valves.rerank_top_k, len(cleaned_documents)),
                "return_documents": True
            }
            
            response = await client.post(
                f"{self.valves.rerank_api_base}/v1/rerank",
                headers=headers,
                json=data
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
                if len(text) > 12000:
                    text = text[:12000] + "..."
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
                    self.debug_log(3, f"重排序结果: index={original_index}, score={score:.3f}", "📊")
            
            # 使用消息顺序管理器确保顺序正确，但不包括当前用户消息
            if self.message_order:
                reranked_messages = self.message_order.sort_messages_preserve_user(
                    reranked_messages, self.current_user_message
                )
            
            self.debug_log(1, f"🔄 重排序完成: {len(candidate_messages)} -> {len(reranked_messages)}条", "✅")
            
            await progress.complete_phase(f"重排序到{len(reranked_messages)}条消息")
            return reranked_messages
        
        await progress.complete_phase("重排序失败")
        return candidate_messages

    # ========== 内容最大化核心处理逻辑 ==========
    
    def should_force_maximize_content(self, messages: List[dict], target_tokens: int) -> bool:
        """判断是否应该强制进行内容最大化处理"""
        current_tokens = self.count_messages_tokens(messages)
        utilization = current_tokens / target_tokens if target_tokens > 0 else 0
        
        # 如果当前利用率低于最大利用率，强制最大化
        # 或者如果超过了目标限制，也需要处理
        should_maximize = (
            utilization < self.valves.max_window_utilization 
            or current_tokens > target_tokens
        )
        
        self.debug_log(1, f"🔥 内容最大化判断: {current_tokens:,}tokens / {target_tokens:,}tokens = {utilization:.1%}", "🔥")
        self.debug_log(1, f"🔥 需要最大化: {should_maximize} "
                          f"(利用率{'<' if utilization < self.valves.max_window_utilization else '>='}"
                          f"{self.valves.max_window_utilization:.1%} 或 超限制)", "🔥")
        
        return should_maximize

    async def maximize_content_comprehensive_processing(
        self, messages: List[dict], target_tokens: int, progress: ProgressTracker
    ) -> List[dict]:
        """内容最大化综合处理 - 修复RAG搜索无结果问题，增强容错机制"""
        start_time = time.time()
        
        # 初始化统计
        self.stats.original_tokens = self.count_messages_tokens(messages)
        self.stats.original_messages = len(messages)
        self.stats.token_limit = self.get_model_token_limit("unknown")
        self.stats.target_tokens = target_tokens
        current_tokens = self.stats.original_tokens
        
        # 记录真实的原始数据量
        self.debug_log(1, f"🔥 真实原始数据量: {current_tokens:,} tokens, {len(messages)} 条消息", "🔥")
        
        await progress.start_phase("内容最大化处理", 10)
        
        self.debug_log(1, f"🔥 开始内容最大化处理 [ID:{self.current_processing_id}]: "
                          f"{current_tokens:,} -> {target_tokens:,} tokens", "🔥")
        
        # 1. 使用修复后的消息分离逻辑
        await progress.update_progress(1, 10, "分离当前用户消息和历史消息")
        current_user_message, history_messages = self.separate_current_and_history_messages(messages)
        
        # 保存当前用户消息的引用
        self.current_user_message = current_user_message
        
        # 系统消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        
        # 2. 检测是否需要上下文最大化
        need_context_max = False
        if current_user_message and self.valves.enable_context_maximization:
            query_text = self.extract_text_from_content(current_user_message.get("content", ""))
            need_context_max = await self.detect_context_max_need(query_text, progress.event_emitter)
            if need_context_max:
                self.debug_log(1, f"📚 检测到需要上下文最大化，启用特殊处理策略", "📚")
        
        # 3. 计算保护消息的token
        protected_messages = system_messages
        if current_user_message:
            protected_messages.append(current_user_message)
            self.stats.current_user_tokens = self.count_message_tokens(current_user_message)
        
        protected_tokens = self.count_messages_tokens(protected_messages)
        available_for_processing = target_tokens - protected_tokens
        
        self.debug_log(1, f"🔒 保护消息: {len(protected_messages)}条 ({protected_tokens:,}tokens)", "🔒")
        self.debug_log(1, f"💰 可用处理空间: {available_for_processing:,}tokens", "💰")
        
        # 4. 如果历史消息为空，直接返回
        if not history_messages:
            self.debug_log(1, f"📋 无历史消息，直接返回保护消息", "📋")
            await progress.complete_phase("无历史消息需要处理")
            return protected_messages
        
        # 5. 根据查询类型选择处理策略
        if need_context_max and self.valves.enable_context_maximization:
            # 上下文最大化：使用专门的处理策略
            await progress.update_progress(2, 10, "上下文最大化专用处理")
            processed_history = await self.process_context_maximization(
                history_messages, available_for_processing, progress, need_context_max
            )
            self.debug_log(1, f"📚 上下文最大化处理完成: {len(history_messages)} -> {len(processed_history)}条", "📚")
        else:
            # 具体查询：使用标准的RAG处理策略
            await progress.update_progress(2, 10, "标准RAG处理")
            
            # 尽量保留最近的消息
            try_preserved_messages, remaining_budget = await self.try_preserve_recent_messages(
                history_messages, available_for_processing, progress, need_context_max
            )
            
            # 从历史消息中移除已尽量保留的消息
            if try_preserved_messages:
                preserved_ids = {msg.get("_order_id") for msg in try_preserved_messages}
                remaining_history = [
                    msg for msg in history_messages 
                    if msg.get("_order_id") not in preserved_ids
                ]
            else:
                remaining_history = history_messages
            
            # 智能分片处理大消息 - 使用公用功能
            await progress.update_progress(3, 10, "智能分片处理")
            if remaining_budget > 5000 and remaining_history:
                processed_remaining = await self.smart_chunk_and_summarize_messages(
                    remaining_history, remaining_budget, progress, "RAG"
                )
            else:
                processed_remaining = remaining_history
            
            # RAG处理
            await progress.update_progress(4, 10, "RAG处理")
            if remaining_budget > 2000 and processed_remaining and current_user_message:
                # 向量检索
                if self.valves.enable_vector_retrieval:
                    self.debug_log(1, f"🔍 启动向量检索: {len(processed_remaining)}条候选消息", "🔍")
                    
                    relevant_history = await self.vector_retrieve_relevant_messages(
                        current_user_message, processed_remaining, progress
                    )
                    
                    # 重排序
                    if self.valves.enable_reranking and len(relevant_history) > 5:
                        relevant_history = await self.rerank_messages(
                            current_user_message, relevant_history, progress
                        )
                    
                    processed_remaining = relevant_history
                    self.debug_log(1, f"🔍 RAG处理完成: {len(processed_remaining)}条相关消息", "🔍")
                    
                    # 检查RAG结果是否过少，应用容错保护
                    if len(processed_remaining) < self.valves.min_history_messages:
                        self.debug_log(1, f"🛡️ RAG结果过少({len(processed_remaining)}条)，应用容错保护", "🛡️")
                        
                        # 使用容错保护机制补充消息
                        fallback_messages = self.apply_fallback_preservation(
                            remaining_history, remaining_budget
                        )
                        
                        # 合并去重
                        result_message_ids = {msg.get("_order_id") for msg in processed_remaining if msg.get("_order_id")}
                        for msg in fallback_messages:
                            if msg.get("_order_id") not in result_message_ids:
                                processed_remaining.append(msg)
                                result_message_ids.add(msg.get("_order_id"))
                        
                        self.debug_log(1, f"🛡️ 容错保护后补充到{len(processed_remaining)}条消息", "🛡️")
                else:
                    # 如果不启用向量检索，按优先级排序
                    processed_remaining = self.sort_messages_by_priority(processed_remaining)
            
            # 组合结果
            processed_history = try_preserved_messages + processed_remaining
        
        # 6. 最终预算控制
        await progress.update_progress(6, 10, "最终预算控制")
        final_history = processed_history
        final_tokens = self.count_messages_tokens(final_history)
        
        # 如果仍然超出预算，进行紧急截断
        if final_tokens > available_for_processing:
            truncated_history = []
            used_tokens = 0
            
            for msg in final_history:
                msg_tokens = self.count_message_tokens(msg)
                if used_tokens + msg_tokens <= available_for_processing:
                    truncated_history.append(msg)
                    used_tokens += msg_tokens
                else:
                    break
            
            final_history = truncated_history
            self.stats.emergency_truncations += 1
            
            self.debug_log(1, f"🆘 紧急截断: 保留{len(final_history)}条历史消息({used_tokens:,}tokens)", "🆘")
        
        # 7. 组合最终结果
        await progress.update_progress(8, 10, "组合最终结果")
        
        # 按顺序组合：系统消息 + 处理后的历史消息
        current_result = system_messages + final_history
        
        # 确保消息顺序正确，但不包括当前用户消息
        if self.message_order:
            current_result = self.message_order.sort_messages_preserve_user(
                current_result, self.current_user_message
            )
        
        # 最终组合
        final_messages = []
        
        # 添加系统消息和历史消息
        for msg in current_result:
            final_messages.append(msg)
        
        # 最后添加当前用户消息
        if current_user_message:
            final_messages.append(current_user_message)
        
        # 8. 应用最终的用户消息保护机制
        await progress.update_progress(9, 10, "用户消息保护验证")
        final_messages = self.ensure_current_user_message_preserved(final_messages)
        
        # 9. 更新最终统计
        await progress.update_progress(10, 10, "更新统计")
        self.stats.final_tokens = self.count_messages_tokens(final_messages)
        self.stats.final_messages = len(final_messages)
        self.stats.processing_time = time.time() - start_time
        self.stats.iterations = 1
        
        # 修复统计计算
        if self.stats.original_tokens > 0:
            self.stats.content_loss_ratio = max(
                0, (self.stats.original_tokens - self.stats.final_tokens) / self.stats.original_tokens
            )
        
        if target_tokens > 0:
            self.stats.window_utilization = self.stats.final_tokens / target_tokens
        
        # 检查当前用户消息是否保留
        if current_user_message:
            self.stats.current_user_preserved = any(
                msg.get("_order_id") == current_user_message.get("_order_id")
                for msg in final_messages
            )
        
        # 计算最终指标
        retention_ratio = self.stats.calculate_retention_ratio()
        window_usage = self.stats.calculate_window_usage_ratio()
        
        self.debug_log(1, f"🔥 内容最大化处理完成 [ID:{self.current_processing_id}]: "
                          f"保留{retention_ratio:.1%} 窗口使用{window_usage:.1%}", "🔥")
        
        # 验证当前用户消息是否在最后
        if current_user_message and final_messages:
            last_msg = final_messages[-1]
            if last_msg.get("_order_id") == current_user_message.get("_order_id"):
                self.debug_log(1, f"✅ 当前用户消息保护成功：在最后位置", "✅")
            else:
                self.debug_log(1, f"❌ 当前用户消息保护失败：不在最后位置", "❌")
        
        await progress.update_progress(10, 10, "处理完成")
        await progress.complete_phase(
            f"最大化完成 保留{retention_ratio:.1%} 窗口使用{window_usage:.1%} "
            f"{'[上下文最大化]' if need_context_max else '[具体查询]'}"
        )
        
        return final_messages

    def sort_messages_by_priority(self, messages: List[dict]) -> List[dict]:
        """按优先级排序消息"""
        def get_priority_score(msg):
            content = self.extract_text_from_content(msg.get("content", ""))
            score = 0
            
            # 高优先级关键词加分
            if self.is_high_priority_content(content):
                score += 100
            
            # 长度加分（更长的消息可能包含更多信息）
            score += min(len(content) // 100, 50)
            
            # 角色加分
            if msg.get("role") == "user":
                score += 20
            elif msg.get("role") == "assistant":
                score += 10
            
            # 原始索引加分（越新的消息分数越高）
            original_index = msg.get("_original_index", 0)
            score += original_index * 0.1
            
            return score
        
        return sorted(messages, key=get_priority_score, reverse=True)

    def print_detailed_stats(self):
        """打印详细统计信息"""
        if not self.valves.enable_detailed_stats:
            return
        
        print("\n" + "=" * 70)
        print(self.stats.get_summary())
        print("=" * 70)

    # ========== 多模态处理策略 ==========
    
    async def determine_multimodal_processing_strategy(
        self, messages: List[dict], model_name: str, target_tokens: int
    ) -> Tuple[str, str]:
        """确定多模态处理策略"""
        # 1. 检查是否包含图片
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            return "text_only", "无图片内容，按文本处理"
        
        # 2. 判断模型类型
        is_multimodal = self.is_multimodal_model(model_name)
        should_force_vision = self.should_force_vision_processing(model_name)
        
        self.debug_log(1, f"🤖 模型分析: {model_name} | 多模态支持: {is_multimodal} | 强制视觉处理: {should_force_vision}", "🤖")
        
        # 3. 智能自适应策略
        if is_multimodal:
            # 多模态模型：检查Token预算
            budget_sufficient = self.calculate_multimodal_budget_sufficient(messages, target_tokens)
            if budget_sufficient:
                return "direct_multimodal", "多模态模型，Token预算充足，直接输入"
            else:
                return "multimodal_rag", "多模态模型，Token预算不足，使用多模态向量RAG"
        else:
            # 纯文本模型：需要先识别图片
            return "vision_to_text", "纯文本模型，先识别图片再处理"

    async def process_multimodal_content(
        self, messages: List[dict], model_name: str, target_tokens: int, progress: ProgressTracker
    ) -> List[dict]:
        """多模态内容处理"""
        if not self.valves.enable_multimodal:
            return messages
        
        # 检查是否包含图片
        has_images = self.has_images_in_messages(messages)
        if not has_images:
            self.debug_log(2, "消息中无图片内容，跳过多模态处理", "📝")
            return messages
        
        # 确定多模态处理策略
        strategy, strategy_desc = await self.determine_multimodal_processing_strategy(
            messages, model_name, target_tokens
        )
        
        self.debug_log(1, f"🎯 多模态处理策略: {strategy} - {strategy_desc}", "🎯")
        
        # 根据策略处理
        if strategy == "text_only":
            # 无图片，直接返回
            return messages
        elif strategy == "direct_multimodal":
            # 多模态模型，Token预算充足，直接输入原始内容
            self.debug_log(1, f"✅ 多模态模型直接输入原始内容", "🖼️")
            return messages
        elif strategy == "vision_to_text":
            # 纯文本模型或强制处理，先识别图片再处理
            await progress.start_phase("视觉识别转文本", len(messages))
            
            # 统计图片数量
            total_images = 0
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    total_images += len([
                        item for item in msg.get("content", [])
                        if item.get("type") == "image_url"
                    ])
            
            self.debug_log(1, f"🔍 开始视觉识别转文本：{total_images} 张图片", "🔍")
            
            # 并发处理所有消息
            semaphore = asyncio.Semaphore(self.valves.max_concurrent_requests)
            
            async def process_single_message(i, message):
                if self.has_images_in_content(message.get("content")):
                    async with semaphore:
                        # 创建子进度追踪
                        class SubProgress:
                            def __init__(self, parent, base_count):
                                self.parent = parent
                                self.base_count = base_count
                                self.event_emitter = parent.event_emitter
                            
                            async def update_progress(self, current, total, detail):
                                await self.parent.update_progress(
                                    self.base_count + current, self.parent.phase_total, detail
                                )
                        
                        sub_progress = SubProgress(progress, i)
                        processed_message = await self.process_message_images(message, sub_progress)
                        return processed_message
                else:
                    return message
            
            # 并发处理所有消息
            process_tasks = []
            for i, message in enumerate(messages):
                task = process_single_message(i, message)
                process_tasks.append(task)
            
            # 等待所有任务完成
            processed_messages = await asyncio.gather(*process_tasks)
            
            # 确保消息顺序正确
            if self.message_order:
                processed_messages = self.message_order.sort_messages_preserve_user(
                    processed_messages, self.current_user_message
                )
            
            self.debug_log(1, f"✅ 视觉识别转文本完成：{total_images} 张图片", "✅")
            await progress.complete_phase(f"处理完成 {total_images} 张图片")
            
            return processed_messages
        elif strategy == "multimodal_rag":
            # 多模态模型，Token预算不足，保留原始内容用于后续RAG处理
            self.debug_log(1, f"🔍 多模态RAG策略：保留原始内容用于向量处理", "🔍")
            # 在这个阶段不处理图片，让后续的RAG流程处理
            return messages
        else:
            # 默认策略
            self.debug_log(1, f"⚠️ 未知处理策略 {strategy}，使用默认处理", "⚠️")
            return messages

    # ========== 主要入口函数 ==========
    
    async def inlet(
        self, body: dict, user: Optional[dict] = None, __event_emitter__: Optional[Callable] = None
    ) -> dict:
        """入口函数 - 处理请求 - 修复RAG搜索无结果问题"""
        print("\n🚀 ===== INLET CALLED (Content Maximization Only v2.3.1) =====")
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
        
        # 重置处理状态
        self.reset_processing_state()
        
        # 分析模型信息 - 使用智能匹配器
        self.current_model_info = self.analyze_model(model_name)
        
        # 创建进度追踪器
        progress = ProgressTracker(__event_emitter__)
        
        # 初始化消息顺序管理器
        self.message_order = MessageOrder(messages)
        
        # 使用修复后的消息分离逻辑
        current_user_message, history_messages = self.separate_current_and_history_messages(messages)
        
        # 保存当前用户消息的引用
        self.current_user_message = current_user_message
        
        # 简化Token分析 - 使用简化计算器
        original_tokens = self.count_messages_tokens(messages)
        model_token_limit = self.get_model_token_limit(model_name)
        
        # 使用当前用户消息的token数计算目标
        current_user_tokens = self.count_message_tokens(current_user_message) if current_user_message else 0
        target_tokens = self.calculate_target_tokens(model_name, current_user_tokens)
        
        # 更新统计
        self.stats.token_limit = model_token_limit
        self.stats.target_tokens = target_tokens
        self.stats.current_user_tokens = current_user_tokens
        
        print(f"🔥 简化Token统计: {original_tokens:,}/{model_token_limit:,} (目标:{target_tokens:,})")
        print(f"🎯 模型信息: {self.current_model_info['family']}家族 | "
              f"{'多模态' if self.current_model_info['multimodal'] else '文本'} | "
              f"{self.current_model_info['match_type']}匹配")
        print(f"🔥 内容最大化模式: 启用")
        print(f"🪟 最大窗口利用率: {self.valves.max_window_utilization:.1%}")
        print(f"🔄 激进内容合并: {self.valves.aggressive_content_recovery}")
        print(f"🔒 尽量保留机制: {self.valves.enable_try_preserve} (预算:{self.valves.try_preserve_ratio:.1%})")
        print(f"📚 上下文最大化: {self.valves.enable_context_maximization} "
              f"(直接保留:{self.valves.context_max_direct_preserve_ratio:.1%})")
        print(f"🛡️ 容错保护机制: {self.valves.enable_fallback_preservation} "
              f"(预留:{self.valves.fallback_preserve_ratio:.1%})")
        print(f"🔑 智能关键字生成: {self.valves.enable_keyword_generation}")
        print(f"🧠 AI上下文最大化检测: {self.valves.enable_ai_context_max_detection}")
        print(f"🧩 智能分片: {self.valves.enable_smart_chunking} "
              f"(阈值:{self.valves.large_message_threshold:,}tokens)")
        print(f"📊 Token计算器: 简化版（仅用tiktoken）")
        print(f"🖼️ 多模态模型: {self.valves.multimodal_model}")
        print(f"📝 文本处理模型: {self.valves.text_model}")
        
        # 分析当前用户消息和历史消息
        if current_user_message:
            content_preview = self.message_order.get_message_preview(current_user_message)
            
            # 生成处理ID
            processing_id = hashlib.md5(f"{current_user_message.get('_order_id', '')}{content_preview}".encode()).hexdigest()[:8]
            self.current_processing_id = processing_id
            
            if len(content_preview) > 50:
                content_preview = content_preview[:50] + "..."
            
            print(f"💬 当前用户消息 [ID:{processing_id}]: {current_user_tokens}tokens")
            print(f'💬 当前用户输入: "{content_preview}"')
            print(f"📜 历史消息: {len(history_messages)}条 ({self.count_messages_tokens(history_messages):,}tokens)")
            
            # 使用AI检测上下文最大化需求
            query_text = self.extract_text_from_content(current_user_message.get("content", ""))
            if self.valves.enable_ai_context_max_detection:
                try:
                    need_context_max = await self.detect_context_max_need(query_text, __event_emitter__)
                    print(f"🧠 AI上下文最大化检测结果: {'需要上下文最大化' if need_context_max else '不需要上下文最大化'}")
                    if need_context_max:
                        print(f"📚 上下文最大化将强制处理所有剩余消息")
                        print(f"📚 直接保留比例: {self.valves.context_max_direct_preserve_ratio:.1%}")
                        print(f"🧩 大消息分片阈值: {self.valves.large_message_threshold:,}tokens")
                        print(f"🔄 递归摘要压缩比例: {self.valves.summary_compression_ratio:.1%}")
                        print(f"🔄 强制进入分片+摘要流程处理剩余消息")
                        print(f"🛡️ 容错保护确保最少保留{self.valves.min_history_messages}条历史消息")
                except Exception as e:
                    print(f"🧠 AI上下文最大化检测失败: {e}")
                    need_context_max = self.is_context_max_need_simple(query_text)
                    print(f"🧠 使用简单方法检测: {'需要上下文最大化' if need_context_max else '不需要上下文最大化'}")
            else:
                need_context_max = self.is_context_max_need_simple(query_text)
                print(f"🧠 简单方法检测: {'需要上下文最大化' if need_context_max else '不需要上下文最大化'}")
        else:
            print("⚠️ 未找到当前用户消息")
        
        # 修复判断逻辑
        should_maximize = self.should_force_maximize_content(messages, target_tokens)
        print(f"🔥 是否需要最大化: {should_maximize}")
        print(f"📊 当前窗口利用率: {original_tokens/target_tokens:.1%}")
        print(f"🎯 目标窗口使用率: {self.valves.target_window_usage:.1%}")
        print(f"🔒 最小内容保留比例: {self.valves.min_preserve_ratio:.1%}")
        
        try:
            # 1. 多模态处理
            if self.valves.enable_detailed_progress:
                await progress.start_phase("多模态处理", 1)
            
            processed_messages = await self.process_multimodal_content(
                messages, model_name, target_tokens, progress
            )
            
            processed_tokens = self.count_messages_tokens(processed_messages)
            print(f"📊 多模态处理后: {processed_tokens:,} tokens")
            
            # 2. 内容最大化处理
            if should_maximize:
                print(f"🔥 启动内容最大化处理...")
                
                if current_user_message:
                    query_text = self.extract_text_from_content(current_user_message.get("content", ""))
                    if self.valves.enable_context_maximization:
                        need_context_max = await self.detect_context_max_need(query_text, __event_emitter__)
                        if need_context_max:
                            print(f"📚 上下文最大化专用处理：强制处理所有剩余消息")
                            print(f"📚 修复处理流程，确保处理所有剩余消息")
                            print(f"🧩 公用分片功能，确保分片+摘要流程正常工作")
                            print(f"🛡️ 容错保护机制防止RAG搜索无结果导致的上下文丢失")
                        else:
                            print(f"🔍 具体查询标准处理：RAG流程包含向量检索和重排序")
                            print(f"🛡️ 容错保护确保即使RAG无结果也有足够历史上下文")
                    else:
                        print(f"🔍 标准RAG流程将启动，包含向量检索和重排序")
                        print(f"🛡️ 容错保护机制防止搜索无结果")
                
                print(f"🧩 公用智能分片将处理大消息（阈值:{self.valves.large_message_threshold:,}tokens）")
                print(f"🔄 递归摘要将处理超大内容集合")
                print(f"🛡️ 容错保护预留{self.valves.fallback_preserve_ratio:.1%}预算确保最少{self.valves.min_history_messages}条历史消息")
                
                # 使用修复后的内容最大化综合处理策略
                final_messages = await self.maximize_content_comprehensive_processing(
                    processed_messages, target_tokens, progress
                )
                
                # 打印详细统计
                self.print_detailed_stats()
                
                # 确保返回的消息是深拷贝，不过度清理
                body["messages"] = copy.deepcopy(final_messages)
                
                # 最终统计
                final_tokens = self.count_messages_tokens(final_messages)
                window_utilization = final_tokens / target_tokens if target_tokens > 0 else 0
                
                print(f"🔥 内容最大化处理完成 [ID:{self.current_processing_id}]")
                print(f"📊 最终统计: {len(final_messages)}条消息, {final_tokens:,}tokens")
                print(f"🪟 窗口利用率: {window_utilization:.1%}")
                print(f"📈 内容保留率: {self.stats.calculate_retention_ratio():.1%}")
                print(f"🔑 关键字生成: {self.stats.keyword_generations}次")
                print(f"🧠 上下文最大化检测: {self.stats.context_maximization_detections}次")
                print(f"🔒 尽量保留: {self.stats.try_preserve_messages}条({self.stats.try_preserve_tokens:,}tokens)")
                print(f"📚 上下文最大化处理: 直接保留{self.stats.context_max_direct_preserve}条, "
                      f"分片{self.stats.context_max_chunked}条, 摘要{self.stats.context_max_summarized}条")
                print(f"🎨 多模态提取: {self.stats.multimodal_extracted}个多模态消息")
                print(f"🧩 智能分片: 创建{self.stats.chunk_created}个分片，处理{self.stats.chunked_messages}条大消息")
                print(f"🔄 递归摘要: {self.stats.recursive_summaries}次，摘要{self.stats.summarized_messages}条消息")
                print(f"🔍 向量检索: {self.stats.vector_retrievals}次")
                print(f"🔄 重排序: {self.stats.rerank_operations}次")
                print(f"🛡️ 容错保护: 后备保留{self.stats.fallback_preserve_applied}次, "
                      f"用户消息恢复{self.stats.user_message_recovery_count}次, RAG无结果{self.stats.rag_no_results_count}次")
                print(f"🎯 Token计算器: 简化版（仅用tiktoken）")
                print(f"🖼️ 多模态模型: {self.valves.multimodal_model}")
                print(f"📝 文本处理模型: {self.valves.text_model}")
                
                # 验证当前用户消息是否保护成功
                if current_user_message and final_messages:
                    last_msg = final_messages[-1]
                    if last_msg.get("role") == "user":
                        print(f"✅ 当前用户消息保护成功！")
                    else:
                        print(f"❌ 最后一条消息不是用户消息！")
                        print(f"❌ 最后一条消息角色: {last_msg.get('role', 'unknown')}")
            else:
                # 直接使用处理后的消息
                self.stats.original_tokens = self.count_messages_tokens(messages)
                self.stats.original_messages = len(messages)
                self.stats.final_tokens = processed_tokens
                self.stats.final_messages = len(processed_messages)
                
                # 检查当前用户消息是否保留
                final_current_message = self.find_current_user_message(processed_messages)
                self.stats.current_user_preserved = final_current_message is not None
                
                # 计算窗口使用率
                window_usage = self.stats.calculate_window_usage_ratio()
                print(f"🪟 窗口使用率: {window_usage:.1%}")
                
                if self.valves.enable_detailed_progress:
                    await progress.complete_phase("无需最大化处理")
                
                # 确保返回的消息是深拷贝，不过度清理
                body["messages"] = copy.deepcopy(processed_messages)
                
                print(f"✅ 直接使用处理后的消息 [ID:{self.current_processing_id}]")
        
        except Exception as e:
            print(f"❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()
            
            if self.valves.enable_detailed_progress:
                await progress.update_status(f"处理失败: {str(e)[:50]}", True)
        
        print(f"🏁 ===== INLET DONE (Content Maximization Only v2.3.1) [ID:{self.current_processing_id}] =====\n")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None, __event_emitter__: Optional[Callable] = None) -> dict:
        """出口函数 - 返回响应"""
        return body
