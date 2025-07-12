
# 🚀 Advanced Multimodal Context Manager

一个强大的Open WebUI插件，提供智能长上下文管理和多模态内容处理功能。

## 📋 目录 | Table of Contents

- [中文文档](#中文文档)
- [English Documentation](#english-documentation)

---

# 中文文档

## 🌟 功能特性

### 🖼️ 多模态处理
- **智能图片识别**：自动将图片转换为详细的文本描述
- **多种处理策略**：支持全模型处理、非多模态模型处理、自定义列表等
- **图片信息保留**：在摘要过程中确保图片描述信息不丢失
- **缓存机制**：相同图片复用识别结果，提高效率

### 💾 智能上下文管理
- **Token限制管理**：根据模型自动调整Token使用量
- **保护策略**：智能保护当前查询和最近对话轮次
- **递归摘要**：多轮摘要确保重要信息不丢失
- **紧急截断**：在极端情况下的保护机制

### 🔍 向量检索（预留功能）
- **多模态向量**：支持图片和文本混合向量化
- **语义检索**：基于语义相似度检索相关内容
- **智能重排序**：优化检索结果的相关性排序
- **多种向量策略**：支持不同的向量化策略

### 🎛️ 灵活配置
- **处理策略**：可自定义哪些模型需要特殊处理
- **参数调节**：丰富的配置选项满足不同需求
- **调试模式**：多级调试信息帮助问题排查
- **实时状态**：前端显示处理进度和状态

## 🚀 安装使用

### 安装步骤
1. 打开Open WebUI管理界面
2. 进入"设置" → "管道"
3. 点击"+"添加新管道
4. 将插件代码粘贴到编辑器中
5. 点击"保存"完成安装

### 基础配置
安装后需要配置以下关键参数：

```yaml
# 必需配置
vision_api_key: "你的视觉API密钥"
vision_api_base: "https://your-vision-api.com"

# 可选配置
multimodal_processing_strategy: "smart_adaptive"
enable_processing: true
```

## 📖 详细功能说明

### 1. 多模态处理功能

#### 🎯 处理策略
- **智能自适应** (`smart_adaptive`): 根据模型类型和上下文自动选择最佳处理方式
- **全模型处理** (`all_models`): 对所有模型都进行图片处理
- **非多模态模型** (`non_multimodal_only`): 只对不支持多模态的模型处理
- **自定义列表** (`custom_list`): 根据指定的模型列表进行处理

#### 🖼️ 图片处理流程
1. **检测图片**：自动识别消息中的图片内容
2. **调用视觉API**：将图片发送到配置的视觉模型
3. **生成描述**：获取详细的图片描述文本
4. **替换内容**：将图片替换为描述文本（保留原图的情况除外）
5. **缓存结果**：保存识别结果供后续使用

#### 📸 图片保留策略
- **保留原图**：对于原生支持多模态的模型，可选择保留原始图片
- **替换为文本**：对于不支持多模态的模型，将图片替换为文本描述
- **混合模式**：根据上下文长度动态选择处理方式

### 2. 智能上下文管理

#### 🔐 保护机制
- **当前查询保护**：始终保护用户的最新查询
- **对话轮次保护**：保护指定数量的完整对话轮次
- **重要消息保护**：确保关键信息不被摘要丢失

#### 📊 Token管理
- **动态限制**：根据模型类型自动调整Token限制
- **安全比例**：预留安全margin防止超限
- **智能分配**：在保护和摘要之间智能分配Token

#### 🔄 递归摘要
- **多轮摘要**：必要时进行多轮摘要处理
- **质量检查**：确保摘要质量符合要求
- **信息保留**：最大化保留重要信息

### 3. 向量检索功能（预留）

#### 🎯 检索策略
- **多模态优先**：优先使用多模态向量进行检索
- **文本优先**：优先使用文本向量进行检索
- **混合模式**：结合多种向量类型进行检索

#### 🔍 相似度匹配
- **动态阈值**：根据内容类型调整相似度阈值
- **语义理解**：基于语义相似度而非字面匹配
- **上下文感知**：考虑上下文信息提高匹配准确性

## ⚙️ 配置参数详解

### 基础控制
- `enable_processing`: 启用/禁用所有处理功能
- `excluded_models`: 排除的模型列表（逗号分隔）
- `debug_level`: 调试级别（0-3）

### 多模态配置
- `multimodal_processing_strategy`: 多模态处理策略
- `force_vision_processing_models`: 强制处理图片的模型列表
- `preserve_images_in_multimodal`: 多模态模型是否保留原图
- `always_process_images_before_summary`: 摘要前是否总是处理图片

### Token管理
- `default_token_limit`: 默认Token限制
- `token_safety_ratio`: Token安全比例
- `max_preserve_ratio`: 保护消息最大Token比例
- `max_single_message_tokens`: 单条消息最大Token数

### 视觉API配置
- `vision_api_base`: 视觉API基础URL
- `vision_api_key`: 视觉API密钥
- `vision_model`: 视觉模型名称
- `vision_prompt_template`: 视觉识别提示词模板

### 摘要配置
- `summary_api_base`: 摘要API基础URL
- `summary_api_key`: 摘要API密钥
- `summary_model`: 摘要模型名称
- `max_summary_length`: 摘要最大长度

## 🔧 常见问题

### Q: 为什么图片没有被处理？
A: 检查以下几点：
1. 确认`enable_multimodal`已启用
2. 检查模型是否在排除列表中
3. 确认视觉API配置正确
4. 查看调试日志了解详细信息

### Q: 如何优化处理速度？
A: 可以尝试：
1. 调整`max_concurrent_requests`增加并发数
2. 使用缓存机制避免重复处理
3. 优化`chunk_size`和`overlap_size`参数
4. 选择合适的处理策略

### Q: 摘要质量不好怎么办？
A: 可以：
1. 调整`max_summary_length`参数
2. 优化摘要提示词
3. 增加`max_recursion_depth`允许更多递归
4. 检查摘要模型是否合适

## 📝 更新日志

### v1.4.4
- 新增多模态处理策略配置
- 修复图片信息在摘要中丢失的问题
- 增强对原生多模态模型的支持
- 优化Token管理和保护机制

### v1.4.3
- 实现递归摘要功能
- 添加智能上下文管理
- 优化多模态处理流程
- 增加详细的调试信息

---

# English Documentation

## 🌟 Features

### 🖼️ Multimodal Processing
- **Intelligent Image Recognition**: Automatically converts images to detailed text descriptions
- **Multiple Processing Strategies**: Supports all models, non-multimodal models, custom lists, etc.
- **Image Information Preservation**: Ensures image descriptions are not lost during summarization
- **Caching Mechanism**: Reuses recognition results for identical images to improve efficiency

### 💾 Smart Context Management
- **Token Limit Management**: Automatically adjusts token usage based on model
- **Protection Strategy**: Intelligently protects current queries and recent conversation rounds
- **Recursive Summarization**: Multi-round summarization ensures important information is not lost
- **Emergency Truncation**: Protection mechanism in extreme situations

### 🔍 Vector Retrieval (Reserved Feature)
- **Multimodal Vectors**: Supports mixed vectorization of images and text
- **Semantic Retrieval**: Retrieves relevant content based on semantic similarity
- **Smart Reranking**: Optimizes relevance ranking of retrieval results
- **Multiple Vector Strategies**: Supports different vectorization strategies

### 🎛️ Flexible Configuration
- **Processing Strategies**: Customizable model-specific processing
- **Parameter Adjustment**: Rich configuration options for different needs
- **Debug Mode**: Multi-level debug information for troubleshooting
- **Real-time Status**: Frontend display of processing progress and status

## 🚀 Installation & Usage

### Installation Steps
1. Open Open WebUI admin interface
2. Go to "Settings" → "Pipelines"
3. Click "+" to add new pipeline
4. Paste the plugin code into the editor
5. Click "Save" to complete installation

### Basic Configuration
After installation, configure these key parameters:

```yaml
# Required Configuration
vision_api_key: "your-vision-api-key"
vision_api_base: "https://your-vision-api.com"

# Optional Configuration
multimodal_processing_strategy: "smart_adaptive"
enable_processing: true
```

## 📖 Detailed Feature Description

### 1. Multimodal Processing

#### 🎯 Processing Strategies
- **Smart Adaptive** (`smart_adaptive`): Automatically selects optimal processing based on model type and context
- **All Models** (`all_models`): Processes images for all models
- **Non-multimodal Only** (`non_multimodal_only`): Only processes for models that don't support multimodal
- **Custom List** (`custom_list`): Processes based on specified model list

#### 🖼️ Image Processing Flow
1. **Detect Images**: Automatically identifies image content in messages
2. **Call Vision API**: Sends images to configured vision model
3. **Generate Descriptions**: Obtains detailed image description text
4. **Replace Content**: Replaces images with description text (except when preserving originals)
5. **Cache Results**: Saves recognition results for future use

#### 📸 Image Preservation Strategy
- **Preserve Originals**: For native multimodal models, option to keep original images
- **Replace with Text**: For non-multimodal models, replace images with text descriptions
- **Hybrid Mode**: Dynamically choose processing method based on context length

### 2. Smart Context Management

#### 🔐 Protection Mechanism
- **Current Query Protection**: Always protects user's latest query
- **Conversation Round Protection**: Protects specified number of complete conversation rounds
- **Important Message Protection**: Ensures key information is not lost in summarization

#### 📊 Token Management
- **Dynamic Limits**: Automatically adjusts token limits based on model type
- **Safety Ratio**: Reserves safety margin to prevent overflow
- **Smart Allocation**: Intelligently allocates tokens between protection and summarization

#### 🔄 Recursive Summarization
- **Multi-round Summarization**: Performs multiple rounds of summarization when necessary
- **Quality Check**: Ensures summarization quality meets requirements
- **Information Retention**: Maximizes retention of important information

### 3. Vector Retrieval (Reserved)

#### 🎯 Retrieval Strategies
- **Multimodal First**: Prioritizes multimodal vectors for retrieval
- **Text First**: Prioritizes text vectors for retrieval
- **Mixed Mode**: Combines multiple vector types for retrieval

#### 🔍 Similarity Matching
- **Dynamic Thresholds**: Adjusts similarity thresholds based on content type
- **Semantic Understanding**: Based on semantic similarity rather than literal matching
- **Context Awareness**: Considers context information to improve matching accuracy

## ⚙️ Configuration Parameters

### Basic Control
- `enable_processing`: Enable/disable all processing functions
- `excluded_models`: List of excluded models (comma-separated)
- `debug_level`: Debug level (0-3)

### Multimodal Configuration
- `multimodal_processing_strategy`: Multimodal processing strategy
- `force_vision_processing_models`: List of models to force image processing
- `preserve_images_in_multimodal`: Whether to preserve original images in multimodal models
- `always_process_images_before_summary`: Whether to always process images before summarization

### Token Management
- `default_token_limit`: Default token limit
- `token_safety_ratio`: Token safety ratio
- `max_preserve_ratio`: Maximum token ratio for protected messages
- `max_single_message_tokens`: Maximum tokens for single message

### Vision API Configuration
- `vision_api_base`: Vision API base URL
- `vision_api_key`: Vision API key
- `vision_model`: Vision model name
- `vision_prompt_template`: Vision recognition prompt template

### Summarization Configuration
- `summary_api_base`: Summary API base URL
- `summary_api_key`: Summary API key
- `summary_model`: Summary model name
- `max_summary_length`: Maximum summary length

## 🔧 FAQ

### Q: Why aren't images being processed?
A: Check the following:
1. Confirm `enable_multimodal` is enabled
2. Check if model is in exclusion list
3. Verify vision API configuration is correct
4. Check debug logs for detailed information

### Q: How to optimize processing speed?
A: Try:
1. Adjust `max_concurrent_requests` to increase concurrency
2. Use caching mechanism to avoid duplicate processing
3. Optimize `chunk_size` and `overlap_size` parameters
4. Choose appropriate processing strategy

### Q: What if summarization quality is poor?
A: You can:
1. Adjust `max_summary_length` parameter
2. Optimize summarization prompt
3. Increase `max_recursion_depth` for more recursion
4. Check if summary model is suitable

## 📝 Changelog

### v1.4.4
- Added multimodal processing strategy configuration
- Fixed image information loss in summarization
- Enhanced support for native multimodal models
- Optimized token management and protection mechanisms

### v1.4.3
- Implemented recursive summarization functionality
- Added smart context management
- Optimized multimodal processing flow
- Added detailed debug information

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Thanks to the Open WebUI community and all contributors who made this project possible.
