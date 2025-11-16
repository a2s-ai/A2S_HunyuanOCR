<p align="center">
 <img src="./assets/hunyuan_logo.png" width="400"/> <br>
</p>

<div align="center">

# HunyuanOCR: An Advanced multilingual VLM with 1B parameter for end-to-end OCR tasks

📑 Paper & Model Weights are coming soon!

[Demo](#demo) | [Install](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation)

</div>

## 🔥 News
- **[2025/11/20]** 📝 Inference code and model weights publicly available.

## 📖 Introduction
HunyuanOCR is an open-source, commercial-grade, and highly efficient multilingual VLM designed for diverse OCR tasks. It integrates a wide range of capabilities, including text spotting, complex document parsing, information extraction, text-centric VQA and multilingual text image translation into a single end-to-end architecture. With only 1B parameters, the model achieves strong efficiency and competitive performance, outperforming established open-source cascade systems and rivalling large-scale VLMs as well as certain commercial APIs on specific tasks. Its purely end-to-end design enables single-pass inference, substantially simplifying deployment and reducing operational complexity.

<div align="center">
  <img src="./assets/hyocr-pipeline.png" alt="HunyuanOCR framework" width="90%">
</div>

## ✨ Key Features

- 💪 **Compact Yet Powerful**: The first open-source, compact (1B parameters), and efficient multilingual VLM designed specifically for OCR, delivering commercial-grade performance across multiple applications while maintaining deployment simplicity.
- 📑 **Unified Multi-task Capabilities**: Seamlessly integrates multiple OCR-related tasks including text detection, recognition, document parsing, information extraction, visual question answering, and text image translation in a single end-to-end architecture.
- 🌏 **Extensive Language Support**: Provides robust support for over 100 languages, excelling in both single-language and mixed-language scenarios across various document types and formats.
- 🚀 **Easy Deployment**: Simple integration with existing systems


## 🛠️ Dependencies and Installation

### System Requirements
- 🖥️ Operating System: Linux
- 🐍 Python: 3.12+ (recommended and tested)
- ⚡ CUDA: 12.8
- 🔥 PyTorch: 2.7.1
- 🎮 GPU: NVIDIA GPU with CUDA support
- 🧠 GPU Memory: ≥3×80GB (4×80GB recommended for better performance)
- 💾 Disk Space: 170GB for model weights

### Installation
```bash
pip install https://mirrors.tencent.com/repository/generic/transformers/transformers-4.56.0.post2-py3-none-any.whl
pip install opencv-python-headless
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

## 📋 Model Card

| Component | Architecture | Parameters | Function |
|-----------|-------------|------------|-----------|
| Vision Encoder | SigVLIP-v2 (ViT-based) | 400M | Image processing and feature extraction |
| Language Model | Hunyuan-LLM | 500M | Text understanding and generation |
| Vision-Language Bridge | MLP Adapter | 90M | Multimodal feature fusion |
| **Total** | - | **~1B** | End-to-end OCR and document understanding |

## 🚀 Quick Start with Transformers

### Model Inference

HunyuanOCR provides intuitive inference interfaces. Here's how to get started:

#### 1. Basic Setup
```python
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, HunYuanVLV1ForConditionalGeneration
from qwen_vl_utils import process_vision_info
```

#### 2. Model Loading
```python
def load_model():
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        "PATH_TO_MODEL",
        use_fast=False,
        trust_remote_code=True
    )
    
    model = HunYuanVLV1ForConditionalGeneration.from_pretrained(
        "PATH_TO_MODEL",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return model, processor
```

## 3. Inference Function
```python
def inference(text: str, img_path: str, model, processor) -> list:
    # Construct input format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": text},
        ],
    }]
    
    # Process inputs
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
             for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Model inference
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate results
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024*8,
            repetition_penalty=1.03,
            do_sample=False
        )
    
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
```

## 4. Usage Examples
```python
# Load model
model, processor = load_model()

# Single image inference
img_path = "path/to/your/image.jpg"
query = "Please extract the text content from this image"
result = inference(text=query, img_path=img_path, model=model, processor=processor)
print("Inference result:", result)

# Batch processing
import json
from tqdm import tqdm

with open("input.jsonl", "r") as fin, open("output.jsonl", "w") as fout:
    for line in tqdm(fin):
        data = json.loads(line)
        result = inference(
            text=data["question"],
            img_path=data["img_path"],
            model=model,
            processor=processor
        )
        data["response"] = result
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
```

## 💬 Application-oriented Prompts

| Task | English | Chinese |
|------|---------|---------|
| **Spotting** | Detect and recognize text in the image, and output the text coordinates in a formatted manner. | 检测并识别图片中的文字，将文本坐标格式化输出。 |
| **Parsing** | Identify the formula in the image and represent it using LaTeX format. | 识别图片中的公式，用 LaTeX 格式表示。 |
| | Parse the table in the image into HTML. | 把图中的表格解析为 HTML。 |
| | Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts. | 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。 |
| | Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. | 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。 |
| **IE** | Please output the value of Key. | 请输出 Key 的值。 |
| | Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format. | 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。 |
| | Extract the subtitles from the image. | 提取图片中的字幕。 |
| **Translation** | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. | 先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 |

## 📊 Evaluation

### Text Spotting Performance

| Model Type | Methods | Overall | Art | Doc | Game | Hand | Ads | Receipt | Screen | Scene | Video |
|------------|---------|---------|-----|-----|------|------|-----|----------|---------|--------|--------|
| **Traditional methods** | PaddleOCR | 53.38 | 32.83 | 70.23 | 51.59 | 56.39 | 57.38 | 50.59 | 63.38 | 44.68 | 53.35 |
| | BaiduOCR | 61.9 | 38.5 | **78.95** | 59.24 | 59.06 | 66.7 | **63.66** | 68.18 | 55.53 | 67.38 |
| **General VLM** | Qwen3VL-2B-Instruct | 29.68 | 29.43 | 19.37 | 20.85 | 50.57 | 35.14 | 24.42 | 12.13 | 34.90 | 40.1 |
| | Qwen3VL-235B-Instruct | 53.62 | 46.15 | 43.78 | 48.00 | 68.90 | 64.01 | 47.53 | 45.91 | 54.56 | 63.79 |
| | Seed1.6-VL-Instruct | 59.23 | 45.36 | 55.04 | 59.68 | 67.46 | 65.99 | 55.68 | 59.85 | 53.66 | 70.33 |
| **OCR-Specific VLM** | HunyuanOCR | **70.92** | **56.76** | 73.63 | **73.54** | **77.10** | **75.34** | 63.51 | **76.58** | **64.56** | **77.31** |

> **Summary**: HunyuanOCR achieves the best overall performance (70.92%) across different scenarios, significantly outperforming both traditional OCR methods and general VLMs.

### Document Parsing Performance (Edit Distance)

| Model | English |  |  |  | Chinese |  |  |  |
|-------|---------|---------|----------|--------|----------|---------|----------|---------|
|  | overall | text | formula | table | overall | text | formula | table |
| Dolphin | 0.356 | 0.352 | 0.465 | 0.258 | 0.44 | 0.44 | 0.604 | 0.367 |
| SmolDocling | 0.493 | 0.262 | 0.753 | 0.729 | 0.816 | 0.838 | 0.997 | 0.907 |
| dots.ocr-3B | 0.182 | 0.137 | 0.320 | 0.166 | 0.261 | 0.229 | 0.468 | 0.160 |
| DeepseekOCR | **0.123** | 0.049 | **0.242** | 0.147 | **0.157** | 0.087 | **0.377** | **0.08** |

> **Summary**: DeepseekOCR demonstrates superior performance in both English and Chinese document parsing, achieving the lowest edit distances across most categories.

### Information Extraction and VQA Performance

| Model | Cards & Receipts | Video Subtitles | OCRBench |
|-------|------------------|-----------------|-----------|
| DeepSeek-OCR | 25.29 | 5.41 | 430 |
| PP-ChatOCR | 53.64 | 3.1 | - |
| Qwen3VL-2B-Instruct | 66.12 | 3.75 | 858 |
| Seed1.5-VL | 68.81 | 60.45 | 881 |
| Qwen3VL-235B-A22B-Instruct | 77.0 | 50.74 | **920** |
| Gemini-2.5-Pro | 80.63 | 53.65 | 872 |
| **HunyuanOCR (∼1B)** | **92.41** | **92.87** | 858 |

> **Summary**: Despite its compact size, HunyuanOCR significantly outperforms larger models in cards/receipts processing and video subtitle extraction, while maintaining competitive performance on OCRBench.

## 💡 Case Studies
<details>

<summary>Click to view example results</summary>

Document Processing


Table Recognition


Mixed Language Processing


</details>


## 📚 Citation
@misc{hunyuanocr2025,
    title={HunyuanOCR: Advanced OCR Engine for Document Understanding},
    author={Tencent Hunyuan Team},
    year={2025},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/Tencent/HunyuanOCR}}
}

## 🙏 Acknowledgements
Thanks to all contributors who helped build HunyuanOCR
Special thanks to the Tencent Hunyuan Team
We appreciate the support from the open-source community


## 📄 License
This project is licensed under the Apache 2.0 License.
