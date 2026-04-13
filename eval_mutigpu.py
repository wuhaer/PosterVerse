import os
import json
import argparse
import torch
import torch.distributed as dist
import re
from PIL import Image
from tqdm import tqdm
import math
import logging
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import traceback
from qwen_vl_utils import process_vision_info


def setup_logging(rank):
    """设置日志格式"""
    # 清除已有的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置基本日志级别
    level = logging.INFO if rank in [0, -1] else logging.WARNING
    
    # 创建日志格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [Rank %(rank)s] - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    # 创建处理器并设置格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # 获取logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    # 添加rank信息
    logger = logging.LoggerAdapter(logger, {"rank": rank})
    
    return logger

def setup_dist(backend="nccl"):
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # 单GPU模式
        rank = -1
        world_size = 1
        local_rank = 0
    
    if rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    
    return rank, world_size, local_rank

def load_image(image_path):
    """加载图像文件"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_html_from_response(response_text):
    """从模型响应中提取HTML代码"""
    # 尝试提取代码块
    html_match = re.search(r'```(?:html)?\s*(<!DOCTYPE.+?|<html.+?)\s*```', response_text, re.DOTALL)
    if html_match:
        return html_match.group(1)
    
    # 如果没有代码块格式，尝试直接提取HTML
    html_match = re.search(r'(<!DOCTYPE.+?<\/html>|<html.+?<\/html>)', response_text, re.DOTALL)
    if html_match:
        return html_match.group(1)
    
    # 如果没有找到标准HTML，返回整个响应
    return response_text

class PosterHTMLGenerator:
    def __init__(self, model_path, device, rank=-1, world_size=1, local_rank=0, 
                 max_new_tokens=2048, temperature=0.7, top_p=0.9):
        """初始化海报HTML生成器"""
        self.model_path = model_path
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        # self.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # 设置日志记录器
        self.logger = setup_logging(rank)
        
        self._init_model()
    
    def _init_model(self):
        """初始化模型和tokenizer"""
        self.logger.info(f"Loading Qwen2.5VL model from {self.model_path}...")
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 根据是否是分布式加载模型
        if self.world_size > 1 and self.rank != -1:
            # 分布式模式只加载当前设备需要的模型部分
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,  # 使用bfloat16节省显存
                trust_remote_code=True
            ).eval()
        else:
            # 单GPU模式
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval()
        
        self.logger.info(f"Model loaded successfully on rank {self.rank}")

    def load_dataset(self, input_file):
        """加载并分割数据集"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据rank分配数据
        if self.world_size > 1 and self.rank != -1:
            per_rank = math.ceil(len(data) / self.world_size)
            start_idx = self.rank * per_rank
            end_idx = min((self.rank + 1) * per_rank, len(data))
            my_data = data[start_idx:end_idx]
            self.logger.info(f"Rank {self.rank}: Loaded {len(my_data)} samples (from {start_idx} to {end_idx-1}) out of {len(data)} total samples")
            return my_data
        else:
            self.logger.info(f"Loaded {len(data)} samples from {input_file}")
            return data

    def generate_html(self, image, prompt):
        """使用Qwen2.5VL生成HTML"""
        try:
            # 准备输入
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # 使用processor处理输入
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # 将输入移动到正确的设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
            # 生成文本
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
            
            # 确保在同一设备上进行操作
            input_ids = inputs['input_ids'].to(generated_ids.device)
            
            # 解码生成的ID，只保留新生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # 提取HTML内容
            html_content = extract_html_from_response(response)
            
            return html_content, response
            
        except Exception as e:
            self.logger.error(f"Error generating HTML: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"<!-- Error generating HTML: {str(e)} -->", str(e)

    def process_sample(self, sample, output_dir):
        """处理单个样本"""

        sample_id = sample.get("id", 0)
        
        # 提取用户消息
        user_message = next((m for m in sample["messages"] if m["role"] == "user"), None)
        if not user_message:
            self.logger.warning(f"Sample {sample_id} has no user message, skipping")
            return {"id": sample_id, "status": "error", "error": "No user message found"}
        
        # 提取提示内容（移除<image>标签）
        user_content = user_message["content"]
        if isinstance(user_content, str) and "<image>" in user_content:
            prompt = user_content.split("<image>", 1)[1].strip()
        else:
            prompt = user_content
        
        # 提取图像路径
        image_path = sample["images"][0] if sample.get("images") else None
        if not image_path or not os.path.exists(image_path):
            self.logger.warning(f"Sample {sample_id} has invalid image path: {image_path}")
            return {"id": sample_id, "status": "error", "error": f"Invalid image path: {image_path}"}
        
        # 加载图像
        image = load_image(image_path)
        if image is None:
            return {"id": sample_id, "status": "error", "error": f"Failed to load image: {image_path}"}
        
        import pdb; pdb.set_trace()
        # 生成HTML
        html_content, full_response = self.generate_html(image, prompt)
        

        
        return {
            "id": sample_id,
            "status": "success",
            "image_path": image_path,
            "prompt": prompt,
        }
        


    def run_generation(self, input_file, output_dir):
        """运行完整的HTML生成流程"""

        
        # 加载数据集
        dataset = self.load_dataset(input_file)
        
        
        # 处理每个样本
        results = []

        for i, sample in enumerate(tqdm(dataset, desc=f"Rank {self.rank} generating", 
                                        disable=self.rank not in [0, -1])):

            # 处理样本
            result = self.process_sample(sample, output_dir)
            results.append(result)
            
        
            
 


        


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-GPU HTML Poster Generation with Qwen2.5VL')
    parser.add_argument('--model_path', type=str, 
                        default="/path/to/htmlgenerationmodel",
                        help='Path to the model directory or name')
    parser.add_argument('--input_file', type=str, 
                        default="/path/to/json",
                        help='Path to the input dataset JSON file')
    parser.add_argument('--output_dir', type=str, 
                        default="./html_generation_results",
                        help='Output directory for generated HTML')
    parser.add_argument('--max_new_tokens', type=int, 
                        default=8192,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, 
                        default=0.7,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, 
                        default=0.9,
                        help='Top-p sampling parameter')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_dist()
    
    # 设置日志
    logger = setup_logging(rank)
    logger.info(f"Starting HTML generation with rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"qwen25vl_html_{timestamp}")
    
    # 创建生成器
    generator = PosterHTMLGenerator(
        model_path=args.model_path,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 运行HTML生成
    generator.run_generation(args.input_file, output_dir)
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()
    
    logger.info(f"Rank {rank} finished HTML generation")


if __name__ == "__main__":
    main()
