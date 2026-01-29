"""
PosterVerse
===========================

- Stage 1: Blueprint Creation
- Stage 2: Graphical Background Generation
- Stage 3: Unified Layout-Text Rendering
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime
import os
from src.flux.xflux_pipeline import XFluxPipeline
import argparse
from PIL import Image
from main_batch_lora import img_gen
from eval_mutigpu import PosterHTMLGenerator
import subprocess
import re
import tempfile
import asyncio
from playwright.async_api import async_playwright
import base64
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default='',
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--lora_path", type=str, default='',
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=1, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=1.0, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-cond-emb", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1080, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1920, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    parser.add_argument(
        "--json_file", type=str, default=None, help="Path to save"
    )
    parser.add_argument('--request_prompt', type=str, 
                        default="",
                        help='Path to the model directory or name')
    parser.add_argument('--stage3_model_path', type=str, 
                        default="",
                        help='Path to the model directory or name')
    parser.add_argument('--stage1_model_path', type=str, 
                        default="",
                        help='Path to the model directory or name')
    parser.add_argument('--input_file', type=str, 
                        default="",
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
    parser.add_argument('--gpu_id', type=int, 
                        default=0,
                        help='GPU device ID to use')
    return parser


def extract_html_from_response(response_text):
    """
    Extract HTML code from model response.
    
    Args:
        response_text: Raw response text from model
        
    Returns:
        Extracted HTML code
    """
    # Try to extract from code block
    html_match = re.search(r'```(?:html)?\s*(<!DOCTYPE.+?|<html.+?)\s*```', response_text, re.DOTALL)
    if html_match:
        return html_match.group(1)
    
    # Try to extract raw HTML
    html_match = re.search(r'(<!DOCTYPE.+?<\/html>|<html.+?<\/html>)', response_text, re.DOTALL)
    if html_match:
        return html_match.group(1)
    
    # Return entire response if no HTML found
    return response_text

class PosterEvaluator:
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = model_path
        self.device = device
        self._init_model()
        
    def _init_model(self):
        logger.info("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype="auto",
            low_cpu_mem_usage=True
        ).to(self.device)
        logger.info("Model loaded successfully")

    def extract_prompts(self, input_file: str) -> List[Dict]:
        """
        Extract user prompts from test data file.
        
        Args:
            input_file: Path to input JSON file
            
        Returns:
            List of prompt dictionaries
        """
        prompts = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    for message in data['messages']:
                        if message['role'] == 'user':
                            user_content = message['content']
                            assistant_content = next(
                                (m['content'] for m in data['messages'] if m['role'] == 'assistant'),
                                None
                            )
                            prompts.append({
                                'prompt': user_content,
                                'expected': assistant_content
                            })
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
                    continue
        return prompts

    def generate_response(self, prompt: str) -> str:
        """
        Generate model response for given prompt.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Generated response text
        """
        prompt = f"{prompt}"
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs['input_ids'],
                max_length=2048,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.batch_decode(
                generated_ids[:, model_inputs.input_ids.size(-1):],
                skip_special_tokens=True
            )[0]
        return response

    def run(self, prompts: str):
        """
        Run the complete evaluation process with retry logic.
        
        Args:
            prompts: User input prompt
            
        Returns:
            Dictionary containing prompt and structured response
        """

        count = 0
        is_valid_json = False
        has_all_keys = None
        generated_json = None
        previous_responses = set()
        
        max_attempts = 10

        while count < max_attempts:
            try:
                count += 1
                generated_json = None

                # Generate response
                response = self.generate_response(prompts)

                if not response:
                    print(f"Attempt {count}: Empty response received")
                    continue
                
                # Check for duplicate responses
                if response in previous_responses:
                    print(f"Attempt {count}: Duplicate response received")
                    continue
                previous_responses.add(response)

                # Extract and validate JSON
                if response.find('{') != -1 and response.find('}') != -1:
                    json_start = response.find('{')
                    json_end = None
                    
                    stack = []
                    in_string = False
                    for i in range(json_start, len(response)):
                        char = response[i]
                        if char == '"':
                            in_string = not in_string
                        if not in_string:
                            if char == '{':
                                stack.append(char)
                            elif char == '}':
                                stack.pop()
                                if not stack:
                                    json_end = i + 1
                                    break
                    
                    if json_end is not None:
                        json_str = response[json_start:json_end]
                        try:
                            generated_json = json.loads(json_str)
                        except:
                            print(f"Attempt {count}: Failed to parse JSON")
                    else:
                        raise ValueError("Could not find matching JSON brackets")
                else:
                    generated_json = json.loads(response)
                
                print(f'Debug \n{response} \n\n{json_str}\n\n\n\n Debug End')
                if generated_json is not None:
                    has_all_keys = all(key in generated_json for key in ['theme', 'element', 'color', 'style'])
                    if has_all_keys:
                        print(f'Success \n{response} \n\n{json_str}\n\n\n\n Success End')
                        break
                        
                print(f"Attempting {count} time(s) \n{generated_json} \n{json_str}")

            except:
                print(f"Attempt {count}: Failed to generate valid JSON")
            
        if not has_all_keys or generated_json is None:
            print(f"Failed to generate valid JSON after {max_attempts} attempts")
            return {
                'prompt': prompts,
                'response': {
                    'error': f"Failed to generate valid JSON after {max_attempts} attempts",
                    'theme': '',
                    'element': '',
                    'color': '',
                    'style': ''
                }
            }

        complete_output = {
            'prompt': prompts,
            'response': generated_json
        }

        return complete_output
                


def process_batch(request_prompt, args, save_paths, model_path, grade):
    """
    Process a single batch through the complete pipeline.
    
    Pipeline stages:
    1. Blueprint Creation (Stage 1 model)
    2. Graphical Background Generation (Stage 2)
    3. Unified Layout-Text Rendering (Stage 3 model)
    
    Args:
        request_prompt: User's natural language prompt
        args: Configuration arguments
        save_paths: Base path for saving outputs
        stage1_model_path: Path to Stage 1 model
    """
    
    classification_dict = {
        "Illustration":"Design",
        "Real":"Real",
        "Pure":"Pure",
        "Design":"Design",
    }

    # Create output directories
    if not os.path.exists(save_paths):
        os.makedirs(save_paths)
        os.makedirs(os.path.join(save_paths, 'final_html'))
        os.makedirs(os.path.join(save_paths, 'background_images'))
        os.makedirs(os.path.join(save_paths, 'json'))

    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)


    try:
        
        # ============================================================================
        # Stage 1: Blueprint Creation
        # ============================================================================
        logger.info("=" * 60)
        logger.info("Stage 1: Blueprint Creation")
        logger.info("=" * 60)

        evaluator = PosterEvaluator(model_path, device)
        complete_output = evaluator.run(request_prompt)
        
        del evaluator
        torch.cuda.empty_cache()

        # Save Blueprint Creation
        json_file = os.path.join(save_paths, 'json')
        os.makedirs(json_file, exist_ok=True)
        
        
        with open(f'{json_file}/tmp.json', 'w', encoding='utf-8') as f:
            json.dump(complete_output['response'], f, ensure_ascii=False, indent=4)

        prompt = complete_output['response']['caption']['en']

        # ====================================================================
        # Stage 2: Graphical Background Generation
        # ====================================================================
        logger.info("=" * 60)
        logger.info("Stage 2: Graphical Background Generation")
        logger.info("=" * 60)

        args.device = device
        xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload) 
        
        classification_type = classification_dict[complete_output['response']['classification']]
        print(f'selected lora type: {classification_type}')
        args.lora_path = f'{args.lora_local_path}/{classification_type}-lora.safetensors'
        images = img_gen(prompt, args, xflux_pipeline)

        # Cleanup Stage 2 pipeline
        del xflux_pipeline
        torch.cuda.empty_cache()

        # Save generated image
        images_file = os.path.join(save_paths, 'background_images')
        os.makedirs(images_file, exist_ok=True)

        gen_image_path = f'{images_file}/tmp.png'
        images.save(gen_image_path)
        
        # ====================================================================
        # Stage 3: Unified Layout-Text Rendering
        # ====================================================================
        logger.info("=" * 60)
        logger.info("Stage 3: Unified Layout-Text Rendering")
        logger.info("=" * 60)

        generator = PosterHTMLGenerator(
            model_path=args.stage3_model_path,
            device=device,
            rank=0,
            world_size=1,
            local_rank=0,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        generator.model = generator.model.to(device)

        # Prepare input data for HTML generation
        json_data = complete_output['response']
        theme = json_data.get('theme', '')
        element = json_data.get('element', '')
        color = json_data.get('color', '')
        style = json_data.get('style', '')
        application = json_data.get('application', '')
        text_content = json_data.get('text', '')
        width = json_data.get('width', '')
        height = json_data.get('height', '')

        print(f'\n\n\n\n{gen_image_path}')
        input_data = {
            "path": gen_image_path,
            "width": width,
            "height": height,
            "theme": theme,
            "element": element,
            "color": color,
            "style": style,
            "application": application,
            "text": text_content
        }
        
        input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

        user_prompt = f"请分析图片并根据提供的海报基调和文案内容，生成一个HTML文件实现最佳视觉排版效果。\n```json\n{input_json_str}\n```"

        # Generate HTML
        html_content, response = generator.generate_html(images, user_prompt)

        # Cleanup Stage 3 model
        del generator
        torch.cuda.empty_cache()

        # Save HTML output
        html_file = os.path.join(save_paths, 'final_html')
        os.makedirs(html_file, exist_ok=True)

        with open(f'{html_file}/final.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    except Exception as e:
        print(f"Failed to process: {str(e)}")


def main(grade):
    args = create_argparser().parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_paths = f'./results/{timestamp}'
    process_batch(args.request_prompt, args, save_paths, args.stage1_model_path, grade)

   
if __name__ == "__main__":
    num = 0
    grade = ['simple', 'medium', 'hard']
    while True:
        main(grade[num % len(grade)])
        num += 1