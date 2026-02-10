#!/usr/bin/env python3
"""
Automatic configuration generator that creates Python config entries
for batches of endpoints to reach coverage goals.
"""

import json
import re
from pathlib import Path


def make_class_name(endpoint_id: str) -> str:
    """Generate a PascalCase class name from endpoint ID."""
    # Remove fal-ai/ prefix
    name = endpoint_id.replace('fal-ai/', '')
    
    # Split by / and -
    parts = name.replace('/', '-').split('-')
    
    # Convert to PascalCase
    class_name = ''.join(word.title() for word in parts if word)
    
    # Fix common naming patterns
    class_name = class_name.replace('V', 'V')  # Keep version numbers
    class_name = re.sub(r'([a-z])(\d)', r'\1\2', class_name)  # Fix version spacing
    
    return class_name


def generate_config_entry(endpoint_id: str, model_info: dict, category: str) -> str:
    """Generate a Python configuration entry for an endpoint."""
    
    class_name = make_class_name(endpoint_id)
    title = model_info['title']
    description = model_info.get('shortDescription', title)
    
    # Generate appropriate tags based on category
    tag_map = {
        'image-to-image': ['editing', 'transformation', 'image-to-image', 'img2img'],
        'text-to-image': ['generation', 'text-to-image', 'txt2img', 'ai-art'],
        'image-to-video': ['video', 'animation', 'image-to-video', 'img2vid'],
        'text-to-video': ['video', 'generation', 'text-to-video', 'txt2vid'],
        'video-to-video': ['video', 'editing', 'video-to-video', 'vid2vid'],
        'training': ['training', 'fine-tuning', 'lora', 'model-training'],
        'text-to-audio': ['audio', 'generation', 'text-to-audio', 'tts'],
        'audio-to-audio': ['audio', 'processing', 'audio-to-audio', 'transformation'],
        'text-to-speech': ['speech', 'synthesis', 'text-to-speech', 'tts'],
        'vision': ['vision', 'analysis', 'image-understanding', 'detection'],
        'image-to-3d': ['3d', 'generation', 'image-to-3d', 'modeling'],
        'audio-to-video': ['video', 'generation', 'audio-to-video', 'visualization'],
    }
    
    tags = tag_map.get(category, [category.replace('-', '_')])
    
    # Add model-specific tags from endpoint ID
    endpoint_lower = endpoint_id.lower()
    if 'flux' in endpoint_lower:
        tags.insert(0, 'flux')
    if 'lora' in endpoint_lower:
        if 'lora' not in tags:
            tags.append('lora')
    if 'turbo' in endpoint_lower or 'fast' in endpoint_lower:
        if 'fast' not in tags:
            tags.append('fast')
    if 'pro' in endpoint_lower or 'max' in endpoint_lower:
        if 'professional' not in tags:
            tags.append('professional')
    
    # Keep only first 6 tags
    tags = tags[:6]
    
    # Generate use cases based on category
    use_case_templates = {
        'image-to-image': [
            'Professional photo editing and enhancement',
            'Creative image transformations',
            'Batch image processing workflows',
            'Product photography refinement',
            'Automated image optimization',
        ],
        'text-to-image': [
            'AI-powered art generation',
            'Marketing and advertising visuals',
            'Concept art and ideation',
            'Social media content creation',
            'Rapid prototyping and mockups',
        ],
        'image-to-video': [
            'Animate static images',
            'Create engaging social media content',
            'Product demonstrations',
            'Marketing and promotional videos',
            'Visual storytelling',
        ],
        'text-to-video': [
            'AI-generated video content',
            'Marketing and advertising videos',
            'Educational content creation',
            'Social media video posts',
            'Automated video production',
        ],
        'video-to-video': [
            'Video style transfer',
            'Video enhancement and restoration',
            'Automated video editing',
            'Special effects generation',
            'Content repurposing',
        ],
    }
    
    use_cases = use_case_templates.get(category, [
        'Automated content generation',
        'Creative workflows',
        'Batch processing',
        'Professional applications',
        'Rapid prototyping',
    ])
    
    # Format the configuration entry
    config = f'''    "{endpoint_id}": {{
        "class_name": "{class_name}",
        "docstring": "{description}",
        "tags": {json.dumps(tags)},
        "use_cases": {json.dumps(use_cases, indent=12)[:-1]}
        ],
    }},'''
    
    return config


def main():
    """Generate configurations for all needed endpoints."""
    
    # Load the plan
    with open('/tmp/50_percent_coverage_plan.json', 'r') as f:
        plan = json.load(f)
    
    print("=== Auto-Config Generator for 50% Coverage ===\n")
    print(f"Need to add {plan['summary']['need_to_add']} configurations total\n")
    
    # Group by module/category
    module_map = {
        'image-to-image': 'image_to_image',
        'text-to-image': 'text_to_image',
        'image-to-video': 'image_to_video',
        'text-to-video': 'text_to_video',
        'video-to-video': 'video_to_video',
        'training': 'training',
        'text-to-audio': 'text_to_audio',
        'audio-to-audio': 'audio_to_audio',
        'text-to-speech': 'text_to_speech',
        'vision': 'vision',
        'image-to-3d': 'image_to_3d',
        'audio-to-video': 'audio_to_video',
    }
    
    output_dir = Path('/tmp/generated_configs')
    output_dir.mkdir(exist_ok=True)
    
    for category, info in plan['by_category'].items():
        if info['needed'] <= 0:
            continue
        
        module_name = module_map.get(category, category.replace('-', '_'))
        print(f"\n{'=' * 70}")
        print(f"Category: {category}")
        print(f"Need: {info['needed']} configs")
        print(f"Module: {module_name}")
        print('=' * 70)
        
        # Generate configs for this category
        configs = []
        candidates = info['top_candidates'][:info['needed']]
        
        for i, candidate in enumerate(candidates, 1):
            endpoint_id = candidate['id']
            print(f"{i:3}. {endpoint_id}")
            
            config_entry = generate_config_entry(endpoint_id, candidate, category)
            configs.append(config_entry)
        
        # Write to file
        output_file = output_dir / f'{module_name}_additions.txt'
        with open(output_file, 'w') as f:
            f.write(f"# Additions for {module_name}.py\n")
            f.write(f"# Add these {len(configs)} configs to reach 50% coverage\n\n")
            f.write('\n'.join(configs))
        
        print(f"✅ Generated {len(configs)} configs -> {output_file}")
    
    print(f"\n{'=' * 70}")
    print(f"✅ All config additions saved to {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review generated configs in {output_dir}")
    print(f"2. Copy relevant additions to codegen/configs/*.py")
    print(f"3. Run: python codegen/generate.py --module <name> --output-dir src/nodetool/nodes/fal")
    print(f"4. Run: nodetool package scan")
    print(f"5. Run: nodetool codegen")


if __name__ == '__main__':
    main()
