#!/usr/bin/env python3
"""
Patch script for flash-linear-attention bitnet module
This script modifies the AutoConfig and AutoModel registration calls
to include exist_ok=True parameter to avoid conflicts.
"""

import os
import sys


def patch_fla_bitnet():

        
    # 尝试在系统Python路径中查找
    import site
    for package_path in site.getsitepackages():
        potential_path = os.path.join(package_path, 'fla', 'models', 'bitnet', '__init__.py')
        if os.path.exists(potential_path):
            print(f'Found path: {potential_path}')
            
            with open(potential_path, 'r') as f:
                content = f.read()
            
            # 检查是否已经修改过
            if 'exist_ok=True' in content:
                print('File already patched')
                return True
            
            # 进行替换
            # content = content.replace(
            #     'AutoConfig.register(ABCConfig.model_type, ABCConfig)',
            #     'AutoConfig.register(ABCConfig.model_type, ABCConfig, exist_ok=True)'
            # )
            # content = content.replace(
            #     'AutoModel.register(ABCConfig, ABCModel)',
            #     'AutoModel.register(ABCConfig, ABCModel, exist_ok=True)'
            # )
            # content = content.replace(
            #     'AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM)',
            #     'AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM, exist_ok=True)'
            # )
            content = content.replace(
                "AutoConfig.register(BitNetConfig.model_type, BitNetConfig)",
                "AutoConfig.register(BitNetConfig.model_type, BitNetConfig, exist_ok=True)"
            )
            content = content.replace(
                "AutoModel.register(BitNetConfig, BitNetModel)",
                "AutoModel.register(BitNetConfig, BitNetModel, exist_ok=True)",
            )
            content = content.replace(
                "AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM)",
                "AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)",
            )
            with open(potential_path, 'w') as f:
                f.write(content)
            
            print('Successfully modified alternative path')
            return True
    
    print('Error: Could not find flash-linear-attention installation')
    return False


if __name__ == '__main__':
    success = patch_fla_bitnet()
    sys.exit(0 if success else 1)