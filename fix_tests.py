#!/usr/bin/env python3
import re
import os
import glob

def fix_tensor_literals(content):
    """Fix floating point literals in Tensor::from_slice calls to use f32"""
    
    # Pattern to match Tensor::from_slice with array literals
    pattern = r'(Tensor::from_slice\(&\[)((?:[^]]*?))(], &\[[^\]]+\]\)\.unwrap\(\))'
    
    def replace_literals(match):
        prefix = match.group(1)
        array_content = match.group(2)
        suffix = match.group(3)
        
        # Add f32 to first number if not already present
        # Match numbers like: 0.0, -1.5, 123, etc.
        array_content = re.sub(
            r'(?<![f\d])(\s*-?\d+\.?\d*)(?![f\d])',
            r'\1f32',
            array_content,
            count=1
        )
        
        return prefix + array_content + suffix
    
    return re.sub(pattern, replace_literals, content, flags=re.DOTALL)

def fix_data_access(content):
    """Fix .data_f32() calls to use .storage().as_slice::<f32>()"""
    content = re.sub(r'\.data_f32\(\)', '.storage().as_slice::<f32>()', content)
    return content

def process_file(filepath):
    """Process a single Rust file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    content = fix_tensor_literals(content)
    content = fix_data_access(content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    # Find all Rust files in ghostflow-ml/src
    pattern = 'ghostflow-ml/src/**/*.rs'
    files = glob.glob(pattern, recursive=True)
    
    fixed_count = 0
    for filepath in files:
        if process_file(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == '__main__':
    main()
