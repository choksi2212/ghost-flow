#!/usr/bin/env python3
import re
import os
import glob

def fix_test_functions(content):
    """Fix only test functions, not production code"""
    
    # Find all test functions
    test_pattern = r'(#\[test\]\s*fn\s+\w+\s*\(\)\s*\{)(.*?)(\n\s*\})'
    
    def fix_test(match):
        prefix = match.group(1)
        test_body = match.group(2)
        suffix = match.group(3)
        
        # Only fix Tensor::from_slice in tests
        # Match: Tensor::from_slice(&[0.0, 1.0, ...
        # Replace first number with f32 suffix
        test_body = re.sub(
            r'Tensor::from_slice\(&\[\s*(-?\d+\.?\d*)',
            r'Tensor::from_slice(&[\1f32',
            test_body
        )
        
        # Fix .data_f32() calls
        test_body = test_body.replace('.data_f32()', '.storage().as_slice::<f32>()')
        
        return prefix + test_body + suffix
    
    return re.sub(test_pattern, fix_test, content, flags=re.DOTALL)

def process_file(filepath):
    """Process a single Rust file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    original = content
    content = fix_test_functions(content)
    
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
