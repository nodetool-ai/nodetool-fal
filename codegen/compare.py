#!/usr/bin/env python3
"""
Compare generated nodes with existing nodes.

This script helps identify semantic differences between generated and manually written nodes.
"""

import sys
from pathlib import Path
from difflib import unified_diff


def normalize_code(code: str) -> list[str]:
    """Normalize code for comparison by removing whitespace variations."""
    lines = code.split("\n")
    normalized = []
    
    for line in lines:
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        
        # Normalize whitespace
        normalized.append(" ".join(stripped.split()))
    
    return normalized


def compare_nodes(generated_file: Path, existing_file: Path, class_name: str):
    """
    Compare a specific node class between generated and existing files.
    
    Args:
        generated_file: Path to generated file
        existing_file: Path to existing file
        class_name: Name of the class to compare
    """
    # Read files
    generated_code = generated_file.read_text()
    existing_code = existing_file.read_text()
    
    # Extract class definitions
    def extract_class(code: str, class_name: str) -> str:
        lines = code.split("\n")
        class_lines = []
        in_class = False
        indent_level = 0
        
        for line in lines:
            if f"class {class_name}" in line:
                in_class = True
                indent_level = len(line) - len(line.lstrip())
                class_lines.append(line)
            elif in_class:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= indent_level:
                    # End of class
                    break
                class_lines.append(line)
        
        return "\n".join(class_lines)
    
    generated_class = extract_class(generated_code, class_name)
    existing_class = extract_class(existing_code, class_name)
    
    if not generated_class:
        print(f"❌ Class {class_name} not found in generated file")
        return False
    
    if not existing_class:
        print(f"❌ Class {class_name} not found in existing file")
        return False
    
    # Normalize and compare
    gen_normalized = normalize_code(generated_class)
    exist_normalized = normalize_code(existing_class)
    
    if gen_normalized == exist_normalized:
        print(f"✅ {class_name}: IDENTICAL (semantically)")
        return True
    else:
        print(f"⚠️  {class_name}: DIFFERENCES FOUND")
        print("\nDiff:")
        diff = unified_diff(
            exist_normalized,
            gen_normalized,
            fromfile=f"existing/{existing_file.name}",
            tofile=f"generated/{generated_file.name}",
            lineterm=""
        )
        for line in diff:
            print(line)
        print()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 4:
        print("Usage: python compare.py <generated_file> <existing_file> <class_name>")
        print()
        print("Example:")
        print("  python codegen/compare.py \\")
        print("    generated/image_to_video_generated.py \\")
        print("    src/nodetool/nodes/fal/image_to_video.py \\")
        print("    PixverseV56ImageToVideo")
        sys.exit(1)
    
    generated_file = Path(sys.argv[1])
    existing_file = Path(sys.argv[2])
    class_name = sys.argv[3]
    
    if not generated_file.exists():
        print(f"❌ Generated file not found: {generated_file}")
        sys.exit(1)
    
    if not existing_file.exists():
        print(f"❌ Existing file not found: {existing_file}")
        sys.exit(1)
    
    print(f"Comparing {class_name}...")
    print(f"  Generated: {generated_file}")
    print(f"  Existing:  {existing_file}")
    print()
    
    success = compare_nodes(generated_file, existing_file, class_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
