#!/usr/bin/env python3
"""
Convert dataset description from txt to markdown with embedded visualizations.
"""

from pathlib import Path

def convert_to_markdown():
    """Convert the dataset description to markdown format."""
    
    # Read the original file
    txt_file = Path("DeAn/mo_ta_du_lieu.txt")
    md_file = Path("DeAn/MO_TA_DU_LIEU.md")
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Start markdown content
    md_content = []
    
    # Add title and intro
    md_content.append("# Mô Tả Dữ Liệu (Dataset Description)\n\n")
    md_content.append("> **Bộ dữ liệu X-quang khớp gối** - Phân loại mức độ thoái hóa theo thang điểm Kellgren-Lawrence\n\n")
    md_content.append("---\n\n")
    
    # Process the content
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Main headers (with ==== or ----)
        if i + 1 < len(lines) and (lines[i+1].startswith('====') or lines[i+1].startswith('----')):
            # Determine heading level
            if lines[i+1].startswith('===='):
                md_content.append(f"\n## {line.strip()}\n\n")
            else:
                md_content.append(f"\n### {line.strip()}\n\n")
            i += 2  # Skip the underline
            continue
        
        # Tables (detect by ┌ or │)
        if line.strip().startswith('Bảng'):
            # Add table caption
            md_content.append(f"\n**{line.strip()}**\n\n")
            i += 1
            
            # Process table
            table_lines = []
            while i < len(lines) and ('│' in lines[i] or '┌' in lines[i] or '└' in lines[i] or '├' in lines[i]):
                table_lines.append(lines[i])
                i += 1
            
            # Convert to markdown table
            md_table = convert_ascii_table_to_markdown(table_lines)
            md_content.append(md_table + "\n\n")
            continue
        
        # Section markers
        if line.strip() == '---':
            md_content.append("\n---\n\n")
            i += 1
            continue
        
        # Regular content
        if line.strip():
            md_content.append(line + "\n")
        else:
            md_content.append("\n")
        
        i += 1
    
    # Add visualizations at appropriate sections
    final_content = '\n'.join(md_content)
    
    # Insert visualizations
    final_content = insert_visualizations(final_content)
    
    # Write markdown file
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"✅ Created: {md_file}")
    print(f"   Size: {md_file.stat().st_size} bytes")


def convert_ascii_table_to_markdown(table_lines):
    """Convert ASCII box table to markdown format."""
    
    # Extract rows (lines with │)
    data_rows = [line for line in table_lines if '│' in line and not any(c in line for c in ['┌', '└', '├'])]
    
    if not data_rows:
        return ""
    
    md_rows = []
    for row in data_rows:
        # Split by │ and clean
        cells = [cell.strip() for cell in row.split('│') if cell.strip()]
        md_rows.append('| ' + ' | '.join(cells) + ' |')
    
    # Add header separator after first row
    if md_rows:
        first_row_cell_count = md_rows[0].count('|') - 1
        separator = '| ' + ' | '.join(['---'] * first_row_cell_count) + ' |'
        md_rows.insert(1, separator)
    
    return '\n'.join(md_rows)


def insert_visualizations(content):
    """Insert visualization images at appropriate locations."""
    
    # Section 3.x.2 - Add bbox analysis after "Bảng 3.1"
    content = content.replace(
        "- Chiều cao thường trong khoảng 3072-4096 pixels (tỷ lệ ~1:3)",
        "- Chiều cao thường trong khoảng 3072-4096 pixels (tỷ lệ ~1:3)\n\n"
        "![Phân tích Bounding Box](./images/bbox_analysis.png)\n"
        "*Hình 1: Phân tích thống kê bounding box từ dataset gốc - Phân bố width, height, area, aspect ratio và vị trí center*"
    )
    
    # Section 3.x.3 - Add class distribution after class distribution table
    content = content.replace(
        "Class imbalance ratio: 1,368 / 102 = 13.63:1",
        "Class imbalance ratio: 1,368 / 102 = 13.63:1\n\n"
        "![Phân bố lớp và Class Imbalance](./images/class_imbalance.png)\n"
        "*Hình 2: Vấn đề mất cân bằng lớp trong dataset - KL2 chiếm ưu thế (43.3%) trong khi KL0 chỉ có 3.2%*"
    )
    
    # Section 3.x.4 - Add augmentation strategy in balancing section
    content = content.replace(
        "- Augmentation được áp dụng: 3,678 ảnh (54% tổng dataset sau balancing)",
        "- Augmentation được áp dụng: 3,678 ảnh (54% tổng dataset sau balancing)\n\n"
        "![Chiến lược Data Augmentation](./images/augmentation_strategy.png)\n"
        "*Hình 3: Chiến lược augmentation để cân bằng dữ liệu - Sử dụng flip transformations cho minority classes*"
    )
    
    # Add pipeline workflow at the beginning of pipeline section
    content = content.replace(
        "BƯỚC 1: PHÁT HIỆN VÀ CROP VÙNG KHỚP GỐI",
        "![Pipeline Workflow](./images/pipeline_workflow.png)\n"
        "*Hình 4: Luồng xử lý dữ liệu 5 bước từ ảnh gốc đến sẵn sàng huấn luyện*\n\n"
        "## BƯỚC 1: PHÁT HIỆN VÀ CROP VÙNG KHỚP GỐI"
    )
    
    return content


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONVERTING DATASET DESCRIPTION TO MARKDOWN")
    print("="*80 + "\n")
    
    convert_to_markdown()
    
    print(f"\n✅ Conversion complete!")
    print(f"\nGenerated files:")
    print(f"  - DeAn/MO_TA_DU_LIEU.md (main document)")
    print(f"  - DeAn/images/*.png (5 visualizations)")
    print(f"\nTo view: Open DeAn/MO_TA_DU_LIEU.md in a markdown viewer or IDE")
    print("="*80 + "\n")
