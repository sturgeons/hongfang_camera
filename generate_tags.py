"""
生成 AprilTag / ArUco 标签用于打印
运行后会在 tags 文件夹生成可打印的标签图片
"""
import cv2
import numpy as np
import os

def generate_apriltag(tag_id, size=200, border=50):
    """生成 AprilTag 36h11 标签"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size)
    
    # 添加白色边框（便于识别）
    bordered = cv2.copyMakeBorder(tag, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=255)
    
    # 添加ID标注
    h, w = bordered.shape
    result = np.ones((h + 40, w), dtype=np.uint8) * 255
    result[:h, :] = bordered
    cv2.putText(result, f"AprilTag ID: {tag_id}", (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    return result

def generate_aruco(tag_id, size=200, border=50):
    """生成 ArUco 6x6 标签"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    tag = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size)
    
    bordered = cv2.copyMakeBorder(tag, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=255)
    
    h, w = bordered.shape
    result = np.ones((h + 40, w), dtype=np.uint8) * 255
    result[:h, :] = bordered
    cv2.putText(result, f"ArUco ID: {tag_id}", (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    return result

def generate_color_card(color_id, color_name, bgr_color, size=200):
    """生成彩色标记卡片"""
    # 创建彩色方块
    card = np.ones((size, size, 3), dtype=np.uint8)
    card[:] = bgr_color
    
    # 添加白色边框
    border = 30
    bordered = cv2.copyMakeBorder(card, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    # 添加黑色细边框
    cv2.rectangle(bordered, (border-2, border-2), 
                  (bordered.shape[1]-border+2, bordered.shape[0]-border+2), 
                  (0, 0, 0), 2)
    
    # 添加标注
    h, w = bordered.shape[:2]
    result = np.ones((h + 50, w, 3), dtype=np.uint8) * 255
    result[:h, :] = bordered
    cv2.putText(result, f"ID:{color_id} {color_name}", (10, h + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return result

def generate_print_page(tags, cols=5, rows=5):
    """生成打印页面，每页 cols x rows 个标签"""
    pages = []
    total = len(tags)
    per_page = cols * rows
    
    for page_idx in range(0, total, per_page):
        page_tags = tags[page_idx:page_idx + per_page]
        
        # 补齐不足的位置（用白色填充）
        if len(page_tags) < per_page:
            h, w = page_tags[0].shape[:2]
            white = np.ones((h, w), dtype=np.uint8) * 255
            page_tags.extend([white] * (per_page - len(page_tags)))
        
        # 排列成网格
        rows_list = []
        for r in range(rows):
            start = r * cols
            row = np.hstack(page_tags[start:start + cols])
            rows_list.append(row)
        
        page = np.vstack(rows_list)
        pages.append(page)
    
    return pages

def main():
    # 创建输出目录
    os.makedirs("tags/apriltag", exist_ok=True)
    os.makedirs("tags/aruco", exist_ok=True)
    os.makedirs("tags/color", exist_ok=True)
    os.makedirs("tags/print_pages", exist_ok=True)
    
    # 生成数量
    NUM_TAGS = 1000
    
    print(f"生成 {NUM_TAGS} 个标签中...")
    
    # 生成 ArUco 6x6_1000 (ID 0-999)
    print(f"\n生成 ArUco 6x6_1000 (ID 0-{NUM_TAGS-1})...")
    aruco_tags = []
    for i in range(NUM_TAGS):
        tag = generate_aruco(i, size=200, border=40)
        cv2.imwrite(f"tags/aruco/aruco_{i}.png", tag)
        aruco_tags.append(tag)
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{NUM_TAGS} 个")
    
    # 生成 AprilTag 36h11 (最多587个，因为字典限制)
    # 如果需要更多，可以用 DICT_APRILTAG_16h5 (最多30个) 或其他字典组合
    apriltag_max = 587  # AprilTag 36h11 字典最大ID
    print(f"\n生成 AprilTag 36h11 (ID 0-{min(NUM_TAGS, apriltag_max)-1})...")
    apriltag_tags = []
    for i in range(min(NUM_TAGS, apriltag_max)):
        tag = generate_apriltag(i, size=200, border=40)
        cv2.imwrite(f"tags/apriltag/apriltag_{i}.png", tag)
        apriltag_tags.append(tag)
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{min(NUM_TAGS, apriltag_max)} 个")
    
    # 生成打印页面 (每页 5x5 = 25 个标签)
    print("\n生成打印页面 (每页25个标签)...")
    
    # ArUco 打印页
    aruco_pages = generate_print_page(aruco_tags, cols=5, rows=5)
    for idx, page in enumerate(aruco_pages):
        cv2.imwrite(f"tags/print_pages/aruco_page_{idx+1}.png", page)
    print(f"  ArUco: {len(aruco_pages)} 页")
    
    # AprilTag 打印页
    apriltag_pages = generate_print_page(apriltag_tags, cols=5, rows=5)
    for idx, page in enumerate(apriltag_pages):
        cv2.imwrite(f"tags/print_pages/apriltag_page_{idx+1}.png", page)
    print(f"  AprilTag: {len(apriltag_pages)} 页")
    
    print(f"\n✅ 完成！")
    print(f"\n生成统计:")
    print(f"  - ArUco 标签: {NUM_TAGS} 个 (ID 0-{NUM_TAGS-1})")
    print(f"  - AprilTag 标签: {min(NUM_TAGS, apriltag_max)} 个 (ID 0-{min(NUM_TAGS, apriltag_max)-1})")
    print(f"  - ArUco 打印页: {len(aruco_pages)} 页")
    print(f"  - AprilTag 打印页: {len(apriltag_pages)} 页")
    print(f"\n文件位置:")
    print(f"  - 单个标签: tags/aruco/, tags/apriltag/")
    print(f"  - 打印页面: tags/print_pages/")
    print(f"\n打印建议: A4纸打印，每页25个标签，实际尺寸约4cm x 4cm")

if __name__ == "__main__":
    main()

