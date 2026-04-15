from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import re
from paddleocr import PaddleOCR

app = Flask(__name__)

print("加载 PaddleOCR 模型...")
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
    print("模型加载完成")
except Exception as e:
    print(f"模型加载失败: {e}")
    ocr = None

def parse_code(text):
    """提取字母+数字"""
    if not text:
        return None
    clean = text.upper().replace(' ', '').replace('O', '0').replace('I', '1')
    match = re.search(r'([A-Z]{1,2})(\d{1,3})', clean)
    return match.group(1) + match.group(2) if match else None

def detect_grid_universal(img):
    """
    通用网格检测：不依赖颜色，使用边缘和直线检测
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 方法1：边缘检测 + 霍夫直线
    edges = cv2.Canny(gray, 50, 150)
    
    # 检测所有直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=min(h,w)//10, maxLineGap=20)
    
    if lines is not None and len(lines) > 10:
        # 分类水平线和垂直线
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算角度
            if x2 == x1:  # 垂直线
                v_lines.append((x1 + x2) // 2)
            elif y2 == y1:  # 水平线
                h_lines.append((y1 + y2) // 2)
            else:
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 20:  # 接近水平
                    h_lines.append((y1 + y2) // 2)
                elif angle > 70:  # 接近垂直
                    v_lines.append((x1 + x2) // 2)
        
        # 去重聚类（合并相近的线）
        h_lines = cluster_lines(sorted(list(set(h_lines))), threshold=15)
        v_lines = cluster_lines(sorted(list(set(v_lines))), threshold=15)
        
        if len(h_lines) >= 3 and len(v_lines) >= 3:
            return create_grid_from_lines(h_lines, v_lines, h, w, method='edge_detect')
    
    # 方法2：形态学检测（适合虚线/点状线）
    grid_result = detect_by_morphology(gray, h, w)
    if grid_result:
        return grid_result
    
    # 方法3：投影分析（适合无明显线条但有规律排列）
    grid_result = detect_by_projection(gray, h, w)
    if grid_result:
        return grid_result
    
    # 方法4：均匀分割（保底方案）
    return create_uniform_grid(h, w, default_rows=52, default_cols=52)

def cluster_lines(lines, threshold=15):
    """聚类合并相近的线"""
    if not lines:
        return []
    
    clustered = []
    current_group = [lines[0]]
    
    for i in range(1, len(lines)):
        if lines[i] - lines[i-1] < threshold:
            current_group.append(lines[i])
        else:
            clustered.append(sum(current_group) // len(current_group))
            current_group = [lines[i]]
    
    clustered.append(sum(current_group) // len(current_group))
    return clustered

def detect_by_morphology(gray, h, w):
    """形态学方法检测网格结构"""
    # 自适应阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 水平结构元素
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    
    # 垂直结构元素
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    
    # 合并
    grid_mask = cv2.bitwise_or(horiz, vert)
    
    # 查找直线位置
    h_proj = np.sum(grid_mask, axis=1)
    v_proj = np.sum(grid_mask, axis=0)
    
    h_lines = find_line_positions(h_proj, threshold=np.max(h_proj)*0.3)
    v_lines = find_line_positions(v_proj, threshold=np.max(v_proj)*0.3)
    
    if len(h_lines) >= 3 and len(v_lines) >= 3:
        return create_grid_from_lines(h_lines, v_lines, h, w, method='morphology')
    
    return None

def find_line_positions(proj, threshold, min_gap=20):
    """从投影中找到线位置"""
    lines = []
    in_line = False
    line_start = 0
    
    for i, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            line_start = i
        elif val <= threshold and in_line:
            in_line = False
            line_end = i
            lines.append((line_start + line_end) // 2)
    
    # 合并相近的线
    if len(lines) > 1:
        lines = cluster_lines(lines, threshold=min_gap)
    
    return lines

def detect_by_projection(gray, h, w):
    """通过内容投影分析检测网格"""
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 投影
    h_proj = np.sum(binary, axis=1)
    v_proj = np.sum(binary, axis=0)
    
    # 检测周期性（网格的规律性）
    h_period = detect_period(h_proj)
    v_period = detect_period(v_proj)
    
    if h_period and v_period:
        rows = h // h_period
        cols = w // v_period
        return create_uniform_grid(h, w, rows, cols, method='projection')
    
    return None

def detect_period(data):
    """检测数据的周期性"""
    # 自相关分析找周期
    if len(data) < 100:
        return None
    
    # 简化：找主要频率
    fft = np.fft.fft(data - np.mean(data))
    freqs = np.fft.fftfreq(len(data))
    
    # 找最大幅度的正频率
    positive_idx = np.where(freqs > 0)[0]
    if len(positive_idx) == 0:
        return None
    
    main_freq_idx = positive_idx[np.argmax(np.abs(fft[positive_idx]))]
    period = int(1 / freqs[main_freq_idx])
    
    # 验证周期合理性
    if 10 < period < len(data) // 3:
        return period
    
    return None

def create_grid_from_lines(h_lines, v_lines, h, w, method='lines'):
    """从检测到的线创建网格"""
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1
    
    cells = []
    for i in range(rows):
        row = []
        for j in range(cols):
            x1 = max(0, v_lines[j])
            y1 = max(0, h_lines[i])
            x2 = min(w, v_lines[j + 1])
            y2 = min(h, h_lines[i + 1])
            row.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        cells.append(row)
    
    return {
        'rows': rows,
        'cols': cols,
        'cells': cells,
        'method': method,
        'h_lines': h_lines,
        'v_lines': v_lines
    }

def create_uniform_grid(h, w, rows=52, cols=52, method='uniform'):
    """均匀分割网格"""
    cell_h = h // rows
    cell_w = w // cols
    
    cells = []
    for i in range(rows):
        row = []
        for j in range(cols):
            x1 = j * cell_w
            y1 = i * cell_h
            x2 = (j + 1) * cell_w
            y2 = (i + 1) * cell_h
            row.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        cells.append(row)
    
    return {
        'rows': rows,
        'cols': cols,
        'cells': cells,
        'method': method
    }

@app.route('/')
def index():
    return "Grid OCR Service Running (Universal Detection)"

@app.route('/parse', methods=['POST'])
def parse_grid():
    try:
        data = request.get_json()
        image_b64 = data.get('image', '')
        rows_hint = data.get('rows')  # 可选：提示行数
        cols_hint = data.get('cols')  # 可选：提示列数
        
        if not image_b64:
            return jsonify({'success': False, 'error': '缺少图片数据'})
        
        # Base64 解码
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': '图片解码失败'})
        
        if ocr is None:
            return jsonify({'success': False, 'error': 'OCR 模型未加载'})
        
        # 自动检测网格
        grid_info = detect_grid_universal(img)
        
        # 如果提供了提示，覆盖检测结果
        if rows_hint and cols_hint:
            h, w = img.shape[:2]
            grid_info = create_uniform_grid(h, w, rows_hint, cols_hint, method='user_hint')
        
        rows = grid_info['rows']
        cols = grid_info['cols']
        cells = grid_info['cells']
        
        # 解析每个格子
        grid = []
        stats = {}
        
        for i in range(rows):
            row = []
            for j in range(cols):
                cell = cells[i][j]
                x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
                
                # 边距处理
                margin_x = int((x2 - x1) * 0.1)  # 10% 边距
                margin_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 + margin_x)
                y1 = max(0, y1 + margin_y)
                x2 = min(img.shape[1], x2 - margin_x)
                y2 = min(img.shape[0], y2 - margin_y)
                
                if x2 <= x1 or y2 <= y1:
                    row.append({'r': i+1, 'c': j+1, 'code': None})
                    continue
                
                cell_img = img[y1:y2, x1:x2]
                
                try:
                    result = ocr.ocr(cell_img, cls=True)
                    
                    code = None
                    if result and result[0]:
                        texts = [line[1][0] for line in result[0]]
                        code = parse_code(' '.join(texts))
                    
                    row.append({
                        'r': i+1, 
                        'c': j+1, 
                        'code': code,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    if code:
                        stats[code] = stats.get(code, 0) + 1
                        
                except Exception as cell_err:
                    row.append({
                        'r': i+1, 
                        'c': j+1, 
                        'code': None
                    })
            
            grid.append(row)
        
        return jsonify({
            'success': True,
            'grid_info': {
                'rows': rows,
                'cols': cols,
                'method': grid_info.get('method', 'unknown')
            },
            'grid': grid,
            'stats': stats,
            'total': rows * cols,
            'found': sum(stats.values())
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
