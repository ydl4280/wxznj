from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import re
from paddleocr import PaddleOCR

app = Flask(__name__)

# 初始化 OCR（启动时加载）
print("加载 PaddleOCR 模型...")
try:
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        show_log=False,
        use_gpu=False
    )
    print("模型加载完成")
except Exception as e:
    print(f"模型加载失败: {e}")
    ocr = None

# 网格配置（根据你的图片校准）
GRID = {
    'rows': 52,
    'cols': 52,
    'start_x': 45,      # 第1列左边缘像素
    'start_y': 75,      # 第1行上边缘像素
    'cell_w': 11,       # 格子宽度
    'cell_h': 11        # 格子高度
}

def parse_code(text):
    """提取字母+数字，如 H7, A25"""
    if not text:
        return None
    clean = text.upper().replace(' ', '').replace('O', '0').replace('I', '1')
    match = re.search(r'([A-Z]{1,2})(\d{1,3})', clean)
    return match.group(1) + match.group(2) if match else None

@app.route('/')
def index():
    return "Grid OCR Service Running"

@app.route('/parse', methods=['POST'])
def parse_grid():
    try:
        data = request.get_json()
        image_b64 = data.get('image', '')
        
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
        
        # 解析网格
        grid = []
        stats = {}
        
        for r in range(GRID['rows']):
            row = []
            for c in range(GRID['cols']):
                # 计算坐标
                x1 = GRID['start_x'] + c * GRID['cell_w']
                y1 = GRID['start_y'] + r * GRID['cell_h']
                x2 = x1 + GRID['cell_w']
                y2 = y1 + GRID['cell_h']
                
                # 边界检查
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    row.append({'r': r+1, 'c': c+1, 'code': None})
                    continue
                
                # 裁剪并识别
                cell = img[y1:y2, x1:x2]
                
                try:
                    result = ocr.ocr(cell, cls=True)
                    
                    code = None
                    if result and result[0]:
                        texts = [line[1][0] for line in result[0]]
                        code = parse_code(' '.join(texts))
                    
                    row.append({'r': r+1, 'c': c+1, 'code': code})
                    
                    if code:
                        stats[code] = stats.get(code, 0) + 1
                        
                except Exception as cell_err:
                    row.append({'r': r+1, 'c': c+1, 'code': None, 'error': str(cell_err)})
            
            grid.append(row)
        
        return jsonify({
            'success': True,
            'grid': grid,
            'stats': stats,
            'total': GRID['rows'] * GRID['cols'],
            'found': sum(stats.values())
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)