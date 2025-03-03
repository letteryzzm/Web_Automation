import os
import cv2
import numpy as np
import time
import uuid
import logging
import json
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bs4 import BeautifulSoup

# 使用OpenCV的DNN模块进行目标检测
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available. Text detection will be limited.")

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = "results"

# 确保上传和结果目录存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)


class WebElementDetector:
    """检测网页元素和区块的类"""

    def __init__(self):
        self.elements = []
        self.sections = []

    def detect_with_contours(self, image):
        """使用轮廓检测方法识别可能的UI元素"""
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 膨胀边缘使轮廓更连续
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 过滤掉太小的轮廓
        min_area = 100  # 最小面积阈值
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        ui_elements = []

        for i, contour in enumerate(filtered_contours):
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 忽略太大的边界框（可能是整个页面）
            if w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                continue

            # 提取感兴趣区域
            roi = image[y : y + h, x : x + w]

            # 判断元素类型（简化版）
            element_type = self._classify_element(roi, x, y, w, h)

            ui_elements.append(
                {
                    "id": i,
                    "type": element_type,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": 0.8,  # 简化的置信度
                    "text": self._extract_text(roi) if TESSERACT_AVAILABLE else "",
                }
            )

        self.elements = ui_elements
        return ui_elements

    def detect_sections(self, image):
        """检测页面的主要区块"""
        # 使用颜色分割识别不同区域
        # 转为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 使用均值偏移分割算法
        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 形态学操作闭合区域
        kernel = np.ones((25, 25), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 查找主要区块
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 根据大小过滤区块
        min_section_area = image.shape[0] * image.shape[1] * 0.01  # 至少占图像1%
        sections = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_section_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # 获取区块内容
            section_img = image[y : y + h, x : x + w]

            # 尝试识别区块类型
            section_type = self._classify_section(section_img)

            sections.append(
                {
                    "id": i,
                    "type": section_type,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area,
                    "elements": self._find_elements_in_section(x, y, w, h),
                }
            )

        self.sections = sections
        return sections

    def _classify_element(self, roi, x, y, w, h):
        """简单分类UI元素类型"""
        # 获取颜色特征
        avg_color = np.mean(roi, axis=(0, 1))

        # 形状特征
        aspect_ratio = w / h if h > 0 else 0

        # 根据特征简单分类
        if 0.9 <= aspect_ratio <= 1.1 and w < 50 and h < 50:
            return "button"  # 近似正方形的小元素可能是按钮
        elif aspect_ratio > 3 or aspect_ratio < 0.33:
            return "text_field"  # 细长的元素可能是文本框
        elif w > 100 and h < 50:
            return "navigation"  # 宽而短的可能是导航元素
        elif w > 150 and h > 100:
            return "content_block"  # 较大的块可能是内容区
        else:
            return "unknown"

    def _classify_section(self, section_img):
        """识别区块类型"""
        # 获取尺寸
        h, w = section_img.shape[:2]

        # 根据位置和尺寸特性判断
        if h < 100 and w > section_img.shape[1] * 0.7:
            return "header"
        elif (
            h < 100
            and w > section_img.shape[1] * 0.7
            and section_img.shape[0] - (h + 100) < 100
        ):
            return "footer"
        elif w < 200 and h > 300:
            return "sidebar"
        elif w > section_img.shape[1] * 0.6 and h > 300:
            return "main_content"
        else:
            return "generic_section"

    def _extract_text(self, roi):
        """从ROI中提取文本"""
        if not TESSERACT_AVAILABLE:
            return ""

        try:
            # 预处理以提高文本识别率
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # 使用Tesseract OCR
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            logger.error(f"文本提取错误: {e}")
            return ""

    def _find_elements_in_section(self, section_x, section_y, section_w, section_h):
        """查找区块内的元素"""
        elements_in_section = []

        for element in self.elements:
            # 检查元素中心点是否在区块内
            element_center_x = element["x"] + element["width"] / 2
            element_center_y = element["y"] + element["height"] / 2

            if (
                section_x <= element_center_x <= section_x + section_w
                and section_y <= element_center_y <= section_y + section_h
            ):
                elements_in_section.append(element["id"])

        return elements_in_section

    def detect_with_template_matching(
        self, image, templates_folder="element_templates"
    ):
        """使用模板匹配检测常见UI元素"""
        if not os.path.exists(templates_folder):
            logger.warning(f"模板文件夹 {templates_folder} 不存在")
            return []

        ui_elements = []
        element_id = 0

        # 遍历模板
        for template_file in os.listdir(templates_folder):
            template_path = os.path.join(templates_folder, template_file)

            # 确定元素类型
            element_type = template_file.split("_")[0].lower()

            # 读取模板
            template = cv2.imread(template_path)
            if template is None:
                continue

            # 模板匹配
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

            # 设置阈值
            threshold = 0.7
            loc = np.where(result >= threshold)

            # 添加检测到的元素
            for pt in zip(*loc[::-1]):
                x, y = pt
                w, h = template.shape[1], template.shape[0]

                # 检查是否与已有元素重叠
                overlap = False
                for element in ui_elements:
                    if self._check_overlap(
                        x,
                        y,
                        w,
                        h,
                        element["x"],
                        element["y"],
                        element["width"],
                        element["height"],
                    ):
                        overlap = True
                        break

                if not overlap:
                    ui_elements.append(
                        {
                            "id": element_id,
                            "type": element_type,
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "confidence": result[y, x],
                            "text": "",
                        }
                    )
                    element_id += 1

        return ui_elements

    def _check_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """检查两个矩形是否重叠"""
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def mark_elements_on_image(self, image):
        """在图像上标记检测到的元素"""
        marked_image = image.copy()

        # 标记各个元素
        for element in self.elements:
            x, y, w, h = element["x"], element["y"], element["width"], element["height"]

            # 根据元素类型选择颜色
            color = (0, 0, 255)  # 默认红色
            if element["type"] == "button":
                color = (0, 0, 255)  # 红色
            elif element["type"] == "text_field":
                color = (0, 255, 0)  # 绿色
            elif element["type"] == "navigation":
                color = (255, 0, 0)  # 蓝色
            elif element["type"] == "content_block":
                color = (255, 255, 0)  # 青色

            # 绘制矩形
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), color, 2)

            # 添加类型标签
            cv2.putText(
                marked_image,
                element["type"],
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return marked_image

    def mark_sections_on_image(self, image):
        """在图像上标记检测到的区块"""
        marked_image = image.copy()

        # 标记各个区块
        for section in self.sections:
            x, y, w, h = section["x"], section["y"], section["width"], section["height"]

            # 根据区块类型选择颜色
            color = (255, 0, 0)  # 默认蓝色
            if section["type"] == "header":
                color = (255, 0, 255)  # 紫色
            elif section["type"] == "footer":
                color = (255, 255, 0)  # 青色
            elif section["type"] == "sidebar":
                color = (0, 255, 255)  # 黄色
            elif section["type"] == "main_content":
                color = (0, 165, 255)  # 橙色

            # 绘制虚线矩形
            for i in range(0, h, 10):
                cv2.line(marked_image, (x, y + i), (x + 10, y + i), color, 2)
                cv2.line(marked_image, (x + w - 10, y + i), (x + w, y + i), color, 2)

            for i in range(0, w, 10):
                cv2.line(marked_image, (x + i, y), (x + i, y + 10), color, 2)
                cv2.line(marked_image, (x + i, y + h - 10), (x + i, y + h), color, 2)

            # 添加类型标签
            cv2.putText(
                marked_image,
                section["type"],
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return marked_image


# Flask路由
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.route("/api/analyze", methods=["POST"])
def analyze_screenshot():
    """处理上传的截图并进行分析"""
    # 检查是否有文件
    if "screenshot" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["screenshot"]

    # 检查文件名
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # 生成唯一文件名
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4().hex)[:8]
    filename = f"{timestamp}_{unique_id}_{file.filename}"

    # 保存上传的文件
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # 读取图像并进行处理
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "Cannot read image file"}), 400

    # 初始化检测器
    detector = WebElementDetector()

    # 检测UI元素
    try:
        ui_elements = detector.detect_with_contours(image)
        sections = detector.detect_sections(image)

        # 在图像上标记元素
        marked_image = detector.mark_elements_on_image(image)

        # 在图像上标记区块
        marked_image = detector.mark_sections_on_image(marked_image)

        # 保存结果图像
        result_filename = f"analyzed_{filename}"
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
        cv2.imwrite(result_path, marked_image)

        # 准备响应
        response = {
            "original_image": f"/uploads/{filename}",
            "analyzed_image": f"/results/{result_filename}",
            "elements": ui_elements,
            "sections": sections,
            "stats": {
                "element_count": len(ui_elements),
                "section_count": len(sections),
            },
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"分析出错: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/dom", methods=["POST"])
def analyze_dom():
    """分析上传的HTML文档"""
    # 检查是否有文件
    if "html_file" not in request.files:
        return jsonify({"error": "No HTML file uploaded"}), 400

    file = request.files["html_file"]

    # 检查文件名
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # 读取HTML内容
        content = file.read().decode("utf-8")

        # 解析HTML
        soup = BeautifulSoup(content, "html.parser")

        # 提取关键元素信息
        elements = []

        # 查找所有可交互元素
        for i, tag in enumerate(
            soup.find_all(["button", "a", "input", "select", "textarea"])
        ):
            element_type = tag.name
            element_id = tag.get("id", "")
            element_class = " ".join(tag.get("class", []))
            element_text = tag.get_text().strip()

            elements.append(
                {
                    "id": i,
                    "type": element_type,
                    "element_id": element_id,
                    "class": element_class,
                    "text": element_text[:50]
                    + ("..." if len(element_text) > 50 else ""),
                    "attributes": {
                        k: v for k, v in tag.attrs.items() if k not in ["id", "class"]
                    },
                }
            )

        # 查找主要区块
        sections = []
        section_tags = [
            "div",
            "section",
            "article",
            "nav",
            "header",
            "footer",
            "main",
            "aside",
        ]

        for i, tag in enumerate(soup.find_all(section_tags)):
            # 跳过无内容或太小的区块
            if len(tag.get_text().strip()) < 20:
                continue

            # 确定区块类型
            section_type = "generic_section"
            if (
                tag.name == "header"
                or tag.get("id") == "header"
                or "header" in tag.get("class", [])
            ):
                section_type = "header"
            elif (
                tag.name == "footer"
                or tag.get("id") == "footer"
                or "footer" in tag.get("class", [])
            ):
                section_type = "footer"
            elif (
                tag.name == "nav"
                or tag.get("id") == "nav"
                or "nav" in tag.get("class", [])
            ):
                section_type = "navigation"
            elif (
                tag.name == "aside"
                or tag.get("id") == "sidebar"
                or "sidebar" in tag.get("class", [])
            ):
                section_type = "sidebar"
            elif (
                tag.name == "main"
                or tag.get("id") == "main"
                or "main" in tag.get("class", [])
            ):
                section_type = "main_content"

            sections.append(
                {
                    "id": i,
                    "type": section_type,
                    "element_id": tag.get("id", ""),
                    "class": " ".join(tag.get("class", [])),
                    "text_length": len(tag.get_text()),
                    "children_count": len(tag.find_all()),
                }
            )

        return jsonify(
            {
                "elements": elements,
                "sections": sections,
                "stats": {
                    "element_count": len(elements),
                    "section_count": len(sections),
                },
            }
        )

    except Exception as e:
        logger.error(f"DOM分析出错: {str(e)}")
        return jsonify({"error": f"DOM analysis failed: {str(e)}"}), 500


# 如果需要直接上传图片的端点
@app.route("/api/analyze_image", methods=["POST"])
def analyze_image_data():
    """分析Base64编码的图像"""
    try:
        data = request.json
        if "image_data" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # 解码Base64图像
        image_data = data["image_data"]
        if image_data.startswith("data:image"):
            # 从数据URI中提取Base64部分
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Cannot decode image data"}), 400

        # 保存原始图像
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4().hex)[:8]
        filename = f"{timestamp}_{unique_id}_uploaded.png"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        cv2.imwrite(file_path, image)

        # 初始化检测器
        detector = WebElementDetector()

        # 检测UI元素和区块
        ui_elements = detector.detect_with_contours(image)
        sections = detector.detect_sections(image)

        # 在图像上标记元素和区块
        marked_image = detector.mark_elements_on_image(image)
        marked_image = detector.mark_sections_on_image(marked_image)

        # 保存结果图像
        result_filename = f"analyzed_{filename}"
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
        cv2.imwrite(result_path, marked_image)

        # 将结果图像编码为Base64
        _, buffer = cv2.imencode(".png", marked_image)
        marked_image_base64 = base64.b64encode(buffer).decode("utf-8")

        # 准备响应
        response = {
            "original_image": f"/uploads/{filename}",
            "analyzed_image": f"/results/{result_filename}",
            "analyzed_image_data": f"data:image/png;base64,{marked_image_base64}",
            "elements": ui_elements,
            "sections": sections,
            "stats": {
                "element_count": len(ui_elements),
                "section_count": len(sections),
            },
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"图像分析出错: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# 主函数
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
