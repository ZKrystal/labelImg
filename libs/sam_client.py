"""
SAM 交互式分割客户端 — 基于 ultralytics，支持 SAM1/SAM2/SAM3

用法:
    client = SamClient()
    client.load_model("weights/sam3.pt")  # 自动识别版本
    client.set_image(image_bgr)
    mask, bbox = client.predict_point(x, y)        # 点提示 (全版本)
    mask = client.predict_box(x1, y1, x2, y2)       # 框提示 (全版本)
    results = client.predict_text(["river","water"]) # 文字提示 (仅SAM3)
"""

import numpy as np
import cv2
import os


class SamClient:
    """SAM 客户端，支持 SAM1 / SAM2 / SAM3"""

    def __init__(self):
        self.sam_model = None            # ultralytics.SAM 实例 (点/框)
        self.semantic_predictor = None   # SAM3SemanticPredictor (文字，仅SAM3)
        self.current_image = None
        self._model_type = None          # "sam1" | "sam2" | "sam3" | None

    def load_model(self, checkpoint_path):
        """加载 SAM 模型，自动识别版本"""
        import torch
        from ultralytics import SAM

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 清理旧模型
        if self.sam_model is not None:
            del self.sam_model
        if self.semantic_predictor is not None:
            del self.semantic_predictor
        self.sam_model = None
        self.semantic_predictor = None

        # 点/框提示 — 全版本共用
        self.sam_model = SAM(checkpoint_path)
        self.sam_model.to(device)
        self._last_features = None
        self._last_image_path = None

        # 检测版本
        model_lower = os.path.basename(checkpoint_path).lower()
        if "sam3" in model_lower:
            self._model_type = "sam3"
            self._init_semantic(checkpoint_path, device)
        elif "sam2" in model_lower or "hiera" in model_lower:
            self._model_type = "sam2"
        else:
            self._model_type = "sam1"

    def _init_semantic(self, checkpoint_path, device):
        """尝试加载 SAM3 文字提示接口"""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
            overrides = dict(
                conf=0.25, task="segment", mode="predict",
                model=checkpoint_path, device=device,
                save=False, verbose=False,
            )
            self.semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
            self._patch_predictor_for_speed()
            self._patch_base_predictor()
        except Exception:
            self.semantic_predictor = None

    @property
    def is_loaded(self):
        return self.sam_model is not None

    @property
    def model_type(self):
        return self._model_type

    @property
    def supports_text(self):
        return self._model_type == "sam3" and self.semantic_predictor is not None

    def _patch_predictor_for_speed(self):
        if self.semantic_predictor is None:
            return
        import types
        from ultralytics.models.sam.predict import SAM2Predictor
        orig = self.semantic_predictor._inference_features
        def _fast_infer(self, features, bboxes=None, labels=None, text=None):
            if bboxes is None and text is None:
                return SAM2Predictor._inference_features(self, features, None, None, None, False, -1)
            return orig(features, bboxes, labels, text)
        self.semantic_predictor._inference_features = types.MethodType(_fast_infer, self.semantic_predictor)


    def _patch_base_predictor(self):
        import types
        from ultralytics.models.sam.predict import SAM2Predictor
        try:
            pred = self.sam_model.predictor
        except Exception:
            return
        orig = getattr(pred, "_inference_features", None)
        if orig is None:
            return
        def _fast(inner_self, features, bboxes=None, labels=None, text=None):
            if bboxes is None and text is None:
                return SAM2Predictor._inference_features(inner_self, features, None, None, None, False, -1)
            return orig(features, bboxes, labels, text)
        pred._inference_features = types.MethodType(_fast, pred)

    def set_image(self, image_bgr):
        """设置当前图片并提取特征"""
        self.current_image = image_bgr
        if self.semantic_predictor:
            self.semantic_predictor.set_image(image_bgr)

    def _masks_to_original(self, result):
        """从 Results 中提取原始分辨率 (mask, bbox)，mask 保持原图尺寸"""
        masks_obj = result.masks
        if masks_obj is None:
            return []

        poly_list = list(masks_obj.xy) if (hasattr(masks_obj, "xy")
                                            and masks_obj.xy is not None) else []
        if not poly_list and self.current_image is not None:
            # fallback: masks.data（原图分辨率）
            mask_data = masks_obj.data.cpu().numpy()
            out = []
            for i in range(len(mask_data)):
                mk = (mask_data[i] > 0.5).astype(np.uint8)
                bb = self._mask_to_bbox(mk)
                if bb:
                    out.append((mk, bb))
            return out
        elif not poly_list:
            return []

        h, w = self.current_image.shape[:2]
        out = []
        for poly_pts in poly_list:
            if len(poly_pts) < 3:
                continue
            xs, ys = poly_pts[:, 0], poly_pts[:, 1]
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max())]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(poly_pts).astype(np.int32)], 1)
            out.append((mask, bbox))
        return out

    def predict_point(self, x, y):
        if self.sam_model is None:
            return None, None
        results = self.sam_model.predict(
            source=self.current_image,
            points=[[x, y]], labels=[1],
            save=False, show=False, verbose=False,
        )
        items = self._masks_to_original(results[0])
        if not items:
            return None, None
        return max(items, key=lambda t: t[0].sum())

    def predict_box(self, x1, y1, x2, y2):
        if self.sam_model is None:
            return None
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        results = self.sam_model.predict(
            source=self.current_image,
            bboxes=[[x1, y1, x2, y2]],
            save=False, show=False, verbose=False,
        )
        items = self._masks_to_original(results[0])
        if not items:
            return None
        return max(items, key=lambda t: t[0].sum())[0]

    def predict_text(self, text_list):
        if self.semantic_predictor is None:
            return []
        try:
            results = self.semantic_predictor(
                text=text_list, save=False, show=False, verbose=False,
            )
        except KeyError:
            return []
        if not results or results[0].masks is None:
            return []
        items = self._masks_to_original(results[0])
        result_boxes = results[0].boxes
        names = (getattr(self.semantic_predictor.model, "names", {})
                 if self.semantic_predictor else {})
        label_list = []
        if (result_boxes is not None and hasattr(result_boxes, "cls")
                and result_boxes.cls is not None):
            class_ids = result_boxes.cls.cpu().numpy().astype(int)
            for cid in class_ids:
                if isinstance(names, dict):
                    label = names.get(cid, text_list[cid] if cid < len(text_list) else "sam")
                else:
                    # names is a list (e.g. ultralytics SAM3)
                    label = names[cid] if cid < len(names) else (text_list[cid] if cid < len(text_list) else "sam")
                label_list.append(label)
        for i in range(len(items) - len(label_list)):
            label_list.append(text_list[i % len(text_list)]
                              if text_list else "sam")
        out = []
        for i, (mask, bbox) in enumerate(items):
            out.append((mask, bbox,
                        label_list[i] if i < len(label_list) else "sam"))
        return out

    @staticmethod
    def _mask_to_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1, y2 = int(np.where(rows)[0][[0, -1]][0]), int(np.where(rows)[0][[0, -1]][1])
        x1, x2 = int(np.where(cols)[0][[0, -1]][0]), int(np.where(cols)[0][[0, -1]][1])
        return [x1, y1, x2, y2]

    @staticmethod
    def mask_to_polygon(mask, epsilon=0.002):
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        c = max(contours, key=cv2.contourArea)
        arc_len = cv2.arcLength(c, True)
        epsilon_px = max(epsilon * arc_len, 1.0)
        approx = cv2.approxPolyDP(c, epsilon_px, True)
        return [(float(p[0][0]), float(p[0][1])) for p in approx]
