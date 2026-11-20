/** [x1, y1, x2, y2] in original-image pixels. */
export type BBox = [number, number, number, number];

export interface Detection {
  id: string;
  label: string;
  confidence: number;
  bbox: BBox;
}

export interface DetectionModelInfo {
  name: string;
  conf_threshold: number;
  iou_threshold: number;
  device: string;
}

export interface DetectResponse {
  image_id: string;
  /** Original (EXIF-oriented) pixel size that all bboxes refer to. */
  width: number;
  height: number;
  detections: Detection[];
  model: DetectionModelInfo;
  note: string;
}
