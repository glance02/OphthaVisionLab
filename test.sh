curl -X POST http://localhost:8000/api/ai/analyze \
  -F "file=@/home/seborid/visonFM_/backend/示例图片.png" \
  -F "run_segmentation=true" \
  -F "run_classification=true" \
  -F "temperature=0.7"