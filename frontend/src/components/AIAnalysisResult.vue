<template>
  <div class="ai-analysis-result">
    <!-- 处理完成提示 -->
    <el-alert
      title="AI 智能分析完成"
      type="success"
      :closable="false"
      show-icon
      class="result-alert"
    >
      <template #default>
        <span>
          使用模型: {{ result.model_results?.model_info?.name || 'VisionFM' }} |
          总耗时: {{ result.metadata?.total_time_ms }}ms |
          {{ result.from_cache ? '(缓存命中)' : '' }}
        </span>
      </template>
    </el-alert>

    <!-- 图片展示 -->
    <div class="images-display">
      <!-- 原图 -->
      <div class="image-box">
        <h4 class="image-title">原始图片</h4>
        <div class="image-wrapper">
          <el-image :src="result.images?.original" fit="contain" class="result-image" />
        </div>
      </div>

      <!-- 分割掩码 -->
      <div v-if="result.images?.mask" class="image-box">
        <h4 class="image-title">血管分割掩码</h4>
        <div class="image-wrapper">
          <el-image :src="result.images.mask" fit="contain" class="result-image" />
        </div>
      </div>
    </div>

    <!-- 模型检测结果 -->
    <el-card class="model-results-card" shadow="never">
      <template #header>
        <div class="card-header">
          <el-icon><DataAnalysis /></el-icon>
          <span>模型检测结果</span>
        </div>
      </template>

      <!-- 分类结果 -->
      <div v-if="result.model_results?.classification" class="result-section">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="预测类别">
            <el-tag :type="result.model_results.classification.predicted_class === 0 ? 'success' : 'danger'">
              {{ result.model_results.classification.class_label }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="疾病概率">
            {{ (result.model_results.classification.probability * 100).toFixed(2) }}%
          </el-descriptions-item>
          <el-descriptions-item label="模型置信度" :span="2">
            <el-progress
              :percentage="Math.round(result.model_results.classification.confidence * 100)"
              :stroke-width="12"
              :text-inside="true"
            />
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- 分割结果 -->
      <div v-if="result.model_results?.segmentation" class="result-section">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="血管面积比">
            {{ (result.model_results.segmentation.vessel_area_ratio * 100).toFixed(2) }}%
          </el-descriptions-item>
          <el-descriptions-item label="血管密度">
            {{ (result.model_results.segmentation.vessel_density * 100).toFixed(2) }}%
          </el-descriptions-item>
          <el-descriptions-item label="血管弯曲度指数">
            {{ result.model_results.segmentation.tortuosity_index?.toFixed(4) || 'N/A' }}
          </el-descriptions-item>
          <el-descriptions-item label="血管像素数">
            {{ result.model_results.segmentation.vessel_pixels?.toLocaleString() || 'N/A' }}
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- 无结果提示 -->
      <el-empty v-if="!result.model_results?.classification && !result.model_results?.segmentation" description="未运行本地模型" />
    </el-card>

    <!-- AI 分析报告 -->
    <el-card class="ai-report-card" shadow="never">
      <template #header>
        <div class="card-header">
          <el-icon><ChatDotRound /></el-icon>
          <span>AI 分析报告</span>
          <el-tag size="small" type="info" class="model-tag">
            {{ result.ai_analysis?.model_used }}
          </el-tag>
        </div>
      </template>

      <div class="ai-content" v-html="formatMarkdown(result.ai_analysis?.content)"></div>
    </el-card>

    <!-- 元信息 -->
    <el-card class="metadata-card" shadow="never">
      <template #header>
        <div class="card-header">
          <el-icon><InfoFilled /></el-icon>
          <span>调用信息</span>
        </div>
      </template>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item label="图像哈希">
          <span class="hash-text">{{ result.metadata?.image_hash?.substring(0, 8) }}...</span>
        </el-descriptions-item>
        <el-descriptions-item label="时间戳">
          {{ formatTime(result.metadata?.timestamp) }}
        </el-descriptions-item>
        <el-descriptions-item label="API 响应时间">
          {{ result.metadata?.api_response_time_ms }}ms
        </el-descriptions-item>
        <el-descriptions-item label="输入 Tokens">
          {{ result.metadata?.input_tokens?.toLocaleString() }}
        </el-descriptions-item>
        <el-descriptions-item label="输出 Tokens">
          {{ result.metadata?.output_tokens?.toLocaleString() }}
        </el-descriptions-item>
        <el-descriptions-item label="总 Tokens">
          {{ result.metadata?.tokens_used?.toLocaleString() }}
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- 操作按钮 -->
    <div class="action-buttons">
      <el-button type="primary" @click="$emit('new-prediction')">
        <el-icon><Plus /></el-icon>
        新分析
      </el-button>
      <el-button @click="downloadReport">
        <el-icon><Download /></el-icon>
        下载报告
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Plus, Download, DataAnalysis, ChatDotRound, InfoFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

interface Props {
  result: any
}

const props = defineProps<Props>()

const emit = defineEmits<{
  'new-prediction': []
}>()

// 简单的 Markdown 转 HTML（处理标题和粗体）
const formatMarkdown = (content: string): string => {
  if (!content) return ''

  let html = content
    // 标题
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // 粗体
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // 列表
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    // 换行
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')

  // 包装列表
  html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')

  return `<p>${html}</p>`
}

const formatTime = (timestamp: string): string => {
  if (!timestamp) return 'N/A'
  try {
    return new Date(timestamp).toLocaleString('zh-CN')
  } catch {
    return timestamp
  }
}

const downloadReport = () => {
  const reportData = {
    analysis_result: props.result.ai_analysis?.content,
    model_results: props.result.model_results,
    metadata: props.result.metadata,
    generated_at: new Date().toISOString()
  }

  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `ai_analysis_report_${Date.now()}.json`
  link.click()
  URL.revokeObjectURL(url)
  ElMessage.success('报告已下载')
}
</script>

<style scoped>
.ai-analysis-result {
  width: 100%;
}

.result-alert {
  margin-bottom: 24px;
}

.images-display {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.image-box {
  border: 1px solid var(--el-border-color);
  border-radius: 8px;
  overflow: hidden;
  background: var(--el-bg-color);
}

.image-title {
  margin: 0;
  padding: 12px 16px;
  background: var(--el-fill-color-light);
  font-size: 14px;
  font-weight: 600;
  text-align: center;
  border-bottom: 1px solid var(--el-border-color);
}

.image-wrapper {
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 250px;
}

.result-image {
  width: 100%;
  height: auto;
  max-height: 400px;
  object-fit: contain;
}

.model-results-card,
.ai-report-card,
.metadata-card {
  margin-bottom: 20px;
  border-radius: 8px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.model-tag {
  margin-left: auto;
}

.result-section {
  margin-bottom: 16px;
}

.result-section:last-child {
  margin-bottom: 0;
}

.ai-content {
  line-height: 1.8;
  color: var(--el-text-color-primary);
}

.ai-content :deep(h1) {
  font-size: 20px;
  margin: 0 0 16px 0;
  color: var(--el-color-primary);
}

.ai-content :deep(h2) {
  font-size: 18px;
  margin: 20px 0 12px 0;
  color: var(--el-text-color-primary);
}

.ai-content :deep(h3) {
  font-size: 16px;
  margin: 16px 0 10px 0;
  color: var(--el-text-color-primary);
}

.ai-content :deep(p) {
  margin: 8px 0;
}

.ai-content :deep(ul),
.ai-content :deep(ol) {
  margin: 8px 0;
  padding-left: 20px;
}

.ai-content :deep(li) {
  margin: 4px 0;
}

.ai-content :deep(strong) {
  color: var(--el-color-primary);
}

.hash-text {
  font-family: monospace;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.action-buttons {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 24px;
}

/* 响应式 */
@media (max-width: 768px) {
  .images-display {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .image-wrapper {
    min-height: 200px;
  }

  .result-image {
    max-height: 300px;
  }

  .action-buttons {
    flex-direction: column;
  }

  .action-buttons .el-button {
    width: 100%;
  }
}
</style>
