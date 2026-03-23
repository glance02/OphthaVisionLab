<template>
  <div class="app">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <el-icon class="logo-icon"><View /></el-icon>
        <h1 class="app-title">VisionFM</h1>
        <span class="header-subtitle">眼科 AI 智能分析系统</span>
      </div>
      <div class="header-right">
        <el-button text @click="handleNewTask" v-if="result">
          <el-icon><RefreshRight /></el-icon>
          新建分析
        </el-button>
      </div>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <!-- Left Panel: Upload & Settings -->
      <aside class="left-panel">
        <!-- Upload Area -->
        <div class="upload-section" v-if="!processing && !result">
          <el-upload
            class="upload-area"
            drag
            :auto-upload="false"
            :show-file-list="false"
            accept=".jpg,.jpeg,.png"
            :on-change="handleFileChange"
            :limit="1"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="upload-text">拖拽图片到此处或点击上传</div>
            <div class="upload-hint">支持 JPG/PNG 格式，最大 10MB</div>
          </el-upload>

          <!-- Image Preview -->
          <div v-if="previewUrl" class="preview-box">
            <div class="preview-header">
              <span>已选择图片</span>
              <el-button type="danger" text size="small" @click="clearPreview">清除</el-button>
            </div>
            <el-image :src="previewUrl" fit="contain" class="preview-image" />
          </div>
        </div>

        <!-- Settings -->
        <div class="settings-section" v-if="!processing && !result">
          <!-- Task Type Selector -->
          <div class="task-selector">
            <label class="task-label">选择任务类型</label>
            <el-radio-group v-model="settings.taskType" class="task-radio-group">
              <el-radio-button value="ai">AI 智能分析</el-radio-button>
              <el-radio-button value="idrid_ma">微动脉瘤分割</el-radio-button>
              <el-radio-button value="vessel_seg">血管分割</el-radio-button>
              <el-radio-button value="dr_classify">DR 分类</el-radio-button>
            </el-radio-group>
          </div>

          <el-collapse v-model="settingsOpen">
            <el-collapse-item title="分析设置" name="settings">
              <!-- AI Analysis Settings -->
              <template v-if="settings.taskType === 'ai'">
                <el-form-item label="运行分割">
                  <el-switch v-model="settings.run_segmentation" />
                </el-form-item>
                <el-form-item label="运行分类">
                  <el-switch v-model="settings.run_classification" />
                </el-form-item>
                <el-form-item label="分割阈值">
                  <div class="slider-container">
                    <el-slider
                      v-model="settings.seg_threshold"
                      :min="0"
                      :max="1"
                      :step="0.01"
                      :show-tooltip="true"
                    />
                    <span class="slider-value">{{ (settings.seg_threshold * 100).toFixed(0) }}%</span>
                  </div>
                </el-form-item>
                <el-form-item label="AI 创意度">
                  <div class="slider-container">
                    <el-slider
                      v-model="settings.temperature"
                      :min="0"
                      :max="1"
                      :step="0.1"
                      :show-tooltip="true"
                    />
                    <span class="slider-value">{{ settings.temperature }}</span>
                  </div>
                </el-form-item>
              </template>

              <!-- IDRiD MA Settings -->
              <template v-if="settings.taskType === 'idrid_ma'">
                <el-form-item label="分割阈值">
                  <div class="slider-container">
                    <el-slider
                      v-model="settings.ma_threshold"
                      :min="0"
                      :max="1"
                      :step="0.01"
                      :show-tooltip="true"
                    />
                    <span class="slider-value">{{ (settings.ma_threshold * 100).toFixed(0) }}%</span>
                  </div>
                </el-form-item>
                <div class="task-desc">
                  <el-icon><InfoFilled /></el-icon>
                  <span>IDRiD 微动脉瘤分割模型，用于检测眼底图像中的微动脉瘤病变区域</span>
                </div>
              </template>

              <!-- Vessel Segmentation Settings -->
              <template v-if="settings.taskType === 'vessel_seg'">
                <el-form-item label="分割阈值">
                  <div class="slider-container">
                    <el-slider
                      v-model="settings.seg_threshold"
                      :min="0"
                      :max="1"
                      :step="0.01"
                      :show-tooltip="true"
                    />
                    <span class="slider-value">{{ (settings.seg_threshold * 100).toFixed(0) }}%</span>
                  </div>
                </el-form-item>
              </template>

              <!-- DR Classification Settings -->
              <template v-if="settings.taskType === 'dr_classify'">
                <div class="task-desc">
                  <el-icon><InfoFilled /></el-icon>
                  <span>糖尿病视网膜病变(DR)二分类检测，判断是否存在 DR 病变</span>
                </div>
              </template>
            </el-collapse-item>
          </el-collapse>

          <!-- Analyze Button -->
          <el-button
            type="primary"
            size="large"
            class="analyze-btn"
            :disabled="!previewUrl"
            @click="startAnalysis"
          >
            <el-icon><MagicStick /></el-icon>
            {{ analyzeButtonText }}
          </el-button>
        </div>

        <!-- Processing State -->
        <div class="processing-section" v-if="processing">
          <el-progress
            :percentage="progress"
            :status="progressStatus"
            :stroke-width="24"
            :text-inside="true"
          />
          <div class="status-text">{{ statusMessage }}</div>
          <div class="file-info-mini" v-if="uploadedFile">
            {{ uploadedFile.name }} ({{ (uploadedFile.size / 1024).toFixed(1) }} KB)
          </div>
        </div>

        <!-- Result Actions -->
        <div class="result-actions-mini" v-if="result && !processing">
          <el-button type="primary" @click="downloadImage(result.data.images?.original, 'original')">
            <el-icon><Download /></el-icon> 原图
          </el-button>
          <el-button type="success" @click="downloadImage(result.data.images?.mask, 'mask')" v-if="result.data.images?.mask">
            <el-icon><Download /></el-icon> 掩码
          </el-button>
          <el-button @click="downloadReport" v-if="result.data.ai_analysis">
            <el-icon><Download /></el-icon> 报告
          </el-button>
        </div>
      </aside>

      <!-- Right Panel: Results -->
      <section class="right-panel">
        <!-- Empty State -->
        <div class="empty-state" v-if="!processing && !result">
          <el-icon class="empty-icon"><Picture /></el-icon>
          <h3>上传眼底图像开始 AI 分析</h3>
          <p>自动进行血管分割、分类和 AI 智能分析</p>
        </div>

        <!-- Result with Tabs -->
        <div class="result-container" v-if="result">
          <!-- Result Header -->
          <div class="result-header">
            <el-alert
              :title="resultTitle"
              type="success"
              :closable="false"
              show-icon
            >
              <template #default>
                {{ resultSubtitle }}
              </template>
            </el-alert>
          </div>

          <!-- Tab Navigation -->
          <el-tabs v-model="activeTab" class="result-tabs">
            <!-- IDRiD MA Result -->
            <el-tab-pane label="分割结果" name="segment" v-if="result.task === 'idrid_microaneurysm_segmentation'">
              <div class="result-images">
                <div class="image-col" v-if="result.data.originalImage">
                  <h4>原始图片</h4>
                  <el-image :src="result.data.originalImage" fit="contain" class="result-image" />
                </div>
                <div class="image-col" v-if="result.data.maskImage">
                  <h4>微动脉瘤分割掩码</h4>
                  <el-image :src="result.data.maskImage" fit="contain" class="result-image" />
                </div>
              </div>
              <!-- MA Statistics -->
              <div class="result-stats">
                <el-row :gutter="16">
                  <el-col :span="8">
                    <div class="stat-card">
                      <div class="stat-value">{{ result.data.num_lesions }}</div>
                      <div class="stat-label">病变数量</div>
                    </div>
                  </el-col>
                  <el-col :span="8">
                    <div class="stat-card">
                      <div class="stat-value">{{ result.data.lesion_area }}</div>
                      <div class="stat-label">病变像素</div>
                    </div>
                  </el-col>
                  <el-col :span="8">
                    <div class="stat-card">
                      <div class="stat-value highlight">{{ result.data.lesion_ratio }}</div>
                      <div class="stat-label">病变占比</div>
                    </div>
                  </el-col>
                </el-row>
              </div>
            </el-tab-pane>

            <!-- Standard Segmentation Result -->
            <el-tab-pane label="分割结果" name="segment" v-else-if="result.data.images?.original">
              <!-- Images Row -->
              <div class="result-images">
                <div class="image-col" v-if="result.data.images?.original">
                  <h4>原始图片</h4>
                  <el-image :src="result.data.images.original" fit="contain" class="result-image" />
                </div>
                <div class="image-col" v-if="result.data.images?.mask">
                  <h4>血管分割掩码</h4>
                  <el-image :src="result.data.images.mask" fit="contain" class="result-image" />
                </div>
              </div>
            </el-tab-pane>

            <!-- Classification Result -->
            <el-tab-pane label="分类结果" name="classify" v-if="result.task === 'binary_classification'">
              <div class="classify-result">
                <div class="classify-card" :class="result.data.predicted_class === 1 ? 'danger' : 'normal'">
                  <el-icon :size="48"><Warning v-if="result.data.predicted_class === 1" /><CircleCheck v-else /></el-icon>
                  <div class="classify-label">{{ result.data.predicted_class === 1 ? 'DR 阳性' : '健康' }}</div>
                  <div class="classify-prob">概率: {{ (result.data.probability * 100).toFixed(1) }}%</div>
                  <div class="classify-conf">置信度: {{ (result.data.confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane label="AI 分析报告" name="ai" v-if="result.data.ai_analysis">
              <div class="ai-report-content" v-html="formatMarkdown(result.data.ai_analysis.content)"></div>
            </el-tab-pane>
          </el-tabs>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import axios from 'axios'
import { View, MagicStick, UploadFilled, RefreshRight, Download, Picture, InfoFilled, Warning, CircleCheck } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

// API 配置
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// 状态
const processing = ref(false)
const progress = ref(0)
const progressStatus = ref<'success' | 'exception' | 'warning' | ''>('')
const statusMessage = ref('')
const uploadedFile = ref<File | null>(null)
const previewUrl = ref('')
const result = ref<any>(null)
const settingsOpen = ref(['settings'])
const activeTab = ref('segment')

// 设置
const settings = reactive({
  taskType: 'ai',  // ai, idrid_ma, vessel_seg, dr_classify
  run_segmentation: true,
  run_classification: true,
  seg_checkpoint: 'checkpoints/seg/checkpoint_108_linear.pth',
  cls_checkpoint: 'checkpoints/single_cls/checkpoint_teacher_linear.pth',
  ma_checkpoint: 'checkpoints/idrid_ma/net.pt7',
  seg_threshold: 0.5,
  ma_threshold: 0.5,
  temperature: 0.7
})

// 计算属性
const analyzeButtonText = computed(() => {
  switch (settings.taskType) {
    case 'idrid_ma': return '开始微动脉瘤分割'
    case 'vessel_seg': return '开始血管分割'
    case 'dr_classify': return '开始 DR 分类'
    default: return '开始 AI 分析'
  }
})

const resultTitle = computed(() => {
  if (!result.value) return ''
  if (result.value.task === 'idrid_microaneurysm_segmentation') return '微动脉瘤分割完成'
  if (result.value.task === 'binary_classification') return 'DR 分类完成'
  return '分析完成'
})

const resultSubtitle = computed(() => {
  if (!result.value) return ''
  if (result.value.data.metadata?.total_time_ms) {
    return `耗时: ${result.value.data.metadata.total_time_ms}ms | 模型: ${result.value.data.ai_analysis?.model_used || 'VisionFM'}`
  }
  return `任务: ${result.value.task}`
})

// 文件处理
const handleFileChange = (file: any) => {
  if (file.raw.size > 10 * 1024 * 1024) {
    ElMessage.error('图片大小不能超过 10MB')
    return
  }
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png']
  if (!validTypes.includes(file.raw.type)) {
    ElMessage.error('仅支持 JPG/PNG 格式')
    return
  }
  previewUrl.value = URL.createObjectURL(file.raw)
  uploadedFile.value = file.raw
}

const clearPreview = () => {
  previewUrl.value = ''
  uploadedFile.value = null
}

// 开始分析 - 根据任务类型调用不同接口
const startAnalysis = async () => {
  if (!uploadedFile.value) return

  processing.value = true
  progress.value = 0
  progressStatus.value = ''
  statusMessage.value = '准备上传...'

  const formData = new FormData()
  formData.append('file', uploadedFile.value)

  const uploadProgress = (progressEvent: any) => {
    if (progressEvent.total) {
      const percentCompleted = Math.round((progressEvent.loaded * 30) / progressEvent.total)
      progress.value = Math.min(percentCompleted, 30)
    }
  }

  try {
    statusMessage.value = '正在上传...'
    let response: any

    // 根据任务类型调用不同的 API
    switch (settings.taskType) {
      case 'idrid_ma':
        // IDRiD 微动脉瘤分割
        formData.append('checkpoint', settings.ma_checkpoint)
        formData.append('threshold', settings.ma_threshold.toString())
        formData.append('input_size', '96')
        response = await axios.post(`${API_BASE}/api/idrid/ma`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: uploadProgress,
          timeout: 120000
        })
        break

      case 'vessel_seg':
        // 血管分割
        formData.append('checkpoint', settings.seg_checkpoint)
        formData.append('threshold', settings.seg_threshold.toString())
        formData.append('input_size', '512')
        response = await axios.post(`${API_BASE}/api/segment`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: uploadProgress,
          timeout: 120000
        })
        // 转换格式以统一显示
        response = {
          data: {
            success: true,
            task: 'vessel_segmentation',
            data: {
              images: {
                original: response.data.data.originalImage,
                mask: response.data.data.maskImage
              }
            }
          }
        }
        response.data = response.data.data
        break

      case 'dr_classify':
        // DR 二分类
        formData.append('checkpoint', settings.cls_checkpoint)
        formData.append('input_size', '224')
        response = await axios.post(`${API_BASE}/api/classify/binary`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: uploadProgress,
          timeout: 120000
        })
        break

      case 'ai':
      default:
        // AI 智能分析
        formData.append('run_segmentation', settings.run_segmentation.toString())
        formData.append('run_classification', settings.run_classification.toString())
        formData.append('seg_checkpoint', settings.seg_checkpoint)
        formData.append('cls_checkpoint', settings.cls_checkpoint)
        formData.append('seg_threshold', settings.seg_threshold.toString())
        formData.append('temperature', settings.temperature.toString())
        response = await axios.post(`${API_BASE}/api/ai/analyze`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: uploadProgress,
          timeout: 300000
        })
        break
    }

    progress.value = 50
    statusMessage.value = '处理中...'
    await new Promise(resolve => setTimeout(resolve, 500))

    progress.value = 80
    await new Promise(resolve => setTimeout(resolve, 300))
    progress.value = 100
    progressStatus.value = 'success'
    statusMessage.value = '完成！'

    result.value = response.data
    activeTab.value = 'segment'  // 默认显示分割结果
    ElMessage.success('分析完成')

  } catch (error: any) {
    console.error('处理失败:', error)
    progressStatus.value = 'exception'
    const errorMessage = error.response?.data?.detail || error.message || '未知错误'
    ElMessage.error(`处理失败: ${errorMessage}`)
  } finally {
    setTimeout(() => {
      processing.value = false
    }, 500)
  }
}

// 新建分析
const handleNewTask = () => {
  result.value = null
  progress.value = 0
  progressStatus.value = ''
  statusMessage.value = ''
  activeTab.value = 'segment'
  clearPreview()
}

// 下载图片
const downloadImage = (url: string, prefix: string) => {
  if (!url) return
  const link = document.createElement('a')
  link.href = url
  link.download = `visionfm_${prefix}_${Date.now()}.png`
  link.click()
  ElMessage.success('已下载')
}

// 下载报告
const downloadReport = () => {
  if (!result.value) return
  const reportData = {
    analysis_result: result.value.data.ai_analysis?.content,
    model_results: result.value.data.model_results,
    metadata: result.value.data.metadata,
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

// Markdown 格式化
const formatMarkdown = (content: string): string => {
  if (!content) return ''
  let html = content
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
  html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
  return `<p>${html}</p>`
}
</script>

<style scoped>
.app {
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
.app-header {
  height: 60px;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  font-size: 28px;
  color: #00d9ff;
}

.app-title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.header-subtitle {
  color: rgba(255, 255, 255, 0.5);
  font-size: 14px;
  margin-left: 8px;
  padding-left: 12px;
  border-left: 1px solid rgba(255, 255, 255, 0.2);
}

/* Main Layout */
.app-main {
  flex: 1;
  display: flex;
  gap: 20px;
  padding: 20px 24px;
  overflow: hidden;
}

/* Left Panel */
.left-panel {
  width: 360px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 16px;
  padding: 20px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Task Selector */
.task-selector {
  margin-bottom: 16px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
}

.task-label {
  display: block;
  color: rgba(255, 255, 255, 0.7);
  font-size: 13px;
  margin-bottom: 8px;
}

.task-radio-group {
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.task-radio-group :deep(.el-radio-button) {
  flex: 1;
}

.task-radio-group :deep(.el-radio-button__inner) {
  width: 100%;
  padding: 8px 4px;
  font-size: 12px;
}

.task-desc {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 12px;
  background: rgba(0, 217, 255, 0.1);
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
  line-height: 1.5;
}

.task-desc .el-icon {
  color: #00d9ff;
  flex-shrink: 0;
  margin-top: 2px;
}

/* Upload */
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.upload-area {
  width: 100%;
  min-height: 160px;
}

.upload-area :deep(.el-upload-dragger) {
  background: rgba(255, 255, 255, 0.03);
  border: 2px dashed rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  transition: all 0.3s;
}

.upload-area :deep(.el-upload-dragger:hover) {
  border-color: #00d9ff;
  background: rgba(0, 217, 255, 0.05);
}

.upload-icon {
  font-size: 48px;
  color: rgba(255, 255, 255, 0.3);
  margin-bottom: 12px;
}

.upload-text {
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
  margin-bottom: 4px;
}

.upload-hint {
  color: rgba(255, 255, 255, 0.4);
  font-size: 12px;
}

.preview-box {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  padding: 12px;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: rgba(255, 255, 255, 0.7);
  font-size: 13px;
}

.preview-image {
  width: 100%;
  max-height: 180px;
  border-radius: 4px;
}

/* Settings */
.settings-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.settings-section :deep(.el-collapse) {
  border: none;
  background: transparent;
}

.settings-section :deep(.el-collapse-item__header) {
  background: rgba(255, 255, 255, 0.05);
  border: none;
  color: rgba(255, 255, 255, 0.7);
  border-radius: 8px;
}

.settings-section :deep(.el-collapse-item__wrap) {
  background: transparent;
  border: none;
}

.settings-section :deep(.el-collapse-item__content) {
  padding: 12px 0;
  color: rgba(255, 255, 255, 0.7);
}

.settings-section :deep(.el-form-item) {
  margin-bottom: 12px;
}

.settings-section :deep(.el-form-item__label) {
  color: rgba(255, 255, 255, 0.6);
  font-size: 13px;
}

.slider-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider-container :deep(.el-slider) {
  flex: 1;
}

.slider-value {
  min-width: 40px;
  text-align: right;
  color: #00d9ff;
  font-size: 13px;
}

.analyze-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%);
  border: none;
  color: #1a1a2e;
  font-weight: 600;
}

.analyze-btn:hover {
  opacity: 0.9;
}

.analyze-btn:disabled {
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.3);
}

/* Processing */
.processing-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  gap: 16px;
}

.status-text {
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
}

.file-info-mini {
  color: rgba(255, 255, 255, 0.4);
  font-size: 12px;
}

/* Result Actions Mini */
.result-actions-mini {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.result-actions-mini .el-button {
  flex: 1;
  min-width: 80px;
}

/* Right Panel */
.right-panel {
  flex: 1;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 16px;
  padding: 20px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

/* Empty State */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.3);
}

.empty-icon {
  font-size: 80px;
  margin-bottom: 20px;
}

.empty-state h3 {
  margin: 0 0 8px 0;
  font-size: 20px;
  color: rgba(255, 255, 255, 0.5);
}

.empty-state p {
  margin: 0;
  font-size: 14px;
}

/* Result Container */
.result-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
}

.result-header :deep(.el-alert) {
  background: rgba(0, 255, 136, 0.1);
  border: 1px solid rgba(0, 255, 136, 0.3);
}

/* Tabs */
.result-tabs {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.result-tabs :deep(.el-tabs__header) {
  margin-bottom: 16px;
}

.result-tabs :deep(.el-tabs__nav-wrap::after) {
  background: rgba(255, 255, 255, 0.1);
}

.result-tabs :deep(.el-tabs__item) {
  color: rgba(255, 255, 255, 0.6);
}

.result-tabs :deep(.el-tabs__item.is-active) {
  color: #00d9ff;
}

.result-tabs :deep(.el-tabs__content) {
  flex: 1;
  overflow-y: auto;
}

/* Result Images */
.result-images {
  display: flex;
  gap: 16px;
  height: 100%;
}

.image-col {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  overflow: hidden;
}

.image-col h4 {
  margin: 0;
  padding: 10px 14px;
  background: rgba(255, 255, 255, 0.05);
  font-size: 13px;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.7);
  text-align: center;
}

.result-image {
  flex: 1;
  width: 100%;
  min-height: 200px;
}

/* AI Report Content */
.ai-report-content {
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 20px;
  height: 100%;
  overflow-y: auto;
  font-size: 14px;
  line-height: 1.8;
  color: rgba(255, 255, 255, 0.85);
}

.ai-report-content :deep(h3) {
  font-size: 16px;
  color: #00d9ff;
  margin: 16px 0 12px 0;
}

.ai-report-content :deep(h4) {
  font-size: 15px;
  color: rgba(255, 255, 255, 0.9);
  margin: 14px 0 10px 0;
}

.ai-report-content :deep(p) {
  margin: 8px 0;
}

.ai-report-content :deep(ul) {
  margin: 8px 0;
  padding-left: 20px;
}

.ai-report-content :deep(li) {
  margin: 4px 0;
}

.ai-report-content :deep(strong) {
  color: #00ff88;
}

/* Result Stats */
.result-stats {
  margin-top: 20px;
}

.stat-card {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: #00d9ff;
}

.stat-value.highlight {
  color: #ff6b6b;
}

.stat-label {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
  margin-top: 4px;
}

/* Classification Result */
.classify-result {
  display: flex;
  justify-content: center;
  padding: 40px 20px;
}

.classify-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 40px 60px;
  border-radius: 16px;
  background: rgba(0, 0, 0, 0.2);
}

.classify-card.normal {
  border: 2px solid rgba(0, 255, 136, 0.3);
}

.classify-card.normal .el-icon {
  color: #00ff88;
}

.classify-card.danger {
  border: 2px solid rgba(255, 107, 107, 0.3);
}

.classify-card.danger .el-icon {
  color: #ff6b6b;
}

.classify-label {
  font-size: 24px;
  font-weight: 600;
  color: #fff;
}

.classify-prob, .classify-conf {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.6);
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.02);
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}
</style>
