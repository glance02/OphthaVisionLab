<template>
  <div class="app">
    <!-- Left Column: Settings & Upload -->
    <aside class="left-column">
      <!-- Upper: System Name & Introduction -->
      <div class="upper-section">
        <div class="system-name">
          <el-icon class="logo-icon"><View /></el-icon>
          <span>AI智能眼科分析</span>
        </div>

        <div class="intro-section">
          <span class="section-title">功能介绍</span>
          <p class="intro-text">
            通过眼科图片进行血管分割、微动脉瘤分割和AI智能分析
          </p>
        </div>

        <!-- Settings -->
        <div class="settings-section">
          <span class="section-title">分割阈值设置</span>
          <div class="threshold-control">
            <span>血管分割阈值: {{ settings.seg_threshold.toFixed(2) }}</span>
            <el-slider
              v-model="settings.seg_threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :show-tooltip="false"
            />
          </div>
          <div class="threshold-control">
            <span>微动脉瘤阈值: {{ settings.ma_threshold.toFixed(2) }}</span>
            <el-slider
              v-model="settings.ma_threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :show-tooltip="false"
            />
          </div>
        </div>
      </div>

      <!-- Lower: Upload & Preview -->
      <div class="lower-section">
        <span class="section-title">上传图片</span>
        <el-upload
          v-if="!previewUrl && !processing && !vesselResult && !maResult"
          class="upload-btn"
          drag
          :auto-upload="false"
          :show-file-list="false"
          accept=".jpg,.jpeg,.png"
          :on-change="handleFileChange"
          :limit="1"
        >
          <el-icon class="upload-icon"><UploadFilled /></el-icon>
          <span>点击上传或拖拽</span>
        </el-upload>

        <!-- Preview -->
        <div v-if="previewUrl" class="preview-box">
          <div class="preview-label">原图预览</div>
          <el-image :src="previewUrl" fit="contain" class="preview-image" />
          <el-button type="danger" size="small" @click="clearPreview">删除</el-button>
        </div>

        <!-- Actions -->
        <div class="actions-section" v-if="previewUrl && !processing && !vesselResult">
          <el-button type="primary" size="large" class="start-btn" @click="startAnalysis">
            <el-icon><MagicStick /></el-icon>
            开始分析
          </el-button>
        </div>

        <!-- New Analysis -->
        <div class="actions-section" v-if="(vesselResult || maResult) && !processing">
          <el-button @click="handleNewTask" size="large">
            <el-icon><RefreshRight /></el-icon>
            新建分析
          </el-button>
        </div>
      </div>
    </aside>

    <!-- Middle Column: Results Display -->
    <section class="middle-column">
      <!-- Upper: Vessel Segmentation Results -->
      <div class="result-section upper-section">
        <div class="section-header">
          <span>血管分割结果</span>
        </div>

        <!-- Empty State -->
        <div class="empty-state" v-if="!vesselResult && !processing">
          <el-icon class="empty-icon"><Picture /></el-icon>
          <p>上传图片后显示血管分割结果</p>
        </div>

        <!-- Processing -->
        <div class="processing-status" v-if="processing && currentTask === 'vessel'">
          <el-progress
            :percentage="progress"
            :status="progressStatus"
            :stroke-width="16"
            :text-inside="true"
          />
          <div class="status-text">正在执行血管分割...</div>
        </div>

        <!-- Result Images -->
        <div class="result-images" v-if="vesselResult">
          <div class="image-card">
            <div class="image-header">血管分割结果</div>
            <el-image
              :src="vesselResult.maskImage"
              fit="contain"
              class="result-image"
            />
          </div>
        </div>
      </div>

      <!-- Lower: Microaneurysm Segmentation Results -->
      <div class="result-section lower-section">
        <div class="section-header">
          <span>微动脉瘤分割结果</span>
        </div>

        <!-- Empty State -->
        <div class="empty-state" v-if="!maResult && !processing">
          <el-icon class="empty-icon"><Picture /></el-icon>
          <p>上传图片后显示微动脉瘤分割结果</p>
        </div>

        <!-- Processing -->
        <div class="processing-status" v-if="processing && currentTask === 'ma'">
          <el-progress
            :percentage="progress"
            :status="progressStatus"
            :stroke-width="16"
            :text-inside="true"
          />
          <div class="status-text">正在执行微动脉瘤分割...</div>
        </div>

        <!-- Result Images -->
        <div class="result-images" v-if="maResult">
          <div class="image-card">
            <div class="image-header">微动脉瘤分割结果</div>
            <el-image
              :src="maResult.maskImage"
              fit="contain"
              class="result-image"
            />
          </div>
        </div>
      </div>
    </section>

    <!-- Right Column: AI Analysis Report -->
    <aside class="right-column">
      <!-- Empty State -->
      <div class="empty-state" v-if="!aiReport">
        <el-icon class="empty-icon"><Document /></el-icon>
        <h3>等待分析报告</h3>
        <p>上传图片并分析后显示AI诊断报告</p>
      </div>

      <!-- AI Report Content -->
      <div class="report-content" v-if="aiReport">
        <div class="report-header">
          <el-icon :size="24"><Document /></el-icon>
          <span>AI 诊断报告</span>
        </div>

        <div class="ai-report">
          <div class="report-text" v-html="formatMarkdown(aiReport)"></div>
        </div>

        <!-- Download -->
        <div class="download-section">
          <el-button @click="downloadReport" v-if="aiReport">
            <el-icon><Download /></el-icon>
            下载报告
          </el-button>
        </div>
      </div>

      <!-- File Info -->
      <div class="file-info" v-if="uploadedFile">
        <span class="file-name">{{ uploadedFile.name }}</span>
        <span class="file-size">{{ (uploadedFile.size / 1024).toFixed(1) }} KB</span>
      </div>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import axios from 'axios'
import {
  View, UploadFilled, MagicStick, RefreshRight, Download,
  Document, Picture
} from '@element-plus/icons-vue'
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
const currentTask = ref<'vessel' | 'ma' | 'ai' | ''>('')

// 结果数据
const vesselResult = ref<any>(null)
const maResult = ref<any>(null)
const aiReport = ref('')

// 设置
const settings = reactive({
  seg_threshold: 0.5,
  ma_threshold: 0.5
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

// 开始分析 - 一次性执行三个任务
const startAnalysis = async () => {
  if (!uploadedFile.value) return

  processing.value = true
  progress.value = 0
  progressStatus.value = ''

  const formData = new FormData()
  formData.append('file', uploadedFile.value)

  const uploadProgress = (progressEvent: any, taskMaxProgress: number) => {
    if (progressEvent.total) {
      const percentCompleted = Math.round((progressEvent.loaded * taskMaxProgress) / progressEvent.total)
      progress.value = Math.min(percentCompleted, taskMaxProgress)
    }
  }

  try {
    // 任务1: 血管分割
    statusMessage.value = '步骤1/3: 血管分割中...'
    currentTask.value = 'vessel'
    const vesselFormData = new FormData()
    vesselFormData.append('file', uploadedFile.value!)
    vesselFormData.append('threshold', settings.seg_threshold.toString())
    vesselFormData.append('input_size', '512')

    const vesselResponse = await axios.post(`${API_BASE}/api/segment`, vesselFormData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => uploadProgress(e, 25),
      timeout: 120000
    })

    vesselResult.value = vesselResponse.data.data
    progress.value = 30
    await new Promise(resolve => setTimeout(resolve, 200))

    // 任务2: 微动脉瘤分割
    statusMessage.value = '步骤2/3: 微动脉瘤分割中...'
    currentTask.value = 'ma'
    const maFormData = new FormData()
    maFormData.append('file', uploadedFile.value!)
    maFormData.append('threshold', settings.ma_threshold.toString())
    maFormData.append('input_size', '96')

    const maResponse = await axios.post(`${API_BASE}/api/idrid/ma`, maFormData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => uploadProgress(e, 50),
      timeout: 120000
    })

    maResult.value = maResponse.data.data
    progress.value = 60
    await new Promise(resolve => setTimeout(resolve, 200))

    // 任务3: AI分析
    statusMessage.value = '步骤3/3: AI分析中...'
    currentTask.value = 'ai'
    const aiFormData = new FormData()
    aiFormData.append('file', uploadedFile.value!)
    aiFormData.append('seg_threshold', settings.seg_threshold.toString())

    const aiResponse = await axios.post(`${API_BASE}/api/ai/analyze`, aiFormData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => uploadProgress(e, 80),
      timeout: 300000
    })

    aiReport.value = aiResponse.data.data.ai_analysis.content
    progress.value = 100
    progressStatus.value = 'success'
    statusMessage.value = '分析完成！'

    ElMessage.success('全部分析完成')

  } catch (error: any) {
    console.error('处理失败:', error)
    progressStatus.value = 'exception'
    const errorMessage = error.response?.data?.detail || error.message || '未知错误'
    ElMessage.error(`处理失败: ${errorMessage}`)
  } finally {
    currentTask.value = ''
    setTimeout(() => { processing.value = false }, 500)
  }
}

// 新建分析
const handleNewTask = () => {
  vesselResult.value = null
  maResult.value = null
  aiReport.value = ''
  progress.value = 0
  progressStatus.value = ''
  statusMessage.value = ''
  clearPreview()
}

// 下载报告
const downloadReport = () => {
  if (!aiReport.value) return
  const reportData = {
    analysis_result: aiReport.value,
    vessel_segmentation: vesselResult.value ? {
      original_image: vesselResult.value.originalImage,
      mask_image: vesselResult.value.maskImage
    } : null,
    microaneurysm_segmentation: maResult.value ? {
      original_image: maResult.value.originalImage,
      mask_image: maResult.value.maskImage,
      num_lesions: maResult.value.num_lesions,
      lesion_area: maResult.value.lesion_area,
      lesion_ratio: maResult.value.lesion_ratio
    } : null,
    generated_at: new Date().toISOString()
  }
  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `visionfm_analysis_report_${Date.now()}.json`
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
  display: flex;
  width: 100vw;
  height: 100vh;
  background: #0d1b2a;
  overflow: hidden;
}

/* Left Column */
.left-column {
  width: 280px;
  background: #1e3a5f;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 16px;
  flex-shrink: 0;
  overflow-y: auto;
}

.left-column .upper-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.left-column .lower-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: auto;
}

.system-name {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.system-name .logo-icon {
  font-size: 24px;
  color: #00d9ff;
}

.system-name span {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
}

.intro-section {
  padding: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
}

.intro-section .intro-text {
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
  line-height: 1.5;
  margin: 0;
}

.section-title {
  display: block;
  color: rgba(255, 255, 255, 0.7);
  font-size: 13px;
  margin-bottom: 10px;
}

/* Settings */
.settings-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.threshold-control {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.threshold-control span {
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
}

/* Upload */
.upload-btn {
  width: 100%;
}

.upload-btn :deep(.el-upload-dragger) {
  background: rgba(0, 0, 0, 0.2);
  border: 2px dashed rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 20px;
  transition: all 0.3s;
}

.upload-btn :deep(.el-upload-dragger:hover) {
  border-color: #00d9ff;
}

.upload-icon {
  font-size: 32px;
  color: rgba(255, 255, 255, 0.3);
  margin-bottom: 8px;
}

.upload-btn span {
  color: rgba(255, 255, 255, 0.6);
  font-size: 13px;
}

.preview-box {
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  padding: 10px;
}

.preview-label {
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
}

.preview-image {
  width: 100%;
  max-height: 120px;
  border-radius: 4px;
}

/* Actions */
.actions-section {
  margin-top: auto;
}

.start-btn {
  width: 100%;
  height: 44px;
  background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%);
  border: none;
  color: #1a1a2e;
  font-weight: 600;
}

/* Middle Column */
.middle-column {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #1b263b;
  padding: 16px;
  gap: 16px;
  overflow: hidden;
  min-width: 0;
}

.middle-column .result-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  overflow: hidden;
}

.middle-column .upper-section {
  min-height: 0;
}

.middle-column .lower-section {
  min-height: 0;
}

.section-header {
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.section-header span {
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
  font-weight: 600;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.3);
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.empty-state h3 {
  margin: 0 0 8px 0;
  font-size: 16px;
  color: rgba(255, 255, 255, 0.5);
}

.empty-state p {
  margin: 0;
  font-size: 13px;
}

/* Processing */
.processing-status {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
}

.status-text {
  color: rgba(255, 255, 255, 0.7);
}

/* Result Images */
.result-images {
  flex: 1;
  display: flex;
  gap: 12px;
  padding: 12px;
  min-height: 0;
}

.image-card {
  flex: 1;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.image-header {
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.05);
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
}

.result-image {
  flex: 1;
  width: 100%;
  min-height: 150px;
}

/* Right Column */
.right-column {
  flex: 1;
  min-width: 280px;
  background: #1e3a5f;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 12px;
  flex-shrink: 0;
  overflow-y: auto;
}

.report-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
}

.report-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.report-header .el-icon {
  color: #00d9ff;
}

.report-header span {
  color: #fff;
  font-size: 15px;
  font-weight: 600;
}

/* AI Report */
.ai-report {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  padding: 12px;
  flex: 1;
  overflow-y: auto;
}

.report-text {
  font-size: 12px;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.8);
}

.report-text :deep(h3) {
  font-size: 13px;
  color: #00d9ff;
  margin: 8px 0 6px 0;
}

.report-text :deep(h4) {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  margin: 6px 0 4px 0;
}

.report-text :deep(p) {
  margin: 4px 0;
}

.report-text :deep(strong) {
  color: #00ff88;
}

/* Download */
.download-section {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.download-section .el-button {
  width: 100%;
}

/* File Info */
.file-info {
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.4);
}

.file-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 180px;
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
</style>
