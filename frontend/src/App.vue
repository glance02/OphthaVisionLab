<template>
  <div class="app">
    <el-container class="main-container">
      <!-- Header -->
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo-section">
            <el-icon class="logo-icon"><View /></el-icon>
            <h1 class="app-title">VisionFM</h1>
          </div>
          <div class="header-subtitle">眼科人工智能多任务系统</div>
        </div>
      </el-header>

      <!-- Main Content -->
      <el-main class="app-main">
        <!-- 任务选择器 -->
        <el-card shadow="hover" class="task-selector-card">
          <TaskSelector
            ref="taskSelectorRef"
            @task-change="handleTaskChange"
          />
        </el-card>

        <!-- 主功能卡片 -->
        <el-card class="main-card" shadow="hover">
          <!-- 上传区域 -->
          <div v-if="!processing && !result" class="upload-section">
            <el-alert
              :title="uploadTitle"
              type="info"
              :closable="false"
              show-icon
              class="section-alert"
            >
              <template #default>
                支持 JPG/PNG 格式，建议图片清晰、对比度适中
              </template>
            </el-alert>

            <ImageUpload @upload="handleUpload" />

            <!-- 检查点警告（非分割任务） -->
            <el-alert
              v-if="currentTask !== 'segmentation' && !checkpointReady"
              title="注意：需要提供模型checkpoint"
              type="warning"
              :closable="false"
              show-icon
              class="warning-alert"
            >
              <template #default>
                请在"高级设置"中提供训练好的模型checkpoint路径，否则无法进行预测。
              </template>
            </el-alert>
          </div>

          <!-- 处理中状态 -->
          <div v-if="processing" class="processing-section">
            <el-alert
              title="正在处理"
              type="warning"
              :closable="false"
              show-icon
              class="section-alert"
            >
              <template #default>
                {{ processingMessage }}
              </template>
            </el-alert>

            <div class="progress-container">
              <el-progress
                :percentage="progress"
                :status="progressStatus"
                :stroke-width="20"
                :text-inside="true"
              />
              <p class="status-message">{{ statusMessage }}</p>
            </div>

            <div v-if="uploadedFile" class="file-info">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="文件名">
                  {{ uploadedFile.name }}
                </el-descriptions-item>
                <el-descriptions-item label="文件大小">
                  {{ (uploadedFile.size / 1024).toFixed(2) }} KB
                </el-descriptions-item>
                <el-descriptions-item label="任务类型">
                  {{ taskName }}
                </el-descriptions-item>
              </el-descriptions>
            </div>
          </div>

          <!-- 分割结果展示 -->
          <div v-if="result && currentTask === 'segmentation'" class="result-section">
            <ResultDisplay
              :original-image="result.data.originalImage"
              :mask-image="result.data.maskImage"
              :threshold="result.data.threshold"
              :shape="result.data.shape"
            />

            <div class="action-buttons">
              <el-button size="large" @click="handleNewTask">
                <el-icon><Plus /></el-icon>
                处理新图片
              </el-button>
            </div>
          </div>

          <!-- 分类结果展示 -->
          <div v-if="result && (currentTask === 'binary' || currentTask === 'multiclass')" class="result-section">
            <ClassificationResult
              :result="result.data"
              :task-type="currentTask"
              @new-prediction="handleNewTask"
            />
          </div>
        </el-card>

        <!-- Info Cards -->
        <el-row :gutter="20" class="info-row">
          <el-col :xs="24" :sm="12" :md="8">
            <el-card shadow="hover" class="info-card">
              <template #header>
                <div class="card-header">
                  <el-icon><InfoFilled /></el-icon>
                  <span>模型信息</span>
                </div>
              </template>
              <div class="info-content">
                <p><strong>架构:</strong> Vision Transformer (ViT-Base)</p>
                <p><strong>预训练:</strong> 350万+ 眼科图像</p>
                <p><strong>任务:</strong> {{ taskName }}</p>
              </div>
            </el-card>
          </el-col>

          <el-col :xs="24" :sm="12" :md="8">
            <el-card shadow="hover" class="info-card">
              <template #header>
                <div class="card-header">
                  <el-icon><DataLine /></el-icon>
                  <span>性能指标</span>
                </div>
              </template>
              <div class="info-content">
                <p v-if="currentTask === 'segmentation'"><strong>Dice Score:</strong> ~78.17%</p>
                <p v-else><strong>AUC:</strong> 取决于具体任务</p>
                <p><strong>支持:</strong> GPU / CPU 推理</p>
                <p><strong>响应:</strong> 通常 < 5秒</p>
              </div>
            </el-card>
          </el-col>

          <el-col :xs="24" :sm="12" :md="8">
            <el-card shadow="hover" class="info-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Document /></el-icon>
                  <span>使用说明</span>
                </div>
              </template>
              <div class="info-content">
                <p>1. 选择任务类型（分割/分类）</p>
                <p>2. 上传眼底图像</p>
                <p>3. 等待处理完成</p>
                <p>4. 查看和下载结果</p>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>

      <!-- Footer -->
      <el-footer class="app-footer">
        <p>VisionFM - 眼科人工智能通用基础模型 | 基于 NEJM AI 发表论文</p>
        <p class="footer-note">支持：眼底分割、二分类、多分类 | 仅供研究和演示使用</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import axios from 'axios'
import { View, Plus, InfoFilled, DataLine, Document } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import TaskSelector from './components/TaskSelector.vue'
import ImageUpload from './components/ImageUpload.vue'
import ResultDisplay from './components/ResultDisplay.vue'
import ClassificationResult from './components/ClassificationResult.vue'

// API 配置
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// 状态管理
const taskSelectorRef = ref()
const currentTask = ref('segmentation')
const currentSettings = ref<any>({
  checkpoint: 'checkpoints/checkpoint_108_linear.pth',
  threshold: 0.5,
  input_size: 512
})
const processing = ref(false)
const progress = ref(0)
const progressStatus = ref<'success' | 'exception' | 'warning' | ''>('')
const statusMessage = ref('')
const uploadedFile = ref<File | null>(null)
const result = ref<any>(null)

// 计算属性
const taskName = computed(() => {
  const names = {
    segmentation: '眼底血管分割',
    binary: '二分类',
    multiclass: '多分类'
  }
  return names[currentTask.value] || currentTask.value
})

const uploadTitle = computed(() => {
  return `上传眼底图像 - ${taskName.value}`
})

const processingMessage = computed(() => {
  const messages = {
    segmentation: '正在进行血管分割...',
    binary: '正在进行二分类预测...',
    multiclass: '正在进行多分类预测...'
  }
  return messages[currentTask.value] || '正在处理...'
})

const checkpointReady = computed(() => {
  if (currentTask.value === 'segmentation') {
    return true // 分割有默认checkpoint
  }
  return currentSettings.value?.checkpoint && currentSettings.value.checkpoint.length > 0
})

// 事件处理
const handleTaskChange = (task: string, settings: any) => {
  currentTask.value = task
  currentSettings.value = settings
  // 重置结果
  result.value = null
}

const handleUpload = async (file: File) => {
  // 检查非分割任务是否有checkpoint
  if (currentTask.value !== 'segmentation' && !checkpointReady.value) {
    ElMessage.error('请先在"高级设置"中提供模型checkpoint路径')
    return
  }

  uploadedFile.value = file
  processing.value = true
  progress.value = 0
  progressStatus.value = ''
  statusMessage.value = '准备上传...'

  const formData = new FormData()
  formData.append('file', file)

  // 添加任务特定参数
  if (currentTask.value === 'segmentation') {
    formData.append('checkpoint', currentSettings.value.checkpoint)
    formData.append('threshold', currentSettings.value.threshold.toString())
    formData.append('input_size', currentSettings.value.input_size.toString())
  } else if (currentTask.value === 'binary') {
    formData.append('checkpoint', currentSettings.value.checkpoint)
    formData.append('input_size', currentSettings.value.input_size.toString())
    formData.append('n_last_blocks', currentSettings.value.n_last_blocks.toString())
    formData.append('avgpool', currentSettings.value.avgpool.toString())
  } else if (currentTask.value === 'multiclass') {
    formData.append('checkpoint', currentSettings.value.checkpoint)
    formData.append('num_labels', currentSettings.value.num_labels.toString())
    formData.append('input_size', currentSettings.value.input_size.toString())
    formData.append('n_last_blocks', currentSettings.value.n_last_blocks.toString())
    formData.append('avgpool', currentSettings.value.avgpool.toString())
  }

  try {
    // 确定API端点
    const endpoint = currentTask.value === 'segmentation'
      ? '/api/segment'
      : currentTask.value === 'binary'
      ? '/api/classify/binary'
      : '/api/classify/multiclass'

    // 上传进度
    const uploadProgress = (progressEvent: any) => {
      if (progressEvent.total) {
        const percentCompleted = Math.round((progressEvent.loaded * 30) / progressEvent.total)
        progress.value = Math.min(percentCompleted, 30)
      }
    }

    statusMessage.value = '正在上传图像...'

    const response = await axios.post(`${API_BASE}${endpoint}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: uploadProgress,
      timeout: 120000
    })

    // 模拟处理进度
    progress.value = 50
    statusMessage.value = '正在加载模型...'

    await new Promise(resolve => setTimeout(resolve, 500))
    progress.value = 70
    statusMessage.value = processingMessage.value

    await new Promise(resolve => setTimeout(resolve, 800))
    progress.value = 100
    progressStatus.value = 'success'
    statusMessage.value = '处理完成！'

    result.value = response.data

    ElMessage.success({
      message: `${taskName.value}完成！`,
      duration: 3000
    })

  } catch (error: any) {
    console.error('处理失败:', error)
    progressStatus.value = 'exception'

    const errorMessage = error.response?.data?.detail || error.message || '未知错误'
    ElMessage.error({
      message: `处理失败: ${errorMessage}`,
      duration: 5000
    })

    setTimeout(() => {
      if (!result.value) {
        processing.value = false
        progress.value = 0
      }
    }, 3000)
  } finally {
    if (result.value) {
      setTimeout(() => {
        processing.value = false
      }, 500)
    }
  }
}

const handleNewTask = () => {
  result.value = null
  progress.value = 0
  progressStatus.value = ''
  statusMessage.value = ''
  uploadedFile.value = null
}
</script>

<style scoped>
.app {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.main-container {
  min-height: 100vh;
}

.app-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 !important;
  height: 80px !important;
}

.header-content {
  text-align: center;
}

.logo-section {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 4px;
}

.logo-icon {
  font-size: 32px;
  color: #409eff;
}

.app-title {
  margin: 0;
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-subtitle {
  font-size: 14px;
  color: #909399;
  margin-top: 4px;
}

.app-main {
  padding: 30px 20px !important;
  max-width: 1400px;
  margin: 0 auto;
}

.task-selector-card {
  border-radius: 12px;
  margin-bottom: 20px;
}

.main-card {
  border-radius: 12px;
  margin-bottom: 30px;
}

.section-alert {
  margin-bottom: 24px;
}

.warning-alert {
  margin-top: 20px;
}

.upload-section,
.processing-section,
.result-section {
  padding: 20px 0;
}

.progress-container {
  margin: 40px 0;
}

.status-message {
  text-align: center;
  margin-top: 20px;
  font-size: 16px;
  color: #606266;
}

.file-info {
  margin-top: 30px;
}

.action-buttons {
  margin-top: 30px;
  text-align: center;
}

.info-row {
  margin-top: 30px;
}

.info-card {
  border-radius: 8px;
  margin-bottom: 20px;
  transition: transform 0.3s;
}

.info-card:hover {
  transform: translateY(-4px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.info-content p {
  margin: 8px 0;
  color: #606266;
  font-size: 14px;
}

.app-footer {
  background: rgba(255, 255, 255, 0.9);
  text-align: center;
  color: #606266;
  padding: 20px !important;
  border-top: 1px solid #e4e7ed;
}

.app-footer p {
  margin: 4px 0;
}

.footer-note {
  font-size: 12px;
  color: #909399;
}

@media (max-width: 768px) {
  .app-title {
    font-size: 22px;
  }

  .app-main {
    padding: 15px 10px !important;
  }

  .info-card {
    margin-bottom: 15px;
  }
}
</style>
