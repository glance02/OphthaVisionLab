<template>
  <div class="classification-result">
    <el-alert
      :title="taskType === 'binary' ? '二分类结果' : '多分类结果'"
      :type="result.confidence > 0.7 ? 'success' : 'warning'"
      :closable="false"
      show-icon
      class="result-alert"
    >
      <template #default>
        <div class="result-summary">
          <div class="main-result">
            <span class="label">预测结果:</span>
            <span class="value">{{ displayLabel }}</span>
          </div>
          <div class="confidence">
            <span class="label">置信度:</span>
            <el-progress
              :percentage="Math.round(result.confidence * 100)"
              :color="getConfidenceColor(result.confidence)"
              :stroke-width="20"
              :text-inside="true"
            />
          </div>
        </div>
      </template>
    </el-alert>

    <!-- 概率分布（多分类） -->
    <div v-if="taskType === 'multiclass' && result.probabilities" class="probability-chart">
      <h4>各类别概率分布</h4>
      <div class="prob-bars">
        <div
          v-for="(prob, index) in result.probabilities"
          :key="index"
          class="prob-bar-wrapper"
        >
          <span class="class-label">类别 {{ index }}:</span>
          <el-progress
            :percentage="Math.round(prob * 100)"
            :color="getProbColor(prob, index)"
            :show-text="true"
          />
        </div>
      </div>
    </div>

    <!-- 详细信息 -->
    <el-descriptions :column="2" border class="result-details">
      <el-descriptions-item label="任务类型">
        {{ taskType === 'binary' ? '二分类' : '多分类' }}
      </el-descriptions-item>
      <el-descriptions-item label="预测类别">
        {{ result.predicted_class }}
      </el-descriptions-item>
      <el-descriptions-item v-if="taskType === 'binary'" label="阳性概率">
        {{ (result.probability * 100).toFixed(2) }}%
      </el-descriptions-item>
      <el-descriptions-item label="置信度">
        {{ (result.confidence * 100).toFixed(2) }}%
      </el-descriptions-item>
      <el-descriptions-item v-if="taskType === 'multiclass'" label="类别数量" :span="2">
        {{ result.num_classes }}
      </el-descriptions-item>
    </el-descriptions>

    <!-- 操作按钮 -->
    <div class="action-buttons">
      <el-button type="primary" @click="$emit('new-prediction')">
        <el-icon><Plus /></el-icon>
        新预测
      </el-button>
      <el-button @click="downloadResult">
        <el-icon><Download /></el-icon>
        下载结果
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Plus, Download } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

interface Props {
  result: any
  taskType: 'binary' | 'multiclass'
}

const props = defineProps<Props>()

const emit = defineEmits<{
  'new-prediction': []
}>()

const displayLabel = computed(() => {
  if (props.taskType === 'binary') {
    return props.result.predicted_class === 1 ? '阳性' : '阴性'
  } else {
    return `类别 ${props.result.predicted_class}`
  }
})

const getConfidenceColor = (confidence: number) => {
  if (confidence > 0.8) return '#67c23a'
  if (confidence > 0.6) return '#e6a23c'
  return '#f56c6c'
}

const getProbColor = (prob: number, index: number) => {
  if (index === props.result.predicted_class) {
    return '#409eff'
  }
  const colors = ['#909399', '#c8c9cc', '#e4e7ed']
  return colors[index % colors.length]
}

const downloadResult = () => {
  const resultText = JSON.stringify(props.result, null, 2)
  const blob = new Blob([resultText], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `classification_result_${Date.now()}.json`
  link.click()
  URL.revokeObjectURL(url)
  ElMessage.success('结果已下载')
}
</script>

<style scoped>
.classification-result {
  width: 100%;
}

.result-alert {
  margin-bottom: 24px;
}

.result-summary {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.main-result {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 16px;
}

.main-result .label {
  font-weight: 600;
  color: var(--el-text-color-secondary);
}

.main-result .value {
  font-size: 20px;
  font-weight: 700;
  color: var(--el-color-primary);
}

.confidence .label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: var(--el-text-color-secondary);
}

.probability-chart {
  margin-bottom: 24px;
  padding: 20px;
  background: var(--el-fill-color-lighter);
  border-radius: 8px;
}

.probability-chart h4 {
  margin: 0 0 16px 0;
  font-size: 14px;
  color: var(--el-text-color-primary);
}

.prob-bars {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.prob-bar-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}

.class-label {
  min-width: 60px;
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.result-details {
  margin-bottom: 24px;
}

.action-buttons {
  display: flex;
  gap: 12px;
  justify-content: center;
}
</style>
