<template>
  <div class="task-selector">
    <el-radio-group v-model="selectedTask" size="large" @change="handleTaskChange">
      <el-radio-button value="segmentation">
        <el-icon><Scissor /></el-icon>
        <span>分割</span>
      </el-radio-button>
      <el-radio-button value="binary">
        <el-icon><Monitor /></el-icon>
        <span>二分类</span>
      </el-radio-button>
      <el-radio-button value="multiclass">
        <el-icon><Grid /></el-icon>
        <span>多分类</span>
      </el-radio-button>
      <el-radio-button value="ai_analysis">
        <el-icon><MagicStick /></el-icon>
        <span>AI 分析</span>
      </el-radio-button>
    </el-radio-group>

    <!-- 任务说明 -->
    <el-alert
      :title="taskConfig[selectedTask].title"
      type="info"
      :closable="false"
      show-icon
      class="task-info"
    >
      <template #default>
        {{ taskConfig[selectedTask].description }}
      </template>
    </el-alert>

    <!-- 高级设置 -->
    <el-collapse v-model="advancedOpen" class="advanced-settings">
      <el-collapse-item title="高级设置" name="advanced">
        <!-- 分割设置 -->
        <div v-if="selectedTask === 'segmentation'" class="setting-group">
          <el-form-item label="Checkpoint 路径">
            <el-input
              v-model="settings.segmentation.checkpoint"
              placeholder="checkpoints/checkpoint_108_linear.pth"
            />
          </el-form-item>
          <el-form-item label="分割阈值">
            <el-slider
              v-model="settings.segmentation.threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :show-tooltip="true"
            />
            <span class="value-display">{{ (settings.segmentation.threshold * 100).toFixed(0) }}%</span>
          </el-form-item>
        </div>

        <!-- 二分类设置 -->
        <div v-if="selectedTask === 'binary'" class="setting-group">
          <el-form-item label="Checkpoint 路径" required>
            <el-input
              v-model="settings.binary.checkpoint"
              placeholder="输入训练好的二分类模型路径"
            />
          </el-form-item>
          <el-form-item label="输入尺寸">
            <el-select v-model="settings.binary.input_size">
              <el-option :value="224" label="224x224" />
            </el-select>
          </el-form-item>
        </div>

        <!-- 多分类设置 -->
        <div v-if="selectedTask === 'multiclass'" class="setting-group">
          <el-form-item label="Checkpoint 路径" required>
            <el-input
              v-model="settings.multiclass.checkpoint"
              placeholder="输入训练好的多分类模型路径"
            />
          </el-form-item>
          <el-form-item label="类别数量" required>
            <el-input-number
              v-model="settings.multiclass.num_labels"
              :min="2"
              :max="100"
            />
          </el-form-item>
          <el-form-item label="输入尺寸">
            <el-select v-model="settings.multiclass.input_size">
              <el-option :value="224" label="224x224" />
            </el-select>
          </el-form-item>
        </div>

        <!-- AI 分析设置 -->
        <div v-if="selectedTask === 'ai_analysis'" class="setting-group">
          <el-form-item label="运行分割模型">
            <el-switch v-model="settings.ai_analysis.run_segmentation" />
          </el-form-item>
          <el-form-item label="运行分类模型">
            <el-switch v-model="settings.ai_analysis.run_classification" />
          </el-form-item>
          <el-form-item label="分割 Checkpoint">
            <el-input
              v-model="settings.ai_analysis.seg_checkpoint"
              placeholder="checkpoints/seg/checkpoint_108_linear.pth"
            />
          </el-form-item>
          <el-form-item label="分类 Checkpoint">
            <el-input
              v-model="settings.ai_analysis.cls_checkpoint"
              placeholder="checkpoints/single_cls/checkpoint_teacher_linear.pth"
            />
          </el-form-item>
          <el-form-item label="分割阈值">
            <el-slider
              v-model="settings.ai_analysis.seg_threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :show-tooltip="true"
            />
            <span class="value-display">{{ (settings.ai_analysis.seg_threshold * 100).toFixed(0) }}%</span>
          </el-form-item>
          <el-form-item label="AI 创意度">
            <el-slider
              v-model="settings.ai_analysis.temperature"
              :min="0"
              :max="1"
              :step="0.1"
              :show-tooltip="true"
            />
            <span class="value-display">{{ settings.ai_analysis.temperature }}</span>
          </el-form-item>
        </div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { Scissor, Monitor, Grid, MagicStick } from '@element-plus/icons-vue'

const emit = defineEmits<{
  taskChange: [task: string, settings: any]
}>()

const selectedTask = ref('segmentation')
const advancedOpen = ref([])

// 组件挂载时触发一次事件，初始化父组件的状态
onMounted(() => {
  emit('taskChange', selectedTask.value, settings[selectedTask.value])
})

const settings = reactive({
  segmentation: {
    checkpoint: 'checkpoints/checkpoint_108_linear.pth',
    threshold: 0.5,
    input_size: 512
  },
  binary: {
    checkpoint: '',
    input_size: 224,
    n_last_blocks: 4,
    avgpool: 0
  },
  multiclass: {
    checkpoint: '',
    num_labels: 5,
    input_size: 224,
    n_last_blocks: 4,
    avgpool: 0
  },
  ai_analysis: {
    run_segmentation: true,
    run_classification: true,
    seg_checkpoint: 'checkpoints/seg/checkpoint_108_linear.pth',
    cls_checkpoint: 'checkpoints/single_cls/checkpoint_teacher_linear.pth',
    seg_threshold: 0.5,
    temperature: 0.7
  }
})

const taskConfig = {
  segmentation: {
    title: '眼底血管分割',
    description: '对眼底图像进行血管分割，输出血管分布掩码。适用于视网膜血管分析、疾病诊断辅助等场景。'
  },
  binary: {
    title: '二分类',
    description: '将图像分为两类，如：健康 vs 病变。适用于糖尿病视网膜病变（DR）筛查等任务。'
  },
  multiclass: {
    title: '多分类',
    description: '将图像分为多个类别，如：疾病分级（无/轻度/中度/重度）。适用于细粒度疾病分类任务。'
  },
  ai_analysis: {
    title: 'AI 智能分析',
    description: '整合 VisionFM 本地模型 + 阿里云百炼多模态大模型，生成综合分析报告。自动进行分割和分类，并提供专业的医学影像分析。'
  }
}

const handleTaskChange = () => {
  emit('taskChange', selectedTask.value, settings[selectedTask.value])
}

// 暴露给父组件
defineExpose({
  getTask: () => selectedTask.value,
  getSettings: () => settings[selectedTask.value]
})
</script>

<style scoped>
.task-selector {
  margin-bottom: 24px;
}

.el-radio-group {
  display: flex;
  width: 100%;
  margin-bottom: 20px;
}

.el-radio-button {
  flex: 1;
}

.el-radio-button :deep(.el-radio-button__inner) {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.task-info {
  margin-bottom: 20px;
}

.advanced-settings {
  margin-top: 16px;
}

.setting-group {
  padding: 10px 0;
}

.el-form-item {
  margin-bottom: 16px;
}

.value-display {
  margin-left: 12px;
  color: var(--el-text-color-secondary);
  font-size: 14px;
}
</style>
