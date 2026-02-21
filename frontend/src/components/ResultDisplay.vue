<template>
  <div class="result-container">
    <el-alert
      title="分割完成"
      type="success"
      :closable="false"
      show-icon
      class="result-alert"
    >
      <template #default>
        <span>使用阈值: {{ (threshold * 100).toFixed(0) }}% | 输出尺寸: {{ shape }}</span>
      </template>
    </el-alert>

    <div class="images-display">
      <!-- 原图 -->
      <div class="image-box">
        <h4 class="image-title">原始图片</h4>
        <div class="image-wrapper">
          <el-image :src="originalImage" fit="contain" class="result-image" />
        </div>
      </div>

      <!-- 纯掩码 -->
      <div class="image-box">
        <h4 class="image-title">血管分割掩码</h4>
        <div class="image-wrapper">
          <el-image :src="maskImage" fit="contain" class="result-image" />
        </div>
      </div>
    </div>

    <!-- 控制面板 -->
    <div class="controls-panel">
      <div class="button-group">
        <el-button type="primary" @click="downloadOriginal">
          <el-icon><Download /></el-icon>
          下载原图
        </el-button>
        <el-button type="success" @click="downloadMask">
          <el-icon><Download /></el-icon>
          下载掩码
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Download } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

interface Props {
  originalImage: string
  maskImage: string
  threshold?: number
  shape?: string
}

const props = withDefaults(defineProps<Props>(), {
  threshold: 0.5,
  shape: '512x512'
})

// 下载图片辅助函数
const downloadImage = (url: string, filename: string) => {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  ElMessage.success(`已下载: ${filename}`)
}

const downloadMask = () => {
  downloadImage(props.maskImage, `visionfm_mask_${Date.now()}.png`)
}

const downloadOriginal = () => {
  downloadImage(props.originalImage, `visionfm_original_${Date.now()}.png`)
}
</script>

<style scoped>
.result-container {
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 16px;
}

.result-alert {
  margin-bottom: 20px;
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
  min-height: 300px;
}

.result-image {
  width: 100%;
  height: auto;
  max-height: 500px;
  object-fit: contain;
}

.controls-panel {
  padding: 20px;
  background: var(--el-fill-color-lighter);
  border-radius: 8px;
  text-align: center;
}

.button-group {
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.button-group .el-button {
  min-width: 140px;
}

/* 平板设备 (768px - 1024px) */
@media (max-width: 1024px) {
  .result-container {
    max-width: 100%;
  }

  .images-display {
    gap: 16px;
  }

  .result-image {
    max-height: 400px;
  }
}

/* 移动设备 (< 768px) */
@media (max-width: 768px) {
  .result-container {
    padding: 0 12px;
  }

  .result-alert {
    font-size: 14px;
  }

  .images-display {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .image-wrapper {
    padding: 12px;
    min-height: 250px;
  }

  .result-image {
    max-height: 350px;
  }

  .image-title {
    font-size: 13px;
    padding: 10px 12px;
  }

  .controls-panel {
    padding: 16px;
  }

  .button-group {
    flex-direction: column;
  }

  .button-group .el-button {
    width: 100%;
    min-width: auto;
  }
}

/* 小屏手机 (< 480px) */
@media (max-width: 480px) {
  .result-container {
    padding: 0 8px;
  }

  .image-wrapper {
    padding: 8px;
    min-height: 200px;
  }

  .result-image {
    max-height: 280px;
  }

  .controls-panel {
    padding: 12px;
  }
}
</style>
