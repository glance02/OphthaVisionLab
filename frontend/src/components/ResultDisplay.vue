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

    <div class="image-comparison">
      <!-- 原图 -->
      <div class="image-box">
        <h4 class="image-title">原图</h4>
        <div class="image-wrapper">
          <el-image :src="originalImage" fit="contain" class="result-image" />
        </div>
      </div>

      <!-- 叠加结果 -->
      <div class="image-box">
        <h4 class="image-title">分割结果</h4>
        <div class="image-wrapper overlay-wrapper">
          <el-image :src="originalImage" fit="contain" class="result-image base-image" />
          <el-image
            :src="maskImage"
            fit="contain"
            class="result-image mask-image"
            :style="{ opacity: maskOpacity }"
          />
        </div>
      </div>

      <!-- 纯掩码 -->
      <div class="image-box">
        <h4 class="image-title">纯掩码</h4>
        <div class="image-wrapper">
          <el-image :src="maskImage" fit="contain" class="result-image" />
        </div>
      </div>
    </div>

    <!-- 控制面板 -->
    <div class="controls-panel">
      <div class="slider-container">
        <span class="slider-label">掩码透明度: {{ (maskOpacity * 100).toFixed(0) }}%</span>
        <el-slider
          v-model="maskOpacity"
          :min="0"
          :max="1"
          :step="0.01"
          :show-tooltip="false"
        />
      </div>

      <div class="button-group">
        <el-button type="primary" @click="downloadOriginal">
          <el-icon><Download /></el-icon>
          下载原图
        </el-button>
        <el-button type="success" @click="downloadMask">
          <el-icon><Download /></el-icon>
          下载掩码
        </el-button>
        <el-button type="warning" @click="downloadOverlay">
          <el-icon><Download /></el-icon>
          下载叠加图
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
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

const maskOpacity = ref(0.6)

// 下载图片辅助函数
const downloadImage = (url: string, filename: string) => {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  ElMessage.success(`已下载: ${filename}`)
}

const downloadOriginal = () => {
  downloadImage(props.originalImage, `visionfm_original_${Date.now()}.png`)
}

const downloadMask = () => {
  downloadImage(props.maskImage, `visionfm_mask_${Date.now()}.png`)
}

const downloadOverlay = async () => {
  // 创建 Canvas 来合并原图和掩码
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const img1 = new Image()
  const img2 = new Image()

  img1.onload = () => {
    canvas.width = img1.width
    canvas.height = img1.height
    ctx.drawImage(img1, 0, 0)

    img2.onload = () => {
      ctx.globalAlpha = maskOpacity.value
      ctx.drawImage(img2, 0, 0)

      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob)
          downloadImage(url, `visionfm_overlay_${Date.now()}.png`)
          URL.revokeObjectURL(url)
        }
      }, 'image/png')
    }
    img2.src = props.maskImage
  }
  img1.src = props.originalImage
}
</script>

<style scoped>
.result-container {
  width: 100%;
}

.result-alert {
  margin-bottom: 20px;
}

.image-comparison {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
  position: relative;
  padding: 16px;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.overlay-wrapper {
  position: relative;
}

.result-image {
  width: 100%;
  height: auto;
  max-height: 280px;
}

.mask-image {
  position: absolute;
  top: 16px;
  left: 16px;
  right: 16px;
  bottom: 16px;
  pointer-events: none;
}

.controls-panel {
  padding: 20px;
  background: var(--el-fill-color-lighter);
  border-radius: 8px;
}

.slider-container {
  margin-bottom: 20px;
}

.slider-label {
  display: block;
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 500;
  color: var(--el-text-color-primary);
}

.button-group {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.button-group .el-button {
  flex: 1;
  min-width: 120px;
}
</style>
