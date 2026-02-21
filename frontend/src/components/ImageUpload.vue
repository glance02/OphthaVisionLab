<template>
  <div class="upload-container">
    <el-upload
      class="upload-area"
      drag
      :auto-upload="false"
      :show-file-list="false"
      accept=".jpg,.jpeg,.png"
      :on-change="handleFileChange"
      :limit="1"
      :disabled="disabled"
    >
      <el-icon class="upload-icon"><UploadFilled /></el-icon>
      <div class="upload-text">拖拽图片到此处或点击上传</div>
      <div class="upload-hint">支持 JPG/PNG 格式，最大 10MB</div>
    </el-upload>

    <!-- 预览区域 -->
    <div v-if="previewUrl" class="preview-area">
      <div class="preview-header">
        <span>已选择图片</span>
        <el-button type="danger" text @click="clearPreview">清除</el-button>
      </div>
      <el-image :src="previewUrl" fit="contain" class="preview-image" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

interface Props {
  disabled?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false
})

const emit = defineEmits<{
  upload: [file: File]
}>()

const previewUrl = ref<string>('')

const handleFileChange = (file: any) => {
  // 验证大小
  if (file.raw.size > 10 * 1024 * 1024) {
    ElMessage.error('图片大小不能超过 10MB')
    return
  }

  // 验证类型
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png']
  if (!validTypes.includes(file.raw.type)) {
    ElMessage.error('仅支持 JPG/PNG 格式')
    return
  }

  // 设置预览
  previewUrl.value = URL.createObjectURL(file.raw)

  // 发射事件
  emit('upload', file.raw)
}

const clearPreview = () => {
  previewUrl.value = ''
}
</script>

<style scoped>
.upload-container {
  width: 100%;
}

.upload-area {
  padding: 40px;
  border: 2px dashed var(--el-border-color);
  border-radius: 8px;
  text-align: center;
  transition: border-color 0.3s;
}

.upload-area:hover {
  border-color: var(--el-color-primary);
}

.upload-icon {
  font-size: 67px;
  color: var(--el-color-primary);
  margin-bottom: 16px;
}

.upload-text {
  font-size: 16px;
  color: var(--el-text-color-primary);
  margin-bottom: 8px;
}

.upload-hint {
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.preview-area {
  margin-top: 20px;
  border: 1px solid var(--el-border-color);
  border-radius: 8px;
  padding: 16px;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.preview-image {
  width: 100%;
  max-height: 400px;
}
</style>
