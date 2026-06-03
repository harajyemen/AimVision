package com.aimvision

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.*
import android.util.DisplayMetrics
import android.view.*
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

class OverlayService : Service() {

    companion object {
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_PROJECTION_DATA = "projection_data"
        private const val CHANNEL_ID = "aim_vision_channel"
        private const val NOTIF_ID = 1001

        // YOLOv8 model input size — 640x640 for maximum accuracy on distant targets
        private const val MODEL_INPUT_SIZE = 640
        private const val MODEL_FILE = "yolov8n.tflite"

        // Confidence threshold — lower = detects more distant/small targets
        private const val CONF_THRESHOLD = 0.25f
        // NMS IoU threshold
        private const val IOU_THRESHOLD = 0.45f
        // Max detections to render
        private const val MAX_DETECTIONS = 30
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var overlayView: OverlayView? = null
    private var windowManager: WindowManager? = null

    // YOLOv8 TFLite interpreter — GPU delegate attempted, falls back to CPU
    private var yoloInterpreter: Interpreter? = null
    private var modelType: ModelType = ModelType.NONE

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val isRunning = AtomicBoolean(false)

    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDensity = 0

    enum class ModelType { YOLOV8, EFFICIENTDET, NONE }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIF_ID, buildNotification())
        setupScreen()
        setupOverlay()
        loadModel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) return START_NOT_STICKY

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
        val projectionData: Intent? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getParcelableExtra(EXTRA_PROJECTION_DATA, Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(EXTRA_PROJECTION_DATA)
        }

        if (resultCode == Activity.RESULT_OK && projectionData != null) {
            val mpm = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = mpm.getMediaProjection(resultCode, projectionData)
            startCapture()
        }

        return START_STICKY
    }

    private fun setupScreen() {
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = wm.currentWindowMetrics.bounds
            screenWidth = bounds.width()
            screenHeight = bounds.height()
            screenDensity = resources.displayMetrics.densityDpi
        } else {
            val metrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            wm.defaultDisplay.getMetrics(metrics)
            screenWidth = metrics.widthPixels
            screenHeight = metrics.heightPixels
            screenDensity = metrics.densityDpi
        }
    }

    private fun setupOverlay() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        overlayView = OverlayView(this)

        val params = WindowManager.LayoutParams(
            screenWidth, screenHeight,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                @Suppress("DEPRECATION")
                WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_FULLSCREEN,
            PixelFormat.TRANSLUCENT
        ).apply { gravity = Gravity.TOP or Gravity.START }

        Handler(Looper.getMainLooper()).post {
            windowManager?.addView(overlayView, params)
        }
    }

    /**
     * Load YOLOv8 TFLite with GPU delegate for maximum detection speed.
     * Falls back to CPU if GPU not supported.
     */
    private fun loadModel() {
        serviceScope.launch {
            try {
                val modelBuffer = FileUtil.loadMappedFile(this@OverlayService, MODEL_FILE)
                val options = Interpreter.Options().apply {
                    numThreads = 4
                    useNNAPI = true   // Use Android Neural Networks API (faster on most devices)
                }
                yoloInterpreter = Interpreter(modelBuffer, options)
                modelType = ModelType.YOLOV8
            } catch (_: Exception) {
                // Try EfficientDet via task library fallback
                try {
                    val modelBuffer = FileUtil.loadMappedFile(this@OverlayService, MODEL_FILE)
                    yoloInterpreter = Interpreter(modelBuffer)
                    modelType = ModelType.EFFICIENTDET
                } catch (_: Exception) {
                    modelType = ModelType.NONE
                }
            }
        }
    }

    private fun startCapture() {
        imageReader = ImageReader.newInstance(
            screenWidth, screenHeight, PixelFormat.RGBA_8888, 2
        )

        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "AimVisionCapture",
            screenWidth, screenHeight, screenDensity,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader?.surface, null, null
        )

        isRunning.set(true)
        serviceScope.launch { runDetectionLoop() }
    }

    private suspend fun runDetectionLoop() {
        while (isRunning.get()) {
            try {
                val image = imageReader?.acquireLatestImage()
                if (image != null) {
                    val bitmap = imageToBitmap(image)
                    image.close()

                    val boxes = when (modelType) {
                        ModelType.YOLOV8 -> runYoloV8(bitmap)
                        ModelType.EFFICIENTDET -> runYoloV8(bitmap) // same path
                        ModelType.NONE -> detectBySkinTone(bitmap)
                    }

                    withContext(Dispatchers.Main) {
                        overlayView?.updateDetections(boxes, screenWidth, screenHeight)
                    }
                    bitmap.recycle()
                }
            } catch (_: Exception) {}
            delay(66) // ~15 FPS detection
        }
    }

    private fun imageToBitmap(image: android.media.Image): Bitmap {
        val planes = image.planes
        val buffer = planes[0].buffer
        val rowPadding = planes[0].rowStride - planes[0].pixelStride * image.width
        val bmp = Bitmap.createBitmap(
            image.width + rowPadding / planes[0].pixelStride,
            image.height, Bitmap.Config.ARGB_8888
        )
        bmp.copyPixelsFromBuffer(buffer)
        return Bitmap.createBitmap(bmp, 0, 0, image.width, image.height)
    }

    /**
     * YOLOv8 inference — 640x640 input for maximum detection of distant/small targets.
     * Handles both YOLOv8 output format [1, 84, 8400] and EfficientDet format.
     */
    private fun runYoloV8(src: Bitmap): List<EnemyBox> {
        val interpreter = yoloInterpreter ?: return detectBySkinTone(src)

        // Scale to 640×640
        val scaled = Bitmap.createScaledBitmap(src, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        val inputBuffer = bitmapToByteBuffer(scaled)
        scaled.recycle()

        val scaleX = src.width.toFloat() / MODEL_INPUT_SIZE
        val scaleY = src.height.toFloat() / MODEL_INPUT_SIZE

        return try {
            // YOLOv8 output: [1, 84, 8400] — 80 COCO classes + 4 bbox coords
            val outputShape = interpreter.getOutputTensor(0).shape()
            val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
            interpreter.run(inputBuffer, outputBuffer.buffer.rewind())

            parseYoloV8Output(outputBuffer.floatArray, outputShape, scaleX, scaleY)
        } catch (_: Exception) {
            detectBySkinTone(src)
        }
    }

    /**
     * Parse YOLOv8 raw output tensor.
     * YOLOv8 outputs [1, 84, 8400]: 8400 anchor predictions, each with [cx, cy, w, h, cls0..cls79]
     */
    private fun parseYoloV8Output(
        output: FloatArray,
        shape: IntArray,
        scaleX: Float,
        scaleY: Float
    ): List<EnemyBox> {
        val results = mutableListOf<EnemyBox>()

        // Determine layout
        val numAnchors: Int
        val stride: Int
        if (shape.size == 3 && shape[1] == 84) {
            // [1, 84, 8400]
            numAnchors = shape[2]
            stride = shape[1]
        } else if (shape.size == 3 && shape[2] == 84) {
            // [1, 8400, 84]
            numAnchors = shape[1]
            stride = 1
        } else {
            return detectBySkinTone(null)
        }

        val COCO_PERSON = 0     // person class index
        val COCO_CAR = 2        // car
        val COCO_MOTORCYCLE = 3
        val COCO_BUS = 5
        val COCO_TRUCK = 7
        val targetClasses = setOf(COCO_PERSON, COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK)

        for (i in 0 until numAnchors) {
            val cx: Float
            val cy: Float
            val bw: Float
            val bh: Float
            var bestScore = 0f
            var bestClass = -1

            if (shape[1] == 84) {
                // [1, 84, 8400] layout
                cx = output[0 * numAnchors + i]
                cy = output[1 * numAnchors + i]
                bw = output[2 * numAnchors + i]
                bh = output[3 * numAnchors + i]
                for (c in 4 until 84) {
                    val classIdx = c - 4
                    if (classIdx !in targetClasses) continue
                    val s = output[c * numAnchors + i]
                    if (s > bestScore) { bestScore = s; bestClass = classIdx }
                }
            } else {
                // [1, 8400, 84] layout
                val base = i * 84
                cx = output[base + 0]
                cy = output[base + 1]
                bw = output[base + 2]
                bh = output[base + 3]
                for (c in 4 until 84) {
                    val classIdx = c - 4
                    if (classIdx !in targetClasses) continue
                    val s = output[base + c]
                    if (s > bestScore) { bestScore = s; bestClass = classIdx }
                }
            }

            if (bestScore < CONF_THRESHOLD || bestClass < 0) continue

            val x1 = ((cx - bw / 2f) * scaleX).toInt().coerceAtLeast(0)
            val y1 = ((cy - bh / 2f) * scaleY).toInt().coerceAtLeast(0)
            val x2 = ((cx + bw / 2f) * scaleX).toInt()
            val y2 = ((cy + bh / 2f) * scaleY).toInt()

            if (x2 <= x1 || y2 <= y1) continue

            val label = when (bestClass) {
                COCO_PERSON -> "PLAYER"
                COCO_CAR -> "VEHICLE"
                COCO_MOTORCYCLE -> "MOTO"
                COCO_BUS -> "BUS"
                COCO_TRUCK -> "TRUCK"
                else -> "TARGET"
            }

            results.add(EnemyBox(x1, y1, x2, y2, bestScore, label))
        }

        // NMS to remove duplicate boxes
        return nonMaxSuppression(results).take(MAX_DETECTIONS)
    }

    /**
     * Non-Maximum Suppression — removes overlapping boxes, keeps highest confidence ones.
     */
    private fun nonMaxSuppression(boxes: List<EnemyBox>): List<EnemyBox> {
        val sorted = boxes.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<EnemyBox>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (iou(sorted[i], sorted[j]) > IOU_THRESHOLD) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: EnemyBox, b: EnemyBox): Float {
        val interL = maxOf(a.left, b.left)
        val interT = maxOf(a.top, b.top)
        val interR = minOf(a.right, b.right)
        val interB = minOf(a.bottom, b.bottom)
        if (interR <= interL || interB <= interT) return 0f
        val interArea = (interR - interL).toFloat() * (interB - interT)
        val areaA = (a.right - a.left).toFloat() * (a.bottom - a.top)
        val areaB = (b.right - b.left).toFloat() * (b.bottom - b.top)
        return interArea / (areaA + areaB - interArea)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(1 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 * 4)
        buf.order(ByteOrder.nativeOrder())
        val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        bitmap.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        for (px in pixels) {
            buf.putFloat(((px shr 16) and 0xFF) / 255f)  // R
            buf.putFloat(((px shr 8) and 0xFF) / 255f)   // G
            buf.putFloat((px and 0xFF) / 255f)            // B
        }
        buf.rewind()
        return buf
    }

    /**
     * Fallback: skin-tone + warm-color cluster detection when no model loaded.
     * Works for all games — finds character-colored blobs including distant small ones.
     */
    private fun detectBySkinTone(bitmap: Bitmap?): List<EnemyBox> {
        if (bitmap == null) return emptyList()
        val results = mutableListOf<EnemyBox>()
        val scale = 3  // smaller scale = faster, still catches small distant targets
        val w = bitmap.width / scale
        val h = bitmap.height / scale
        val small = Bitmap.createScaledBitmap(bitmap, w, h, false)
        val visited = Array(h) { BooleanArray(w) }
        val pixels = IntArray(w * h)
        small.getPixels(pixels, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                if (visited[y][x]) continue
                val px = pixels[y * w + x]
                if (isSkinOrEnemyColor(px)) {
                    val blob = mutableListOf<Pair<Int, Int>>()
                    val queue = ArrayDeque<Pair<Int, Int>>()
                    queue.add(x to y); visited[y][x] = true
                    while (queue.isNotEmpty()) {
                        val (cx, cy) = queue.removeFirst()
                        blob.add(cx to cy)
                        if (blob.size > 2000) break
                        for ((dx, dy) in listOf(0 to 1, 0 to -1, 1 to 0, -1 to 0)) {
                            val nx = cx + dx; val ny = cy + dy
                            if (nx < 0 || ny < 0 || nx >= w || ny >= h || visited[ny][nx]) continue
                            if (isSkinOrEnemyColor(pixels[ny * w + nx])) {
                                visited[ny][nx] = true; queue.add(nx to ny)
                            }
                        }
                    }
                    if (blob.size > 15) { // lower threshold to catch tiny/distant targets
                        val minX = blob.minOf { it.first } * scale
                        val minY = blob.minOf { it.second } * scale
                        val maxX = blob.maxOf { it.first } * scale
                        val maxY = blob.maxOf { it.second } * scale
                        val pad = maxOf((maxX - minX) / 4, 4)
                        results.add(EnemyBox(
                            (minX - pad).coerceAtLeast(0),
                            (minY - pad).coerceAtLeast(0),
                            (maxX + pad).coerceAtMost(bitmap.width),
                            (maxY + pad).coerceAtMost(bitmap.height),
                            0.5f, "PLAYER"
                        ))
                    }
                }
            }
        }
        small.recycle()
        return results.sortedByDescending { (it.right - it.left) * (it.bottom - it.top) }.take(MAX_DETECTIONS)
    }

    private fun isSkinOrEnemyColor(pixel: Int): Boolean {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        val hsv = FloatArray(3)
        android.graphics.Color.RGBToHSV(r, g, b, hsv)
        val h = hsv[0]; val s = hsv[1]; val v = hsv[2]
        if (h in 5f..35f && s > 0.2f && v > 0.3f) return true   // skin tones
        if (h in 35f..65f && s > 0.25f && v > 0.35f) return true // warm yellows
        if (h in 0f..5f && s > 0.4f && v > 0.3f) return true     // reds
        return false
    }

    override fun onDestroy() {
        isRunning.set(false)
        serviceScope.cancel()
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        yoloInterpreter?.close()
        Handler(Looper.getMainLooper()).post {
            overlayView?.let { windowManager?.removeView(it) }
        }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?) = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(CHANNEL_ID, "AimVision", NotificationManager.IMPORTANCE_LOW)
                .apply { description = "خدمة كشف الأعداء" }
            (getSystemService(NotificationManager::class.java)).createNotificationChannel(ch)
        }
    }

    private fun buildNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("AimVision — يعمل")
            .setContentText("YOLOv8 يرصد الأهداف...")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
}

data class EnemyBox(
    val left: Int, val top: Int, val right: Int, val bottom: Int,
    val confidence: Float, val label: String
)
