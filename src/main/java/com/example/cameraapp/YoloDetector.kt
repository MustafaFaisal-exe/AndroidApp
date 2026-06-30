package com.example.cameraapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Color
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.roundToInt

class YoloDetector(private val context: Context) {

    companion object {
        private const val TAG = "YoloDetector"
        const val MODEL_FILE = "yolo11n.onnx"
        private const val DEFAULT_INPUT_SIZE = 320
        private const val YOLO_CONF_THRESH = 0.05f
        private const val YOLO_NMS_THRESH = 0.45f
        private const val TRUCK_CLASS_ID = 7
        private const val BUS_CLASS_ID = 5
        private const val CLASSIFY_EVERY_N = 4
        private const val TARGET_BOX_SHRINK = 0.92f
        private const val MIN_BOX_SHRINK = 0.88f
        private const val MAX_BOX_SHRINK = 0.99f
        private const val CONFIDENCE_SHRINK_WEIGHT = 0.06f
        private const val SMALL_BOX_SIDE_THRESHOLD = 120f
        private const val SMALL_BOX_SHRINK_BONUS = 0.02f

        private const val TRUCK_LABEL = "truck"
        private const val BUS_LABEL = "bus"
    }

    enum class Status { IDLE, LOADING, READY, ERROR }

    var status = Status.IDLE
        private set
    var errorMessage = ""
        private set

    var lastTruckView: TruckView? = null
        private set
    var lastCropWidth: Int = 0
        private set
    var lastCropHeight: Int = 0
        private set
    var lastTargetClassId: Int? = null
        private set
    var lastTargetConfidence: Float? = null
        private set
    var lastTargetDetection: Detection? = null
        private set

    var lastBoxes: List<DetectionOverlay.Box> = emptyList()
        private set

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var frameCounter = 0
    private var cachedView: TruckView = TruckView.REAR

    private var inputSize = DEFAULT_INPUT_SIZE
    private var letterboxBuf: Mat? = null
    private var tensorBuf: FloatBuffer? = null
    private val rgbBuf = Mat()
    private val floatBuf = Mat()
    private val channels = mutableListOf<Mat>()

    fun loadAsync(onReady: () -> Unit = {}) {
        if (status == Status.LOADING || status == Status.READY) return
        status = Status.LOADING
        Thread {
            try {
                val modelBytes = loadModelBytes()
                env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                }
                session = env!!.createSession(modelBytes, opts)
                inputSize = resolveInputSize(session!!)
                status = Status.READY
                Log.i(TAG, "ONNX session ready (inputSize=$inputSize)")
                onReady()
            } catch (e: Exception) {
                status = Status.ERROR
                errorMessage = "${e.javaClass.simpleName}: ${e.message ?: "Unknown error"}"
                Log.e(TAG, "Failed to load ONNX model", e)
            }
        }.start()
    }

    fun isReady() = status == Status.READY

    fun close() {
        session?.close(); session = null
        env?.close(); env = null
        letterboxBuf?.release(); letterboxBuf = null
        rgbBuf.release()
        floatBuf.release()
        channels.forEach { it.release() }
        channels.clear()
        status = Status.IDLE
    }

    fun infer(frame: Mat) {
        val currentSession = session ?: run {
            fallback(frame)
            return
        }

        val (letterboxed, ratio, padX, padY) = letterbox(frame)

        val inputName = currentSession.inputNames.first()
        val inputTensor = matToTensor(letterboxed, inputName)
        val outputs = currentSession.run(inputTensor)
        inputTensor.values.forEach { (it as OnnxTensor).close() }
        val outputValue = outputs[0].value
        outputs.close()

        val detections = parseDetections(outputValue, frame.cols(), frame.rows(), ratio, padX, padY)
        val kept = nms(detections)
        
        frameCounter++
        lastTruckView = null
        lastCropWidth = 0
        lastCropHeight = 0
        lastTargetClassId = null
        lastTargetConfidence = null
        lastTargetDetection = null
        lastBoxes = emptyList()

        if (kept.isEmpty()) return

        val truckBus = kept.filter { it.classId == BUS_CLASS_ID || it.classId == TRUCK_CLASS_ID }
        val bestTruckBus = selectBestTruckBus(truckBus) ?: return

        val cropBox = tightenForCrop(bestTruckBus, frame.cols(), frame.rows())
        val x1 = cropBox.x1.roundToInt().coerceIn(0, frame.cols() - 1)
        val y1 = cropBox.y1.roundToInt().coerceIn(0, frame.rows() - 1)
        val x2 = cropBox.x2.roundToInt().coerceIn(x1 + 1, frame.cols())
        val y2 = cropBox.y2.roundToInt().coerceIn(y1 + 1, frame.rows())
        lastCropWidth = x2 - x1
        lastCropHeight = y2 - y1

        if (frameCounter % CLASSIFY_EVERY_N == 0) {
            val roi  = org.opencv.core.Rect(x1, y1, x2 - x1, y2 - y1)
            val crop = Mat(frame, roi)
            cachedView = FrontDetector.IsFront(crop)
            crop.release()
        }
        val bestView = cachedView

        lastTruckView          = cachedView
        lastTargetClassId      = bestTruckBus.classId
        lastTargetConfidence   = bestTruckBus.conf
        lastTargetDetection    = bestTruckBus

        // Display only the best truck/bus box
        val label = buildTruckLabel(bestTruckBus, bestView)
        val color = viewColorForTruck(bestView)
        val tight = tightenForDisplay(bestTruckBus)
        
        lastBoxes = listOf(
            DetectionOverlay.Box(tight.x1, tight.y1, tight.x2, tight.y2, label, color, 6f)
        )
    }

    private fun viewColorForTruck(view: TruckView): Int = when (view) {
        TruckView.FRONT -> Color.GREEN
        TruckView.REAR -> Color.RED
        TruckView.SIDE -> Color.rgb(255, 165, 0)
    }

    private fun buildTruckLabel(d: Detection, view: TruckView): String {
        val name = if (d.classId == BUS_CLASS_ID) BUS_LABEL else TRUCK_LABEL
        val conf = "${"%.0f".format(d.conf * 100)}%"
        val v = when (view) {
            TruckView.FRONT -> "FRONT"
            TruckView.REAR -> "REAR"
            TruckView.SIDE -> "SIDE"
        }
        return "$name $conf → $v"
    }

    private fun tightenForDisplay(d: Detection): Detection {
        return shrinkBox(d, TARGET_BOX_SHRINK.coerceIn(MIN_BOX_SHRINK, MAX_BOX_SHRINK), null, null)
    }

    private fun tightenForCrop(d: Detection, imageWidth: Int, imageHeight: Int): Detection {
        val x1 = d.x1.coerceIn(0f, imageWidth.toFloat())
        val y1 = d.y1.coerceIn(0f, imageHeight.toFloat())
        val x2 = d.x2.coerceIn(0f, imageWidth.toFloat())
        val y2 = d.y2.coerceIn(0f, imageHeight.toFloat())
        return Detection(x1, y1, x2, y2, d.conf, d.classId)
    }

    fun lastTargetCrop(frameWidth: Int, frameHeight: Int): Detection? {
        val target = lastTargetDetection ?: return null
        return tightenForCrop(target, frameWidth, frameHeight)
    }

    private fun shrinkBox(
        d: Detection,
        factor: Float,
        imageWidth: Int?,
        imageHeight: Int?
    ): Detection {
        val width = (d.x2 - d.x1).coerceAtLeast(1f)
        val height = (d.y2 - d.y1).coerceAtLeast(1f)
        val confidenceTrim = (1f - d.conf).coerceIn(0f, 1f) * CONFIDENCE_SHRINK_WEIGHT
        val smallBoxTrim = if (maxOf(width, height) < SMALL_BOX_SIDE_THRESHOLD) SMALL_BOX_SHRINK_BONUS else 0f
        val effectiveFactor = (factor - confidenceTrim - smallBoxTrim).coerceIn(MIN_BOX_SHRINK, MAX_BOX_SHRINK)
        val cx = (d.x1 + d.x2) / 2f
        val cy = (d.y1 + d.y2) / 2f
        val halfW = width * effectiveFactor / 2f
        val halfH = height * effectiveFactor / 2f

        var x1 = cx - halfW
        var y1 = cy - halfH
        var x2 = cx + halfW
        var y2 = cy + halfH

        if (imageWidth != null && imageHeight != null) {
            x1 = x1.coerceIn(0f, imageWidth.toFloat())
            y1 = y1.coerceIn(0f, imageHeight.toFloat())
            x2 = x2.coerceIn(0f, imageWidth.toFloat())
            y2 = y2.coerceIn(0f, imageHeight.toFloat())
        }

        if (x2 - x1 < 2f) {
            val midX = (x1 + x2) / 2f
            x1 = midX - 1f
            x2 = midX + 1f
        }
        if (y2 - y1 < 2f) {
            val midY = (y1 + y2) / 2f
            y1 = midY - 1f
            y2 = midY + 1f
        }

        if (imageWidth != null && imageHeight != null) {
            x1 = x1.coerceIn(0f, imageWidth.toFloat())
            y1 = y1.coerceIn(0f, imageHeight.toFloat())
            x2 = x2.coerceIn(0f, imageWidth.toFloat())
            y2 = y2.coerceIn(0f, imageHeight.toFloat())
        }

        return Detection(x1, y1, x2, y2, d.conf, d.classId)
    }

    private fun selectBestTruckBus(detections: List<Detection>): Detection? {
        return detections.maxByOrNull { it.conf }
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        if (detections.size <= 1) return detections

        val sorted = detections.sortedByDescending { it.conf }
        val kept = ArrayList<Detection>(sorted.size)
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            val current = sorted[i]
            kept.add(current)
            for (j in i + 1 until sorted.size) {
                if (!suppressed[j] && current.classId == sorted[j].classId && iou(current, sorted[j]) > YOLO_NMS_THRESH) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = maxOf(a.x1, b.x1)
        val iy1 = maxOf(a.y1, b.y1)
        val ix2 = minOf(a.x2, b.x2)
        val iy2 = minOf(a.y2, b.y2)
        if (ix2 <= ix1 || iy2 <= iy1) return 0f

        val inter = (ix2 - ix1) * (iy2 - iy1)
        val aArea = (a.x2 - a.x1) * (a.y2 - a.y1)
        val bArea = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (aArea + bArea - inter)
    }

    private fun decodeCandidate(
        v0: Float,
        v1: Float,
        v2: Float,
        v3: Float,
        normalized: Boolean,
        imgW: Int,
        imgH: Int,
        ratio: Float,
        padX: Int,
        padY: Int,
        conf: Float,
        classId: Int
    ): Detection? {
        val scale = if (normalized) inputSize.toFloat() else 1f
        val cx = v0 * scale
        val cy = v1 * scale
        val bw = v2 * scale
        val bh = v3 * scale

        val fx1 = ((cx - bw / 2f - padX) / ratio).coerceIn(0f, imgW.toFloat())
        val fy1 = ((cy - bh / 2f - padY) / ratio).coerceIn(0f, imgH.toFloat())
        val fx2 = ((cx + bw / 2f - padX) / ratio).coerceIn(0f, imgW.toFloat())
        val fy2 = ((cy + bh / 2f - padY) / ratio).coerceIn(0f, imgH.toFloat())

        val width = fx2 - fx1
        val height = fy2 - fy1
        if (width <= 2f || height <= 2f) return null
        if (width.isNaN() || height.isNaN()) return null
        return Detection(fx1, fy1, fx2, fy2, conf, classId)
    }

    private fun parseDetections(
        outputValue: Any,
        imgW: Int,
        imgH: Int,
        ratio: Float,
        padX: Int,
        padY: Int
    ): List<Detection> {
        val matrix = unpackMatrix(outputValue) ?: return emptyList()
        val rows = matrix.size
        val cols = matrix[0].size
        if (rows < 5 || cols < 5) return emptyList()

        val attributesOnRows = rows <= cols
        val attrCount = if (attributesOnRows) rows else cols
        val detCount = if (attributesOnRows) cols else rows
        val hasObjectness = attrCount >= 85
        val classStart = if (hasObjectness) 5 else 4
        val busIndex = classStart + BUS_CLASS_ID
        val truckIndex = classStart + TRUCK_CLASS_ID
        if (attrCount <= maxOf(busIndex, truckIndex)) return emptyList()

        val detections = ArrayList<Detection>(detCount)
        for (i in 0 until detCount) {
            val v0 = if (attributesOnRows) matrix[0][i] else matrix[i][0]
            val v1 = if (attributesOnRows) matrix[1][i] else matrix[i][1]
            val v2 = if (attributesOnRows) matrix[2][i] else matrix[i][2]
            val v3 = if (attributesOnRows) matrix[3][i] else matrix[i][3]
            val objectness = if (hasObjectness) {
                if (attributesOnRows) matrix[4][i] else matrix[i][4]
            } else {
                1f
            }

            val busScore = if (attributesOnRows) matrix[busIndex][i] else matrix[i][busIndex]
            val truckScore = if (attributesOnRows) matrix[truckIndex][i] else matrix[i][truckIndex]

            val (classId, bestScore) = if (truckScore >= busScore) TRUCK_CLASS_ID to truckScore else BUS_CLASS_ID to busScore
            val conf = bestScore * objectness
            if (conf < YOLO_CONF_THRESH) continue

            val normalized = maxOf(v0, v1, v2, v3) <= 1.5f
            val centerBox = decodeCandidate(v0, v1, v2, v3, normalized, imgW, imgH, ratio, padX, padY, conf, classId)
            if (centerBox != null) {
                detections.add(centerBox)
            }
        }
        return detections
    }

    private fun unpackMatrix(outputValue: Any): Array<FloatArray>? {
        val outer = outputValue as? Array<*> ?: return null
        val first = outer.firstOrNull() ?: return null

        val rows: Array<*> = when (first) {
            is Array<*> -> if (outer.size == 1) first else outer
            is FloatArray -> outer
            else -> return null
        }

        val row0 = rows.firstOrNull() ?: return null
        return when (row0) {
            is FloatArray -> rows.mapNotNull { it as? FloatArray }.toTypedArray()
            is Array<*> -> {
                val nested = rows[0] as? Array<*> ?: return null
                if (nested.firstOrNull() !is FloatArray) return null
                nested.mapNotNull { it as? FloatArray }.toTypedArray()
            }
            else -> null
        }
    }

    private data class LetterboxResult(val mat: Mat, val ratio: Float, val padX: Int, val padY: Int)

    private fun letterbox(src: Mat): LetterboxResult {
        val ratio = minOf(inputSize.toFloat() / src.cols(), inputSize.toFloat() / src.rows())
        val nw = (src.cols() * ratio).roundToInt()
        val nh = (src.rows() * ratio).roundToInt()
        val resized = Mat()
        Imgproc.resize(src, resized, Size(nw.toDouble(), nh.toDouble()))

        val dw = inputSize - nw
        val dh = inputSize - nh
        val padX = Math.round(dw / 2.0 - 0.1).toInt()
        val padY = Math.round(dh / 2.0 - 0.1).toInt()

        val lb = letterboxBuf?.takeIf { it.rows() == inputSize && it.cols() == inputSize }
            ?: Mat(inputSize, inputSize, src.type(), Scalar(114.0, 114.0, 114.0)).also { letterboxBuf = it }
        lb.setTo(Scalar(114.0, 114.0, 114.0))
        resized.copyTo(lb.rowRange(padY, padY + nh).colRange(padX, padX + nw))
        resized.release()
        return LetterboxResult(lb, ratio, padX, padY)
    }

    private fun matToTensor(mat: Mat, inputName: String): Map<String, OnnxTensor> {
        Imgproc.cvtColor(mat, rgbBuf, Imgproc.COLOR_BGR2RGB)
        rgbBuf.convertTo(floatBuf, CvType.CV_32FC3, 1.0 / 255.0)

        val buf = tensorBuf?.takeIf { it.capacity() == 3 * inputSize * inputSize }
            ?: FloatBuffer.allocate(3 * inputSize * inputSize).also { tensorBuf = it }
        buf.clear()

        channels.forEach { it.release() }
        channels.clear()
        Core.split(floatBuf, channels)

        val floatArray = FloatArray(inputSize * inputSize)
        for (ch in channels) {
            ch.get(0, 0, floatArray)
            buf.put(floatArray)
        }
        buf.rewind()

        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
        return mapOf(inputName to tensor)
    }

    data class Detection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val conf: Float,
        val classId: Int
    ) {
        fun scale(factor: Float) = Detection(x1 * factor, y1 * factor, x2 * factor, y2 * factor, conf, classId)
    }

    private fun fallback(frame: Mat) {
        lastBoxes = emptyList()
        lastTruckView = null
        lastTargetClassId = null
        lastTargetConfidence = null
        lastTargetDetection = null
    }

    private fun resolveInputSize(session: OrtSession): Int {
        return try {
            val inputName = session.inputNames.firstOrNull() ?: return DEFAULT_INPUT_SIZE
            val info = session.inputInfo[inputName]?.info as? ai.onnxruntime.TensorInfo ?: return DEFAULT_INPUT_SIZE
            val shape = info.shape
            val h = shape.getOrNull(2)?.takeIf { it > 0 }?.toInt()
            val w = shape.getOrNull(3)?.takeIf { it > 0 }?.toInt()
            when {
                h != null && w != null && h == w -> h
                h != null -> h
                w != null -> w
                else -> DEFAULT_INPUT_SIZE
            }
        } catch (e: Exception) {
            Log.w(TAG, "Falling back to default input size", e)
            DEFAULT_INPUT_SIZE
        }
    }

    private fun loadModelBytes(): ByteArray {
        return try {
            context.assets.open(MODEL_FILE).readBytes()
        } catch (e: Exception) {
            val f = File(context.filesDir, MODEL_FILE)
            if (f.exists()) f.readBytes()
            else throw Exception("Place $MODEL_FILE in app/src/main/assets/ and rebuild")
        }
    }
}
