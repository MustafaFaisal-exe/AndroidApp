package com.example.cameraapp

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
class YoloDetector(private val context: Context) {

    companion object {
        private const val TAG = "YoloDetector"
        const val MODEL_FILE = "yolo11n.onnx"
        private const val INPUT_SIZE = 320
        private const val CONF_THRESH = 0.35f
        private const val IOU_THRESH = 0.45f

        // COCO class names (subset shown on-screen)
        private val COCO_NAMES = arrayOf(
            "person","bicycle","car","motorcycle","airplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
            "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
            "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
            "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
            "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"
        )
        private val CLASS_COLORS = listOf(
            Scalar(0.0, 255.0, 0.0),    // Pure Green (BGR: B=0, G=255, R=0)
            Scalar(255.0, 100.0, 0.0),  // Bright Blue
            Scalar(0.0, 0.0, 255.0),    // Pure Red
            Scalar(0.0, 255.0, 255.0),  // Yellow
            Scalar(255.0, 0.0, 255.0),  // Magenta
            Scalar(255.0, 255.0, 0.0),  // Cyan
        )
    }

    enum class Status { IDLE, LOADING, READY, ERROR }

    var status = Status.IDLE
        private set
    var errorMessage = ""
        private set

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null

    fun loadAsync(onReady: () -> Unit = {}) {
        if (status == Status.LOADING || status == Status.READY) return
        status = Status.LOADING
        Thread {
            try {
                val modelBytes = loadModelBytes()
                env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(4) // Increase threads for CPU speedup
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                }
                session = env!!.createSession(modelBytes, opts)
                status = Status.READY
                Log.i(TAG, "ONNX session ready: ${session!!.inputNames}")
                onReady()
            } catch (e: Exception) {
                status = Status.ERROR
                errorMessage = e.message ?: "Unknown error"
                Log.e(TAG, "Failed to load ONNX model: $errorMessage", e)
            }
        }.start()
    }

    fun isReady() = status == Status.READY

    fun close() {
        session?.close(); session = null
        env?.close(); env = null
        status = Status.IDLE
    }
    fun detect(frame: Mat): Mat {
        val currentSession = session ?: return fallback(frame)
        val (letterboxed, ratio, padX, padY) = letterbox(frame)
        val inputTensor = matToTensor(letterboxed, currentSession.inputNames.first())
        letterboxed.release()

        val outputs = currentSession.run(inputTensor)
        val output = (outputs[0].value as Array<*>)[0] as Array<*>   // shape [84, 8400]
        outputs.close()

        val detections = parseDetections(output, frame.cols(), frame.rows(), ratio, padX, padY)
        val kept = nms(detections)

        val annotated = frame.clone()
        for (d in kept) drawDetection(annotated, d)
        return annotated
    }

    private data class LetterboxResult(val mat: Mat, val ratio: Float, val padX: Int, val padY: Int)

    private fun letterbox(src: Mat): LetterboxResult {
        val ratio = (INPUT_SIZE.toFloat() / maxOf(src.cols(), src.rows()))
        val nw = (src.cols() * ratio).toInt()
        val nh = (src.rows() * ratio).toInt()
        val resized = Mat()
        Imgproc.resize(src, resized, Size(nw.toDouble(), nh.toDouble()))
        val padX = (INPUT_SIZE - nw) / 2; val padY = (INPUT_SIZE - nh) / 2
        val lb = Mat(INPUT_SIZE, INPUT_SIZE, src.type(), Scalar(114.0, 114.0, 114.0))
        resized.copyTo(lb.rowRange(padY, padY + nh).colRange(padX, padX + nw))
        resized.release()
        return LetterboxResult(lb, ratio, padX, padY)
    }

    private fun matToTensor(mat: Mat, inputName: String): Map<String, OnnxTensor> {
        val rgb = Mat()
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB)
        
        // Use CV_32FC3 for faster processing
        val floatMat = Mat()
        rgb.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
        rgb.release()

        val buf = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val channels = mutableListOf<Mat>()
        Core.split(floatMat, channels)
        floatMat.release()

        val floatArray = FloatArray(INPUT_SIZE * INPUT_SIZE)
        for (ch in channels) {
            ch.get(0, 0, floatArray)
            buf.put(floatArray)
            ch.release()
        }

        buf.rewind()
        val tensor = OnnxTensor.createTensor(env,
            buf, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
        return mapOf(inputName to tensor)
    }

    data class Detection(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val conf: Float, val classId: Int
    )

    private fun parseDetections(
        output: Array<*>, imgW: Int, imgH: Int,
        ratio: Float, padX: Int, padY: Int
    ): List<Detection> {
        val rows = (output[0] as FloatArray).size  // 8400
        val detectionsList = mutableListOf<Detection>()
        
        // Cache the arrays to avoid repetitive casting/indexing in the loop (approx. 8400 iterations)
        val cxArray = output[0] as FloatArray
        val cyArray = output[1] as FloatArray
        val bwArray = output[2] as FloatArray
        val bhArray = output[3] as FloatArray
        
        // Only extract the classes we care about (2: car, 7: truck)
        // Array index = classId + 4
        val carConfArray = output[6] as FloatArray
        val truckConfArray = output[11] as FloatArray
        
        val targetClasses = listOf(
            Pair(2, carConfArray),
            Pair(7, truckConfArray)
        )
        
        for (i in 0 until rows) {
            for ((classId, confArray) in targetClasses) {
                val conf = confArray[i]
                if (conf >= CONF_THRESH) {
                    val cx = cxArray[i]
                    val cy = cyArray[i]
                    val bw = bwArray[i]
                    val bh = bhArray[i]
                    
                    // 1. Convert center/size to corners in model space (0-320)
                    val l = cx - bw / 2f
                    val t = cy - bh / 2f
                    val r = cx + bw / 2f
                    val b = cy + bh / 2f

                    // 2. Un-letterbox: Map from 320x320 back to original image
                    val x1 = (l - padX) / ratio
                    val y1 = (t - padY) / ratio
                    val x2 = (r - padX) / ratio
                    val y2 = (b - padY) / ratio
                    
                    // 3. Clip to image boundaries
                    val fx1 = x1.coerceIn(0f, imgW.toFloat())
                    val fy1 = y1.coerceIn(0f, imgH.toFloat())
                    val fx2 = x2.coerceIn(0f, imgW.toFloat())
                    val fy2 = y2.coerceIn(0f, imgH.toFloat())

                    if (fx2 > fx1 && fy2 > fy1) {
                        detectionsList.add(Detection(fx1, fy1, fx2, fy2, conf, classId))
                    }
                }
            }
        }
        return detectionsList
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.conf }.toMutableList()
        val kept = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size)
        
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            
            for (j in i + 1 until sorted.size) {
                // Class-SPECIFIC NMS: Allow different classes to exist in the same area
                // This satisfies your request to see multiple types (e.g. laptop AND keyboard)
                if (!suppressed[j] && sorted[i].classId == sorted[j].classId && iou(sorted[i], sorted[j]) > IOU_THRESH) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = maxOf(a.x1, b.x1); val iy1 = maxOf(a.y1, b.y1)
        val ix2 = minOf(a.x2, b.x2); val iy2 = minOf(a.y2, b.y2)
        if (ix2 <= ix1 || iy2 <= iy1) return 0f
        val inter = (ix2 - ix1) * (iy2 - iy1)
        val aA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val bA = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (aA + bA - inter)
    }

    private fun drawDetection(mat: Mat, d: Detection) {
        val color = CLASS_COLORS[d.classId % CLASS_COLORS.size]
        
        // 1. Draw the bounding box (rectangle) - thicker for visibility
        Imgproc.rectangle(mat,
            Point(d.x1.toDouble(), d.y1.toDouble()),
            Point(d.x2.toDouble(), d.y2.toDouble()), color, 4)

        // 2. Prepare the label text
        val name = COCO_NAMES.getOrElse(d.classId) { "cls${d.classId}" }
        val label = "$name ${"%.0f%%".format(d.conf * 100)}"
        
        val fontScale = 2.0 // Large font
        val thickness = 2
        val baseline = IntArray(1)
        val sz = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        
        // 3. Position label at TOP CENTER of the box with screen clamping
        val imgW = mat.cols().toDouble()
        val boxWidth = d.x2 - d.x1
        
        // Calculate centered X, but clamp so it doesn't go off-screen
        var tx = d.x1.toDouble() + (boxWidth - sz.width) / 2.0
        tx = tx.coerceIn(2.0, imgW - sz.width - 2.0)

        val ty = d.y1.toDouble().coerceAtLeast(sz.height + 10.0)
        
        // 4. Draw a filled background box for the text
        Imgproc.rectangle(mat,
            Point(tx + 98, ty - sz.height - 4.0),
            Point(tx + sz.width + 98.0, ty + baseline[0].toDouble()), color, -1)
            
        // 5. Draw the text label in black
        Imgproc.putText(mat, label,
            Point(tx + 100.0, ty - 2.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0.0, 0.0, 0.0), thickness)
    }

    private fun fallback(frame: Mat): Mat {
        val out = frame.clone()
        Imgproc.putText(out, "YOLO not ready", Point(10.0, 36.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 200.0, 255.0), 2)
        return out
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
