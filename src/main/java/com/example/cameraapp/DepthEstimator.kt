package com.example.cameraapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

class DepthEstimator(private val context: Context) {

    companion object {
        private const val TAG = "DepthEstimator"
        const val MODEL_FILE = "depth_model.onnx"
        private const val DEFAULT_INPUT_SIZE = 518
        
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    enum class Status { IDLE, LOADING, READY, ERROR }

    var status = Status.IDLE
        private set
    var errorMessage = ""
        private set

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var inputSize = DEFAULT_INPUT_SIZE

    fun loadAsync(onReady: () -> Unit = {}) {
        if (status == Status.LOADING || status == Status.READY) return
        status = Status.LOADING
        Thread {
            try {
                val modelBytes = loadModelBytes()
                
                // Diagnostic logging
                Log.i(TAG, "Model byte size: ${modelBytes.size}")
                val header = modelBytes.take(16).joinToString("") { "%02x".format(it.toInt() and 0xFF) }
                Log.i(TAG, "Model header (first 16 bytes): $header")

                env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT) // Fix optimizer incompatibility
                    try {
                        addXnnpack(mapOf("intra_op_num_threads" to "2"))
                        Log.i(TAG, "XNNPACK enabled for Depth model")
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to enable XNNPACK: ${e.message}")
                    }
                }
                session = env!!.createSession(modelBytes, opts)
                inputSize = resolveInputSize(session!!)
                status = Status.READY
                Log.i(TAG, "Depth ONNX session ready (inputSize=$inputSize)")
            } catch (e: Exception) {
                status = Status.ERROR
                val msg = e.message ?: ""
                errorMessage = when {
                    msg.contains("Protobuf parsing failed", ignoreCase = true) -> {
                        "Model file appears corrupted or incomplete. Re-download $MODEL_FILE and replace it in assets/, then clean+rebuild."
                    }
                    msg.contains("ORT_NOT_IMPLEMENTED", ignoreCase = true) && msg.contains("ConvInteger", ignoreCase = true) -> {
                        "Quantized model ops (ConvInteger) not supported by this build. Try a non-quantized depth model."
                    }
                    else -> "${e.javaClass.simpleName}: $msg"
                }
                Log.e(TAG, "Failed to load Depth model. Message: $msg, Cause: ${e.cause}", e)
                Log.e(TAG, "Stack trace: ${Log.getStackTraceString(e)}")
            }
            onReady()
        }.start()
    }

    fun isReady() = status == Status.READY

    fun close() {
        session?.close(); session = null
        env?.close(); env = null
        status = Status.IDLE
    }

    fun estimate(bitmap: Bitmap): DepthResult? {
        val currentSession = session ?: return null
        
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputTensor = bitmapToTensor(resized)
        
        val outputs = currentSession.run(mapOf(currentSession.inputNames.first() to inputTensor))
        inputTensor.close()
        
        val outputValue = outputs[0].value
        outputs.close()
        
        // Depth-Anything-V2 output is usually [1, H, W] or [1, 1, H, W]
        @Suppress("UNCHECKED_CAST")
        val depthData: Array<FloatArray> = when (outputValue) {
            is Array<*> -> {
                val first = outputValue.firstOrNull()
                when (first) {
                    is FloatArray -> outputValue as Array<FloatArray>
                    is Array<*> -> {
                        val nested = first.firstOrNull()
                        if (nested is FloatArray) {
                            // Shape [1, H, W] -> outputValue[0] is Array<FloatArray>
                            first as Array<FloatArray>
                        } else if (nested is Array<*>) {
                            // Shape [1, 1, H, W] -> outputValue[0][0] is Array<FloatArray>
                            nested as Array<FloatArray>
                        } else throw Exception("Unexpected output structure")
                    }
                    else -> throw Exception("Unexpected output element type")
                }
            }
            else -> throw Exception("Output is not an array")
        }

        val rows = depthData.size
        val cols = depthData[0].size
        val flatDepth = FloatArray(rows * cols)
        for (i in 0 until rows) {
            System.arraycopy(depthData[i], 0, flatDepth, i * cols, cols)
        }
        
        val min = flatDepth.minOrNull() ?: 0f
        val max = flatDepth.maxOrNull() ?: 0f
        val diff = max - min
        
        val normalized = FloatArray(flatDepth.size)
        if (diff > 0) {
            for (i in flatDepth.indices) {
                normalized[i] = 255f * (flatDepth[i] - min) / diff
            }
        }
        
        // Calculate stats on the normalized [0, 255] map to match Python logic
        val std = calculateStd(normalized)
        val mean = normalized.average().toFloat()
        val heatmap = createHeatmap(normalized, rows, cols, bitmap.width, bitmap.height)
        
        return DepthResult(normalized, mean, diff, std, heatmap, rows, cols)
    }

    fun cropDepthMap(
        result: DepthResult,
        sourceRect: android.graphics.Rect,
        sourceWidth: Int,
        sourceHeight: Int
    ): RegionResult {
        val rows = result.rows
        val cols = result.cols
        
        val x1 = (sourceRect.left.toFloat() / sourceWidth * cols).toInt().coerceIn(0, cols - 1)
        val y1 = (sourceRect.top.toFloat() / sourceHeight * rows).toInt().coerceIn(0, rows - 1)
        val x2 = (sourceRect.right.toFloat() / sourceWidth * cols).toInt().coerceIn(x1 + 1, cols)
        val y2 = (sourceRect.bottom.toFloat() / sourceHeight * rows).toInt().coerceIn(y1 + 1, rows)
        
        val croppedWidth = x2 - x1
        val croppedHeight = y2 - y1
        val croppedData = FloatArray(croppedWidth * croppedHeight)
        
        var sum = 0f
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (y in 0 until croppedHeight) {
            for (x in 0 until croppedWidth) {
                val value = result.depthMap[(y1 + y) * cols + (x1 + x)]
                croppedData[y * croppedWidth + x] = value
                sum += value
                if (value < min) min = value
                if (value > max) max = value
            }
        }
        
        val mean = if (croppedData.isNotEmpty()) sum / croppedData.size else 0f
        val diff = if (croppedData.isNotEmpty()) max - min else 0f
        val std = calculateStd(croppedData)
        
        // Generate heatmap for the cropped region
        val heatmap = createHeatmap(croppedData, croppedHeight, croppedWidth, sourceRect.width(), sourceRect.height())
        
        return RegionResult(mean, diff, std, heatmap)
    }

    data class RegionResult(
        val mean: Float,
        val diff: Float,
        val std: Float,
        val heatmap: Bitmap
    )

    private fun calculateStd(data: FloatArray): Float {
        if (data.isEmpty()) return 0f
        val mean = data.average().toFloat()
        var sum = 0f
        for (x in data) {
            sum += (x - mean) * (x - mean)
        }
        return sqrt(sum / data.size)
    }

    private fun bitmapToTensor(bitmap: Bitmap): OnnxTensor {
        val floatBuffer = FloatBuffer.allocate(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        // R channel
        for (p in pixels) {
            floatBuffer.put(((Color.red(p) / 255f) - MEAN[0]) / STD[0])
        }
        // G channel
        for (p in pixels) {
            floatBuffer.put(((Color.green(p) / 255f) - MEAN[1]) / STD[1])
        }
        // B channel
        for (p in pixels) {
            floatBuffer.put(((Color.blue(p) / 255f) - MEAN[2]) / STD[2])
        }
        floatBuffer.rewind()
        
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
    }

    private fun createHeatmap(normalized: FloatArray, h: Int, w: Int, targetW: Int, targetH: Int): Bitmap {
        val mat = Mat(h, w, CvType.CV_32F)
        mat.put(0, 0, normalized)
        
        val byteMat = Mat()
        mat.convertTo(byteMat, CvType.CV_8U)
        
        val colored = Mat()
        Imgproc.applyColorMap(byteMat, colored, Imgproc.COLORMAP_PLASMA)
        
        val resized = Mat()
        Imgproc.resize(colored, resized, Size(targetW.toDouble(), targetH.toDouble()))
        
        val bmp = ImageProcessor.matToBitmap(resized)
        
        mat.release()
        byteMat.release()
        colored.release()
        resized.release()
        
        return bmp
    }

    private fun resolveInputSize(session: OrtSession): Int {
        return try {
            val inputName = session.inputNames.firstOrNull() ?: return DEFAULT_INPUT_SIZE
            val info = session.inputInfo[inputName]?.info as? ai.onnxruntime.TensorInfo ?: return DEFAULT_INPUT_SIZE
            val shape = info.shape
            Log.i(TAG, "Raw input shape: ${shape.toList()}")
            val size = shape.getOrNull(2)?.toInt() ?: DEFAULT_INPUT_SIZE
            if (size <= 0) {
                Log.w(TAG, "Dynamic input shape detected ($size), falling back to $DEFAULT_INPUT_SIZE")
                DEFAULT_INPUT_SIZE
            } else size
        } catch (e: Exception) {
            Log.e(TAG, "Error resolving input size", e)
            DEFAULT_INPUT_SIZE
        }
    }

    private fun loadModelBytes(): ByteArray {
        val bytes = try {
            context.assets.open(MODEL_FILE).readBytes()
        } catch (e: Exception) {
            val f = File(context.filesDir, MODEL_FILE)
            if (f.exists()) f.readBytes()
            else throw Exception("Place $MODEL_FILE in app/src/main/assets/ and rebuild")
        }

        if (bytes.size < 1_000_000) {
            throw Exception("Model file is too small (${bytes.size} bytes). It is likely corrupted or incomplete.")
        }
        return bytes
    }

    data class DepthResult(
        val depthMap: FloatArray,
        val depthMean: Float,
        val depthDiff: Float,
        val depthStd: Float,
        val heatmap: Bitmap,
        val rows: Int,
        val cols: Int
    )
}
