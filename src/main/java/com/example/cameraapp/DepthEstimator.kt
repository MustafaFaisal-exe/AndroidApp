package com.example.cameraapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
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
                val modelPath = getModelPath()
                env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                }
                session = env!!.createSession(modelPath, opts)
                inputSize = resolveInputSize(session!!)
                status = Status.READY
                Log.i(TAG, "Depth ONNX session ready (inputSize=$inputSize)")
            } catch (e: Exception) {
                status = Status.ERROR
                errorMessage = e.message ?: "Unknown error"
                Log.e(TAG, "Failed to load Depth model", e)
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

        val resized = if (bitmap.width == inputSize && bitmap.height == inputSize) bitmap
        else Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputTensor = bitmapToTensor(resized)

        val outputs = currentSession.run(mapOf(currentSession.inputNames.first() to inputTensor))
        inputTensor.close()

        val outputValue = outputs[0].value
        outputs.close()

        @Suppress("UNCHECKED_CAST")
        val depthData: Array<FloatArray> = when (outputValue) {
            is Array<*> -> {
                val first = outputValue.firstOrNull()
                when (first) {
                    is FloatArray -> outputValue as Array<FloatArray>
                    is Array<*> -> first as Array<FloatArray>
                    else -> throw Exception("Unexpected output structure")
                }
            }
            else -> throw Exception("Output is not an array")
        }

        val modelRows = depthData.size
        val modelCols = depthData[0].size
        val flatModelDepth = FloatArray(modelRows * modelCols)
        for (i in 0 until modelRows) {
            System.arraycopy(depthData[i], 0, flatModelDepth, i * modelCols, modelCols)
        }

        // Upsample the raw model-resolution depth grid back to the ORIGINAL
        // (cropped) bitmap's width/height, matching the Python training
        // pipeline's torch.nn.functional.interpolate(..., mode="bicubic",
        // size=image.shape[:2]) step. Without this, features are computed on
        // a fixed square grid instead of the crop's real resolution/aspect
        // ratio -- this is why gradient_var and friends don't match Python.
        val rawMat = Mat(modelRows, modelCols, CvType.CV_32F)
        rawMat.put(0, 0, flatModelDepth)
        val upsampledMat = Mat()
        Imgproc.resize(
            rawMat, upsampledMat,
            Size(bitmap.width.toDouble(), bitmap.height.toDouble()),
            0.0, 0.0, Imgproc.INTER_CUBIC
        )
        rawMat.release()

        val rows = bitmap.height
        val cols = bitmap.width
        val flatDepth = FloatArray(rows * cols)
        upsampledMat.get(0, 0, flatDepth)
        upsampledMat.release()

        val min = flatDepth.minOrNull() ?: 0f
        val max = flatDepth.maxOrNull() ?: 0f
        val diff = max - min

        val normalized = FloatArray(flatDepth.size)
        if (diff > 0) {
            for (i in flatDepth.indices) {
                normalized[i] = 255f * (flatDepth[i] - min) / diff
            }
        }

        val mean = normalized.average().toFloat()
        val std = calculateStd(normalized, mean)
        val heatmap = createHeatmap(normalized, rows, cols, bitmap.width, bitmap.height)

        return DepthResult(normalized, mean, diff, std, heatmap, rows, cols)
    }

    private fun calculateStd(data: FloatArray, mean: Float): Float {
        if (data.isEmpty()) return 0f
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
        
        // Match Python normalization
        for (p in pixels) floatBuffer.put(((Color.red(p) / 255f) - MEAN[0]) / STD[0])
        for (p in pixels) floatBuffer.put(((Color.green(p) / 255f) - MEAN[1]) / STD[1])
        for (p in pixels) floatBuffer.put(((Color.blue(p) / 255f) - MEAN[2]) / STD[2])
        
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
            info.shape.getOrNull(2)?.toInt()?.takeIf { it > 0 } ?: DEFAULT_INPUT_SIZE
        } catch (e: Exception) {
            DEFAULT_INPUT_SIZE
        }
    }

    private fun getModelPath(): String {
        val f = File(context.filesDir, MODEL_FILE)
        context.assets.open(MODEL_FILE).use { input ->
            FileOutputStream(f).use { output ->
                input.copyTo(output)
            }
        }
        return f.absolutePath
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
