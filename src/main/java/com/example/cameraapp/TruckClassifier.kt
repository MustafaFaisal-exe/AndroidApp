package com.example.cameraapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.ln
import kotlin.math.sqrt

class TruckClassifier(private val context: Context) {

    companion object {
        private const val TAG = "TruckClassifier"
        const val MODEL_FILE = "truck_classifier_model.onnx"
        private const val NEAR_THRESHOLD = 200f
        private const val HIST_BINS_COARSE = 8
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
                val modelPath = getModelPath()
                env = OrtEnvironment.getEnvironment()
                session = env!!.createSession(modelPath)
                status = Status.READY

                // Detailed model inspection logging
                Log.i(TAG, "Truck classifier model ready")
                session?.let { s ->
                    Log.i(TAG, "Input Names: ${s.inputNames.toList()}")
                    s.inputInfo.forEach { (name, info) ->
                        val tensorInfo = info.info as? ai.onnxruntime.TensorInfo
                        Log.i(TAG, "Input '$name' info: shape=${tensorInfo?.shape?.toList()}, type=${tensorInfo?.type}")
                    }
                    Log.i(TAG, "Output Names: ${s.outputNames.toList()}")
                }
            } catch (e: Exception) {
                status = Status.ERROR
                errorMessage = e.message ?: "Unknown error"
                Log.e(TAG, "Failed to load classifier model", e)
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

    fun extractFeatures(depthMap: FloatArray, rows: Int, cols: Int): FloatArray {
        // 1. Std
        val mean = depthMap.average().toFloat()
        var sumSq = 0f
        for (x in depthMap) sumSq += (x - mean) * (x - mean)
        val std = sqrt(sumSq / depthMap.size)

        // 2. Entropy (32 bins)
        val hist32 = calculateHistogram(depthMap, 32)
        var entropy = 0f
        for (h in hist32) {
            if (h > 1e-9f) entropy -= h * (ln(h) / ln(2f)) // log2
        }

        // 3. Coarse Histogram (8 bins)
        val histCoarse = calculateHistogram(depthMap, HIST_BINS_COARSE)

        // 4. Gradient Variance
        val mat = Mat(rows, cols, CvType.CV_32F)
        mat.put(0, 0, depthMap)
        val matU8 = Mat()
        mat.convertTo(matU8, CvType.CV_8U)

        val gradX = Mat()
        val gradY = Mat()
        Imgproc.Sobel(matU8, gradX, CvType.CV_64F, 1, 0, 3)
        Imgproc.Sobel(matU8, gradY, CvType.CV_64F, 0, 1, 3)

        val gradXsq = Mat()
        val gradYsq = Mat()
        Core.multiply(gradX, gradX, gradXsq)
        Core.multiply(gradY, gradY, gradYsq)

        val gradMagSq = Mat()
        Core.add(gradXsq, gradYsq, gradMagSq)

        val gradMag = Mat()
        Core.sqrt(gradMagSq, gradMag)

        val meanGrad = MatOfDouble()
        val stdGrad = MatOfDouble()
        Core.meanStdDev(gradMag, meanGrad, stdGrad)
        val gradVar = stdGrad.get(0, 0)[0] * stdGrad.get(0, 0)[0]

        // 5. Near Pct
        var nearCount = 0
        for (x in depthMap) if (x > NEAR_THRESHOLD) nearCount++
        val nearPct = nearCount.toFloat() / depthMap.size

        // Clean up
        mat.release(); matU8.release(); gradX.release(); gradY.release(); gradMag.release()
        gradXsq.release(); gradYsq.release(); gradMagSq.release()

        // Assembly: [std, entropy] + histCoarse + [gradVar, nearPct]
        val features = FloatArray(2 + HIST_BINS_COARSE + 2)
        features[0] = std
        features[1] = entropy
        System.arraycopy(histCoarse, 0, features, 2, HIST_BINS_COARSE)
        features[2 + HIST_BINS_COARSE] = gradVar.toFloat()
        features[2 + HIST_BINS_COARSE + 1] = nearPct

        Log.d(TAG, "Extracted Features: ${features.joinToString(", ")}")
        return features
    }

    private fun calculateHistogram(data: FloatArray, bins: Int): FloatArray {
        val hist = FloatArray(bins)
        val binSize = 256f / bins
        for (x in data) {
            val bin = (x / binSize).toInt().coerceIn(0, bins - 1)
            hist[bin]++
        }
        val sum = hist.sum() + 1e-9f
        for (i in hist.indices) hist[i] /= sum
        return hist
    }

    fun predict(features: FloatArray): String {
        val currentSession = session ?: return "ERROR"

        val inputName = currentSession.inputNames.first()
        val floatBuffer = FloatBuffer.wrap(features)
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, features.size.toLong()))

        val outputs = currentSession.run(mapOf(inputName to inputTensor))
        inputTensor.close()

        val labelValue = outputs[0].value
        val prediction = when {
            labelValue is LongArray -> if (labelValue[0] == 1L) "FULL" else "EMPTY"
            labelValue is IntArray -> if (labelValue[0] == 1) "FULL" else "EMPTY"
            labelValue is Array<*> -> labelValue[0].toString()
            else -> "UNKNOWN"
        }
        outputs.close()
        return prediction
    }

    private fun getModelPath(): String {
        val f = File(context.filesDir, MODEL_FILE)
        // Always copy from assets for now to ensure we have the latest/correct model
        context.assets.open(MODEL_FILE).use { input ->
            FileOutputStream(f).use { output ->
                input.copyTo(output)
            }
        }
        return f.absolutePath
    }
}
