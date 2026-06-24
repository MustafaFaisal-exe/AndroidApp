package com.example.cameraapp

import android.content.Context
import android.os.Environment
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FrameProcessor(context: Context) {

    val yolo = YoloDetector(context)
    var brightness: Int = 0
    var contrast: Double = 1.0

    companion object {
        val VALID_BASES = setOf("raw", "bc", "pipeline")
        val VALID_OVERLAYS = setOf("none", "yolo")
    }

    fun process(frame: Mat, base: String, overlay: String): Mat {
        val based = applyBase(frame, base)
        val result = applyOverlay(based, overlay)
        if (base != "raw" || overlay != "none") {
            if (result !== based) based.release()
        }
        return result
    }

    fun process(frame: Mat, mode: String): Mat {
        val parts = mode.split("+", limit = 2)
        val base    = parts.getOrElse(0) { "raw" }.let { if (it in VALID_BASES)    it else "raw" }
        val overlay = parts.getOrElse(1) { "none" }.let { if (it in VALID_OVERLAYS) it else "none" }
        return process(frame, base, overlay)
    }

    /**
     * Saves the YOLO-cropped region of [frame] to [outputDir].
     * Runs infer() first if no detection is cached yet.
     *
     * @param frame     The full camera frame (BGR Mat).
     * @param outputDir The folder to save into, e.g. File(context.getExternalFilesDir(null), "crops")
     * @param prefix    Optional filename prefix, default "crop".
     * @return          The saved File, or null if no target was detected or save failed.
     */
    fun saveCrop(frame: Mat, outputDir: File, prefix: String = "crop"): File? {
        // Run inference if not done yet for this frame
        if (yolo.lastTargetDetection == null) {
            yolo.infer(frame)
        }

        val crop = yolo.lastTargetCrop(frame.cols(), frame.rows()) ?: return null

        val x1 = crop.x1.toInt().coerceIn(0, frame.cols() - 1)
        val y1 = crop.y1.toInt().coerceIn(0, frame.rows() - 1)
        val x2 = crop.x2.toInt().coerceIn(x1 + 1, frame.cols())
        val y2 = crop.y2.toInt().coerceIn(y1 + 1, frame.rows())

        val roi = Rect(x1, y1, x2 - x1, y2 - y1)
        val cropped = Mat(frame, roi)  // no copy — sub-matrix view

        outputDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val file = File(outputDir, "${prefix}_${timestamp}.jpg")

        val saved = Imgcodecs.imwrite(file.absolutePath, cropped)
        // cropped is a view, not a clone — do NOT release it

        return if (saved) file else null
    }

    private fun applyBase(frame: Mat, base: String): Mat = when (base) {
        "bc" -> ImageProcessor.preprocessBC(frame, brightness = brightness, contrast = contrast)
        "pipeline" -> ImageProcessor.preprocess(frame)
        else -> frame
    }

    private fun applyOverlay(frame: Mat, overlay: String): Mat {
        if (overlay == "yolo") yolo.infer(frame)
        return frame
    }

    fun close() = yolo.close()
}