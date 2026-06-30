package com.example.cameraapp

import android.content.Context
import org.opencv.core.Mat

class FrameProcessor(context: Context) {

    val yolo = YoloDetector(context)
    var brightness: Int = 0
    var contrast: Double = 1.0

    companion object {
        val VALID_BASES = setOf("bc", "pipeline")
        val VALID_OVERLAYS = setOf("yolo")
    }

    fun process(frame: Mat, base: String, overlay: String): Mat {
        val based = applyBase(frame, base)
        val result = applyOverlay(based, overlay)
        if (result !== based) based.release()
        return result
    }

    fun process(frame: Mat, mode: String): Mat {
        val parts = mode.split("+", limit = 2)
        val base    = parts.getOrElse(0) { "bc" }.let { if (it in VALID_BASES)    it else "bc" }
        val overlay = parts.getOrElse(1) { "yolo" }.let { if (it in VALID_OVERLAYS) it else "yolo" }
        return process(frame, base, overlay)
    }

    private fun applyBase(frame: Mat, base: String): Mat = when (base) {
        "pipeline" -> ImageProcessor.preprocess(frame)
        else -> ImageProcessor.preprocessBC(frame, brightness = brightness, contrast = contrast)
    }

    private fun applyOverlay(frame: Mat, overlay: String): Mat {
        // YOLO and Depth are now handled by TruckLoadAnalyzer every 10 frames
        return frame
    }

    fun close() = yolo.close()
}
