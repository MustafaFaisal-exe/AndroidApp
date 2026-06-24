package com.example.cameraapp

import android.content.Context
import org.opencv.core.Mat

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
