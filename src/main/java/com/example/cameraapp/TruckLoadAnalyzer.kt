package com.example.cameraapp

import android.graphics.Rect
import org.opencv.core.Mat

class TruckLoadAnalyzer(
    private val yolo: YoloDetector,
    private val depthEstimator: DepthEstimator,
    private val classifier: TruckClassifier
) {

    companion object {
        fun calculateFallbackCrop(
            w: Int,
            h: Int,
            marginLeft: Int,
            marginRight: Int,
            marginTop: Int,
            marginBottom: Int
        ): Rect {
            val x1 = marginLeft.coerceIn(0, w - 1)
            val y1 = marginTop.coerceIn(0, h - 1)
            val x2 = (w - marginRight).coerceIn(x1 + 1, w)
            val y2 = (h - marginBottom).coerceIn(y1 + 1, h)
            return Rect(x1, y1, x2, y2)
        }
    }

    fun analyze(
        frame: Mat,
        marginLeft: Int,
        marginRight: Int,
        marginTop: Int,
        marginBottom: Int
    ): TruckAnalysisResult {
        // 1. Try YOLO
        yolo.infer(frame, includeView = false)
        val yoloTarget = yolo.lastTargetDetection
        
        val cropRect: Rect
        val source: String
        
        if (yoloTarget != null) {
            cropRect = Rect(
                yoloTarget.x1.toInt().coerceIn(0, frame.cols()),
                yoloTarget.y1.toInt().coerceIn(0, frame.rows()),
                yoloTarget.x2.toInt().coerceIn(0, frame.cols()),
                yoloTarget.y2.toInt().coerceIn(0, frame.rows())
            )
            source = "yolo"
        } else {
            cropRect = calculateFallbackCrop(
                frame.cols(), frame.rows(),
                marginLeft, marginRight, marginTop, marginBottom
            )
            source = "fallback"
            
            if (cropRect.width() <= 0 || cropRect.height() <= 0) {
                return TruckAnalysisResult("Invalid margins", false, 0f, 0f, 0f, Rect(0,0,0,0), "fallback", null)
            }
        }

        // 2. Crop Mat and convert to Bitmap for Depth model
        val roi = org.opencv.core.Rect(cropRect.left, cropRect.top, cropRect.width(), cropRect.height())
        val croppedMat = Mat(frame, roi)
        val croppedBitmap = ImageProcessor.matToBitmap(croppedMat)
        croppedMat.release()
        
        // 3. Depth Estimation on the CROPPED image
        val depthResult = depthEstimator.estimate(croppedBitmap) ?: run {
            return TruckAnalysisResult("Depth Error", false, 0f, 0f, 0f, cropRect, source, null)
        }
        
        // 4. Classification using the ML model
        val features = classifier.extractFeatures(depthResult.depthMap, depthResult.rows, depthResult.cols)
        val prediction = classifier.predict(features)
        
        return TruckAnalysisResult(
            status = prediction,
            isFull = prediction == "FULL",
            depthMean = depthResult.depthMean,
            depthDiff = depthResult.depthDiff,
            depthStd = depthResult.depthStd,
            cropRect = cropRect,
            source = source,
            depthMapBitmap = depthResult.heatmap
        )
    }
}
