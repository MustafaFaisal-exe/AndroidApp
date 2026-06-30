package com.example.cameraapp

import android.content.Context
import android.graphics.Rect
import org.opencv.core.Mat

class TruckLoadAnalyzer(
    private val context: Context,
    private val yolo: YoloDetector,
    private val depthEstimator: DepthEstimator
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

        /**
         * Logic: if mean depth is high (more yellow), it's full.
         * If mean depth is low (more blue), it's empty.
         */
        fun classifyLoad(depthMean: Float, threshold: Float): Boolean {
            return depthMean > threshold
        }
    }

    fun analyze(
        frame: Mat,
        marginLeft: Int,
        marginRight: Int,
        marginTop: Int,
        marginBottom: Int,
        depthThreshold: Float
    ): TruckAnalysisResult {
        // 1. Try YOLO
        yolo.infer(frame)
        val yoloTarget = yolo.lastTargetDetection
        
        val cropRect: Rect
        val source: String
        
        if (yoloTarget != null) {
            // YOLO detected something
            cropRect = Rect(
                yoloTarget.x1.toInt().coerceIn(0, frame.cols()),
                yoloTarget.y1.toInt().coerceIn(0, frame.rows()),
                yoloTarget.x2.toInt().coerceIn(0, frame.cols()),
                yoloTarget.y2.toInt().coerceIn(0, frame.rows())
            )
            source = "yolo"
        } else {
            // Fallback to center crop
            cropRect = calculateFallbackCrop(
                frame.cols(), frame.rows(),
                marginLeft, marginRight, marginTop, marginBottom
            )
            source = "fallback"
            
            if (cropRect.width() <= 0 || cropRect.height() <= 0) {
                return TruckAnalysisResult("Invalid margins", false, 0f, 0f, 0f, Rect(0,0,0,0), "fallback", null)
            }
        }

        // 2. Run Depth Estimation on the WHOLE frame
        val fullBitmap = ImageProcessor.matToBitmap(frame)
        val depthResult = depthEstimator.estimate(fullBitmap) ?: run {
            val statusMsg = when (depthEstimator.status) {
                DepthEstimator.Status.ERROR -> "Depth Model ERROR"
                DepthEstimator.Status.LOADING -> "Depth Loading..."
                else -> "Depth Not Ready"
            }
            return TruckAnalysisResult(statusMsg, false, 0f, 0f, 0f, cropRect, source, null)
        }
        
        // 3. Crop Depth Map using YOLO box
        val regionResult = depthEstimator.cropDepthMap(
            depthResult,
            cropRect,
            frame.cols(),
            frame.rows()
        )
        
        // 4. Classification based on regional mean
        val isFull = classifyLoad(regionResult.mean, depthThreshold)
        val status = if (isFull) "FULL" else "EMPTY"
        
        return TruckAnalysisResult(
            status = status,
            isFull = isFull,
            depthMean = regionResult.mean,
            depthDiff = regionResult.diff,
            depthStd = regionResult.std,
            cropRect = cropRect,
            source = source,
            depthMapBitmap = regionResult.heatmap
        )
    }
}
