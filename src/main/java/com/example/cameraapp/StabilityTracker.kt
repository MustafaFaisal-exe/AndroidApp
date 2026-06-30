package com.example.cameraapp

import android.graphics.RectF
import kotlin.math.sqrt

class StabilityTracker(
    private val windowSize: Int = 1,
    private val stableFrameThreshold: Int = 1,
    private val minConfidence: Float = 0.05f,
    private val maxConfStdDev: Float = 0.05f,
    private val maxDriftPixels: Float = 30f
) {
    private val detections = mutableListOf<DetectionEntry?>()

    data class DetectionEntry(
        val confidence: Float,
        val box: RectF,
        val classId: Int
    )

    @Synchronized
    fun addDetection(entry: DetectionEntry?) {
        detections.add(entry)
        if (detections.size > windowSize) {
            detections.removeAt(0)
        }
    }

    @Synchronized
    fun isStable(): Boolean {
        val recentDetections = detections.filterNotNull()
        if (recentDetections.size < stableFrameThreshold) return false

        // Check confidence
        val confidences = recentDetections.map { it.confidence }
        if (confidences.any { it < minConfidence }) return false
        
        val mean = confidences.average().toFloat()
        val stdDev = sqrt(confidences.map { (it - mean) * (it - mean) }.average()).toFloat()
        if (stdDev > maxConfStdDev) return false

        // Check drift between consecutive stable frames (where we have detections)
        // We only check if the detections we have are close to each other
        for (i in 0 until recentDetections.size - 1) {
            val b1 = recentDetections[i].box
            val b2 = recentDetections[i+1].box
            val drift = sqrt(((b1.centerX() - b2.centerX()) * (b1.centerX() - b2.centerX()) +
                             (b1.centerY() - b2.centerY()) * (b1.centerY() - b2.centerY())).toDouble()).toFloat()
            if (drift > maxDriftPixels) return false
        }

        return true
    }

    @Synchronized
    fun getProgress(): Int = detections.filterNotNull().size

    @Synchronized
    fun getMaxProgress(): Int = windowSize

    @Synchronized
    fun reset() {
        detections.clear()
    }
}
