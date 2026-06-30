package com.example.cameraapp

import android.graphics.Bitmap
import android.graphics.Rect

data class TruckAnalysisResult(
    val status: String,
    val isFull: Boolean,
    val depthMean: Float,
    val depthDiff: Float,
    val depthStd: Float,
    val cropRect: Rect,
    val source: String, // "yolo" or "fallback"
    val depthMapBitmap: Bitmap?
)
