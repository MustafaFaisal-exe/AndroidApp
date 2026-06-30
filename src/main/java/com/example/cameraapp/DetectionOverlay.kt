package com.example.cameraapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View

/**
 * CustomPainter-style overlay: renders cached detections with letterbox-aware
 * coordinate transformation. Inference runs separately; this only paints.
 */
class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    data class Box(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val label: String,
        val color: Int,
        val strokeWidth: Float = 5f
    ) {
        fun toRectF() = RectF(x1, y1, x2, y2)
    }

    private var imageWidth = 1
    private var imageHeight = 1
    private var scaleType = BoxCoordinateMapper.ScaleType.FILL_CENTER
    private var detections: List<Box> = emptyList()

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        textSize = 32f
        typeface = Typeface.MONOSPACE
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val mappedRect = RectF()

    /** Update cached detections and trigger repaint (like GetX update() + CustomPainter). */
    fun setDetections(
        boxes: List<Box>,
        sourceWidth: Int,
        sourceHeight: Int,
        mapping: BoxCoordinateMapper.ScaleType = scaleType
    ) {
        detections = boxes
        imageWidth = sourceWidth
        imageHeight = sourceHeight
        scaleType = mapping
        invalidate()
    }

    fun clear() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (detections.isEmpty() || imageWidth <= 0 || imageHeight <= 0) return

        val transform = BoxCoordinateMapper.computeTransform(
            imageWidth, imageHeight, width.toFloat(), height.toFloat(), scaleType
        )

        for (box in detections) {
            mappedRect.set(BoxCoordinateMapper.mapRect(box.toRectF(), transform))

            outlinePaint.strokeWidth = box.strokeWidth + 2f
            boxPaint.color = box.color
            boxPaint.strokeWidth = box.strokeWidth

            canvas.drawRect(mappedRect, outlinePaint)
            canvas.drawRect(mappedRect, boxPaint)

            if (box.label.isNotEmpty()) {
                val pad = 4f
                val textW = labelPaint.measureText(box.label)
                val textH = labelPaint.textSize
                labelBgPaint.color = box.color
                canvas.drawRect(
                    mappedRect.left,
                    mappedRect.top - textH - pad * 2,
                    mappedRect.left + textW + pad * 2,
                    mappedRect.top,
                    labelBgPaint
                )
                canvas.drawText(box.label, mappedRect.left + pad, mappedRect.top - pad, labelPaint)
            }
        }
    }
}
